"""SHARP 360: registration utilities (docs/design/SHARP360.md sections 3.5.2 V2-V4 and 3.5.5).

Pure numpy / OpenCV, CPU only (no scipy).  Used by the TRELLIS and "images" back sources
to line generated or user-supplied material up with SHARP's photographed front.

Building blocks:

* :func:`nelder_mead` - a plain simplex minimiser (reflect / expand / contract / shrink).
  The registration costs below are only piecewise smooth (silhouette overlap counts
  pixels), which rules out gradient methods.
* :func:`umeyama` - closed-form (weighted) similarity between paired points: ``Sigma =
  U D V^T``, ``S = diag(1, 1, det U V^T)``, ``R = U S V^T``, ``s = tr(D S) / var_p``,
  ``t = mu_q - s R mu_p``, so ``det R = +1`` always (splat orientations cannot follow a
  reflection).
* :func:`point_zbuffer` - nearest point per pixel with ``np.minimum.at`` (numpy does not
  guarantee "last write wins" for repeated fancy-index assignments, so a sort followed by a
  scatter would not be safe); points can cover a small disc so sparse clouds do not leak.
* Silhouettes: :func:`silhouette_iou`, :func:`contour_points` (arc-length samples of the
  largest outer contour, starting at the top, clockwise on screen) and
  :func:`chamfer_similarity_fit` / :func:`warp_similarity` (V2: a 2D similarity found by a
  symmetric chamfer distance on ``cv2.distanceTransform`` maps and Nelder-Mead), then
  :func:`chamfer_perspective_fit` (a camera-turn homography on a truncated chamfer: the
  keystone of a photo taken from above the subject's middle) and :func:`outline_gap` (how
  far two outlines still are apart - the back lock's pose check).
* :class:`TPS` / :func:`tps_fit` / :func:`tps_warp` (V3, the back lock): a thin-plate
  spline ``K = r^2 log r`` that maps OUTPUT pixels to SOURCE pixels (a backward map, which
  is what ``cv2.remap`` needs), fitted in normalised coordinates so ``lam`` means the same
  for any image size.
* :func:`projective_icp` and :func:`register_cloud_to_front` (3.5.5): put a free 360 deg
  Gaussian cloud of unknown axis convention and scale (TRELLIS v1 writes a unit cube whose
  up axis is not fixed) behind SHARP's front - a discrete search over the six axis
  directions for "up" times 48 yaws, scored by silhouette overlap and front depth on a
  128-cell grid over the photo, Nelder-Mead refinement of the best three at 192 cells,
  projective ICP (z-buffer winners paired with SHARP's front within a shrinking window;
  point-to-plane similarity steps by default, the design's Umeyama on request), then the
  acceptance gates.  The silhouette / depth scores are built so the true pose scores best
  (supersampled coverage, per-point depth residuals); see :func:`_evaluate`.
* Colour (V4): :func:`color_affine_fit` / :func:`apply_color_affine` (an affine colour
  map in LINEAR RGB - exposure and white balance are linear there - fitted with Huber IRLS
  and a ridge toward the identity, then clamped) and :func:`reinhard_lab` (LAB mean / std
  transfer for views with no pixel correspondence).

Pixel conventions: image-space functions (contours, chamfer, similarity, TPS) use OpenCV
pixel coordinates, pixel ``(row i, col j)`` centred at ``x = j, y = i``, like
``cv2.remap`` / ``cv2.warpAffine``.  :func:`point_zbuffer` takes continuous coordinates
whose pixel index is ``floor(u)`` (design 3.0: pixel centres at ``i + 0.5``), as produced
by ``SharpCam.project`` divided by a cell size.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..core.errors import InputError
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter, SplatCloud
from .mesh import _as_reporter, _check
from .splat import _validate, transform_splats
from .splat_fusion import Canon, FrontModel

log = get_logger(__name__)

# ---- point z-buffer
_MAX_SPLAT_R = 16  # largest disc radius (px) a point may cover in point_zbuffer

# ---- silhouettes
_CHAMFER_MAX_PTS = 1500  # edge points per mask used by the chamfer cost
_CHAMFER_STEP_FRAC = 0.03  # initial simplex translation step: 3 % of the target height
_PERSP_MAX_PTS = 1000  # edge points per mask of the perspective refinement

# ---- thin-plate spline
_TPS_CHUNK = 32_768  # query points per kernel block (x 64 control points ~ 16 MB)
_TPS_EXACT_PX = 256 * 256  # larger outputs evaluate the spline on a 4 px lattice ...
_TPS_LATTICE = 4  # ... and interpolate it bilinearly (the spline is very smooth)

# ---- projective ICP
_ICP_R0_FRAC = 0.08  # first association radius: 8 % of the subject's size ...
_ICP_DECAY = 0.6  # ... shrinking by this factor per iteration ...
_ICP_RMIN_FRAC = 0.01  # ... down to 1 % of the size (and at least 2 cells)
_ICP_MAX_WIN = 10  # search window half-size in (strided) cells
_ICP_MIN_PAIRS = 12
_ICP_MAX_STEP_SCALE = 1.25  # a single step may not scale by more than this (or 1/this)

# ---- cloud registration (design 3.5.5)
_REG_OPACITY = 0.3  # "opaque" centres
_REG_POINTS = 20_000  # subsample of opaque centres (ICP, gates) ...
_REG_COARSE_POINTS = 6_000  # ... of which the discrete search scores this many ...
_REG_REFINE_POINTS = 10_000  # ... and the Nelder-Mead refinement this many
_REG_MIN_POINTS = 200
_REG_YAWS = 48  # 7.5 deg steps
_REG_COARSE_LONG = 128  # z-buffer grid (long side) of the discrete search
_REG_FINE_LONG = 192  # ... and of the refinement / gates
_REG_TOP = 3  # candidates refined
_REG_DISTINCT_DEG = 20.0  # refined candidates differ by at least this rotation
_REG_RESID_REF = 0.10  # m: depth residual normaliser of the score / cost
_REG_TILT_MAX = 15.0  # deg: pitch / roll limit of the refinement
_REG_NM_ITERS = 120
_REG_GATE_IOU = 0.70
_REG_GATE_RESID = 0.05  # m
_REG_GATE_SCALE = (0.5, 2.0)  # times the height-based scale s0
#: Axis names of the "up" search, in search order.
UP_AXES: dict[str, tuple[float, float, float]] = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}

# ---- colour
_COLOR_SIGMA0 = 0.02  # linear-RGB scale of the ridge (see color_affine_fit)
_COLOR_MAX_DEV = 0.35  # ||A - I||_F
_COLOR_MAX_B = 0.12
_COLOR_DIAG = (0.6, 1.6)
_COLOR_MIN_N = 8
_HUBER_K = 1.345
_CHI3_MEDIAN = 1.5382  # median of |r| / sigma for 3-D Gaussian residuals
_HUBER_MIN = 1e-4  # Huber threshold floor (linear RGB): exact data must not divide by 0
_REINHARD_RATIO = (0.5, 2.0)  # clamp of the std ratio
_REINHARD_MIN_PX = 16
_COLOR_CHUNK = 1 << 20  # colours per block in apply_color_affine


# =========================================================================== optimisation
def nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: Any,
    step: Any,
    *,
    max_iter: int = 400,
    tol: float = 1e-7,
    cancel: CancelToken | None = None,
) -> tuple[np.ndarray, float]:
    """Minimise ``f`` with the Nelder-Mead simplex method (standard coefficients: reflect
    1, expand 2, contract 0.5, shrink 0.5).

    The initial simplex is ``x0`` plus ``step[i]`` along each axis (``step`` a scalar or
    one value per parameter; a zero step becomes 5 % of ``|x0[i]|`` or 0.05).  Stops after
    ``max_iter`` iterations or when the spread of the vertex values is ``<= tol`` and every
    vertex is within ``tol`` initial steps of the best one (both, like SciPy's ``xatol`` /
    ``fatol``; with piecewise-constant costs the values agree long before the simplex is
    small).  Non-finite values of ``f`` count as ``+inf``.  Returns the best vertex and its
    value.  ``cancel`` is checked every 16 iterations."""
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    n = len(x)
    if n == 0 or not np.isfinite(x).all():
        raise InputError("Nelder-Mead needs a finite, non-empty start point.")
    st = (
        np.broadcast_to(np.asarray(step, dtype=np.float64).ravel(), (n,)).copy()
        if np.size(step) in (1, n)
        else None
    )
    if st is None or not np.isfinite(st).all():
        raise InputError(f"The Nelder-Mead step must be a number or {n} numbers.")
    zero = st == 0
    st[zero] = np.where(x[zero] != 0, 0.05 * np.abs(x[zero]), 0.05)
    scale = np.abs(st)

    def fe(p: np.ndarray) -> float:
        val = float(f(p))
        return val if math.isfinite(val) else math.inf

    sim = np.vstack([x, x + np.diag(st)])
    fs = np.array([fe(p) for p in sim])
    for it in range(max(0, int(max_iter))):
        order = np.argsort(fs, kind="stable")
        sim, fs = sim[order], fs[order]
        if fs[-1] - fs[0] <= tol and float(np.max(np.abs(sim[1:] - sim[0]) / scale)) <= tol:
            break
        if it % 16 == 15:
            _check(cancel)
        centroid = sim[:-1].mean(axis=0)
        worst = sim[-1]
        xr = centroid + (centroid - worst)
        fr = fe(xr)
        if fr < fs[0]:
            xe = centroid + 2.0 * (centroid - worst)
            fe_ = fe(xe)
            sim[-1], fs[-1] = (xe, fe_) if fe_ < fr else (xr, fr)
            continue
        if fr < fs[-2]:
            sim[-1], fs[-1] = xr, fr
            continue
        if fr < fs[-1]:  # outside contraction
            xc = centroid + 0.5 * (xr - centroid)
            fc = fe(xc)
            if fc <= fr:
                sim[-1], fs[-1] = xc, fc
                continue
        else:  # inside contraction
            xc = centroid + 0.5 * (worst - centroid)
            fc = fe(xc)
            if fc < fs[-1]:
                sim[-1], fs[-1] = xc, fc
                continue
        sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])  # shrink toward the best vertex
        fs[1:] = [fe(p) for p in sim[1:]]
    best = int(np.argmin(fs))
    return sim[best].copy(), float(fs[best])


def umeyama(
    src: Any, dst: Any, *, with_scale: bool = True, weights: Any = None
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity ``dst ~= s * R @ src + t`` between paired ``N x d`` points
    (d = 3 for the callers here; 2 works too), optionally weighted (``weights >= 0``).

    ``R`` is a proper rotation (``det R = +1``): a reflection is never returned, the
    closest rotation is.  ``with_scale=False`` gives the rigid fit (``s = 1``).  Rows with a
    non-finite value (or weight) are ignored.  Raises :class:`InputError` for mismatched
    shapes or fewer than 3 usable pairs.  Coincident source points give ``s = 1, R = I``
    and the translation between the centroids."""
    p = np.asarray(src, dtype=np.float64)
    q = np.asarray(dst, dtype=np.float64)
    if p.ndim != 2 or p.shape != q.shape or p.shape[1] < 2:
        raise InputError(f"Point sets must both be N x d (got {p.shape} and {q.shape}).")
    ok = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    if weights is None:
        w = np.ones(len(p))
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != (len(p),):
            raise InputError(f"Expected {len(p)} weights (got {w.shape}).")
        ok &= np.isfinite(w) & (w > 0)
    p, q, w = p[ok], q[ok], w[ok]
    d = p.shape[1]
    if len(p) < 3:
        raise InputError("At least 3 matching points are needed to align two shapes.")
    W = float(w.sum())
    mu_p = w @ p / W
    mu_q = w @ q / W
    pc, qc = p - mu_p, q - mu_q
    var_p = float(w @ (pc * pc).sum(axis=1)) / W
    if var_p <= 1e-300:
        return 1.0, np.eye(d), mu_q - mu_p
    sigma = (qc * w[:, None]).T @ pc / W
    U, D, Vt = np.linalg.svd(sigma)
    S = np.ones(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1] = -1.0
    R = (U * S) @ Vt
    s = float((D * S).sum() / var_p) if with_scale else 1.0
    t = mu_q - s * (R @ mu_p)
    return s, R, t


# =========================================================================== rasterising
def point_zbuffer(
    u: Any, v: Any, z: Any, shape: tuple[int, ...], *, radius: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest point per pixel: ``(zbuf, idx)``, both ``H x W`` (``shape[:2]``): ``zbuf``
    float64 (``inf`` where no point lands), ``idx`` int64 index of the winning point (-1).

    ``u, v`` are continuous pixel coordinates whose pixel index is ``(floor(v),
    floor(u))``; points with a non-finite coordinate or depth are ignored.  ``radius``
    (pixels, a scalar or one value per point, clipped to 16) makes a point also cover the
    pixels whose index offset ``(dx, dy)`` from its own pixel has ``dx^2 + dy^2 <=
    radius^2`` (also when its own pixel is just outside the image), so a sparse cloud closes
    up instead of letting farther points show through.  Ties in ``z`` go to the lowest point
    index (deterministic).  One radius for all points costs one pass over the points plus a
    min filter (``cv2.erode``) of the image; per-point radii cost one pass per disc offset."""
    H, W = int(shape[0]), int(shape[1])
    uu = np.asarray(u, dtype=np.float64).ravel()
    vv = np.asarray(v, dtype=np.float64).ravel()
    zz = np.asarray(z, dtype=np.float64).ravel()
    if not (len(uu) == len(vv) == len(zz)):
        raise InputError(f"u, v and z must have the same length (got {len(uu)}, {len(vv)}, {len(zz)}).")
    if H < 1 or W < 1:
        raise InputError(f"Invalid z-buffer size {H} x {W}.")
    n = len(zz)
    if radius is None or np.ndim(radius) == 0:
        r = (
            0.0
            if radius is None
            else float(np.clip(np.nan_to_num(float(radius), nan=0.0), 0.0, _MAX_SPLAT_R))
        )
        return _zbuffer_disc(uu, vv, zz, H, W, r, want_idx=True)
    rad = np.asarray(radius, dtype=np.float64).ravel()
    if rad.shape != (n,):
        raise InputError(f"Expected one radius per point ({n}), got {rad.shape}.")
    return _zbuffer_varying(uu, vv, zz, H, W, np.clip(np.nan_to_num(rad, nan=0.0), 0.0, _MAX_SPLAT_R))


def _disc_offsets(r: float) -> list[tuple[int, int]]:
    reach = int(math.floor(r))
    return [
        (dx, dy)
        for dy in range(-reach, reach + 1)
        for dx in range(-reach, reach + 1)
        if dx * dx + dy * dy <= r * r + 1e-9
    ]


def _pixels(
    uu: np.ndarray, vv: np.ndarray, zz: np.ndarray, H: int, W: int, pad: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(flat, z, row)`` of the points inside an image padded by ``pad`` pixels."""
    with np.errstate(invalid="ignore"):
        ok = (
            np.isfinite(zz) & (uu >= -pad) & (uu < W + pad) & (vv >= -pad) & (vv < H + pad)
        )  # NaN coordinates fail the comparisons
    rows = np.flatnonzero(ok)
    xi = np.floor(uu[rows]).astype(np.int64) + pad
    yi = np.floor(vv[rows]).astype(np.int64) + pad
    return yi * (W + 2 * pad) + xi, zz[rows], rows


def _zbuffer_disc(
    uu: np.ndarray, vv: np.ndarray, zz: np.ndarray, H: int, W: int, r: float, *, want_idx: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Same disc for every point: a one-pixel z-buffer on an image padded by the disc's
    reach, then a disc-shaped min filter (``cv2.erode``) - exactly "every point covers its
    disc", at the cost of one image pass instead of one pass over the points per offset."""
    pad = int(math.floor(r))
    Hp, Wp = H + 2 * pad, W + 2 * pad
    flat, zr, rows = _pixels(uu, vv, zz, H, W, pad)
    z0 = np.full(Hp * Wp, np.inf)
    np.minimum.at(z0, flat, zr)
    i0: np.ndarray | None = None
    big = np.iinfo(np.int64).max
    if want_idx:
        i0 = np.full(Hp * Wp, big, np.int64)
        win = zr == z0[flat]
        np.minimum.at(i0, flat[win], rows[win])  # lowest index among exact ties
        i0 = i0.reshape(Hp, Wp)
    z0 = z0.reshape(Hp, Wp)
    if pad == 0:
        zb = z0
        idx = None if i0 is None else np.where(i0 == big, -1, i0)
        return zb, idx
    offs = _disc_offsets(r)
    zfull = cv2.erode(z0, _disc_kernel(r), borderType=cv2.BORDER_CONSTANT, borderValue=np.inf)
    zb = np.ascontiguousarray(zfull[pad : pad + H, pad : pad + W])
    if i0 is None:
        return zb, None
    best = np.full((H, W), big, np.int64)
    for dx, dy in offs:  # which neighbour's point holds the minimum (lowest index on ties)
        zs = z0[pad - dy : pad - dy + H, pad - dx : pad - dx + W]
        js = i0[pad - dy : pad - dy + H, pad - dx : pad - dx + W]
        hit = (zs == zb) & np.isfinite(zb)
        best = np.where(hit, np.minimum(best, js), best)
    return zb, np.where(best == big, -1, best)


def _zbuffer_varying(
    uu: np.ndarray, vv: np.ndarray, zz: np.ndarray, H: int, W: int, rad: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point discs: one ``np.minimum.at`` pass over the points per disc offset."""
    flat0, zr, rows = _pixels(uu, vv, zz, H, W, _MAX_SPLAT_R + 1)
    Wp = W + 2 * (_MAX_SPLAT_R + 1)
    xi = flat0 % Wp - (_MAX_SPLAT_R + 1)
    yi = flat0 // Wp - (_MAX_SPLAT_R + 1)
    r2 = rad[rows] ** 2
    offs = _disc_offsets(float(rad.max(initial=0.0)))
    zbuf = np.full(H * W, np.inf)

    def cells(dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
        px, py = xi + dx, yi + dy
        sel = (r2 >= dx * dx + dy * dy - 1e-9) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        local = np.flatnonzero(sel)
        return py[local] * W + px[local], local

    for dx, dy in offs:  # pass 1: nearest depth per pixel
        flat, local = cells(dx, dy)
        np.minimum.at(zbuf, flat, zr[local])
    big = np.iinfo(np.int64).max
    best = np.full(H * W, big, np.int64)
    for dx, dy in offs:  # pass 2: which point holds it (lowest index among exact ties)
        flat, local = cells(dx, dy)
        win = zr[local] == zbuf[flat]
        np.minimum.at(best, flat[win], rows[local[win]])
    idx = np.where(best == big, -1, best)
    return zbuf.reshape(H, W), idx.reshape(H, W)


# =========================================================================== silhouettes
def _as_mask(m: Any, what: str = "mask") -> np.ndarray:
    a = np.asarray(m)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise InputError(f"The {what} must be an H x W map (got shape {a.shape}).")
    if a.dtype == bool:
        return a
    return np.nan_to_num(a.astype(np.float32), nan=0.0) >= 0.5


def silhouette_iou(a: Any, b: Any) -> float:
    """Intersection over union of two masks of equal shape (bool, or float / uint8 with
    ``>= 0.5`` counting as inside).  Two empty masks give 0.0 (nothing to agree on, so an
    acceptance gate never passes on them)."""
    A = _as_mask(a)
    B = _as_mask(b)
    if A.shape != B.shape:
        raise InputError(f"Silhouettes must have the same size (got {A.shape} and {B.shape}).")
    union = int(np.count_nonzero(A | B))
    return float(np.count_nonzero(A & B)) / union if union else 0.0


def contour_points(mask: Any, n: int = 64) -> np.ndarray:
    """``n`` points ``(x, y)`` (float64, OpenCV pixel coordinates) equally spaced by arc
    length along the largest outer contour of ``mask``, starting at its topmost point (the
    middle one when the top is flat) and running clockwise on screen (image ``y`` points
    down; e.g. along the top edge toward ``+x`` first).  The contour runs through the
    centres of the outermost mask pixels.  Raises :class:`InputError` for an empty mask."""
    try:
        k = int(n)
    except (TypeError, ValueError):
        k = 0
    if k < 3:
        raise InputError(f"At least 3 contour points are needed (got {n!r}).")
    m = _as_mask(mask).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise InputError("The outline is empty, so it has no contour.", hint="Check the subject cut-out.")
    c = max(contours, key=lambda cc: (abs(cv2.contourArea(cc)), len(cc)))
    pts = c.reshape(-1, 2).astype(np.float64)
    if len(pts) == 1:
        return np.repeat(pts, k, axis=0)
    x, y = pts[:, 0], pts[:, 1]
    if float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) < 0:  # positive = clockwise (y down)
        pts = pts[::-1].copy()
    top = np.flatnonzero(pts[:, 1] <= pts[:, 1].min() + 0.5)
    start = int(top[np.argmin(np.abs(pts[top, 0] - np.median(pts[top, 0])))])
    pts = np.roll(pts, -start, axis=0)
    closed = np.vstack([pts, pts[:1]])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    s = np.arange(k) * (cum[-1] / k)
    return np.stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])], axis=1)


def _edge(mask_u8: np.ndarray) -> np.ndarray:
    """Boundary pixels of a 0/1 mask (inside pixels with an outside 8-neighbour; the image
    border counts as outside)."""
    er = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (mask_u8 > 0) & (er == 0)


def _edge_dt(edge: np.ndarray) -> np.ndarray:
    """Distance (px) of every pixel to the nearest edge pixel."""
    return cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(
        np.float64
    )


def _sample_dt(dt: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear lookup of a distance map at continuous OpenCV coordinates; outside the image
    the distance to the image is added (so leaving the frame never looks free)."""
    H, W = dt.shape
    out = np.hypot(x - np.clip(x, 0.0, W - 1.0), y - np.clip(y, 0.0, H - 1.0))
    return _bilinear(dt, x, y) + out


def _rot2(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _similarity_affine(params: dict[str, Any]) -> np.ndarray:
    """2 x 3 matrix ``[A | b]`` of ``p -> s Rot(roll) (p - c) + c + t``."""
    s = float(params.get("s", 1.0))
    roll = float(params.get("roll_deg", 0.0))
    c = np.array([float(params.get("cx", 0.0)), float(params.get("cy", 0.0))])
    t = np.array([float(params.get("tx", 0.0)), float(params.get("ty", 0.0))])
    if not (
        math.isfinite(s) and s > 0 and math.isfinite(roll) and np.isfinite(c).all() and np.isfinite(t).all()
    ):
        raise InputError(f"Invalid similarity parameters {params!r}.")
    A = s * _rot2(roll)
    return np.hstack([A, (c + t - A @ c)[:, None]])


def chamfer_similarity_fit(
    src_mask: Any,
    dst_mask: Any,
    *,
    init: dict[str, Any] | None = None,
    allow_roll: bool = False,
    max_roll_deg: float = 5.0,
    cancel: CancelToken | None = None,
) -> tuple[dict[str, float], float]:
    """2D similarity that lays ``src_mask`` onto ``dst_mask`` (design V2).

    A source pixel ``p`` maps to ``s * Rot(roll) @ (p - c_src) + c_src + t`` in the target
    (OpenCV coordinates, ``c_src`` = the source mask's centroid, positive roll turns
    clockwise on screen).  The cost is the symmetric chamfer distance between the two
    outlines in target pixels: ``mean DT_dst(T p) + s * mean DT_src(T^-1 q)`` over (at most
    1500) boundary pixels ``p`` / ``q`` of each mask, with ``DT_*`` the
    ``cv2.distanceTransform`` to that mask's boundary (bilinear; leaving the image costs the
    distance to it).  Nelder-Mead over ``(log s, tx, ty[, roll])`` starts from ``init`` (a
    previous result, any pivot) or from the bounding boxes (``s`` = height ratio, box
    centres aligned) and is restarted once from its result.  With ``allow_roll`` the roll
    is kept within ``+-max_roll_deg`` (a smooth ``tanh`` bound).

    Returns ``(params, cost)``: ``params = {"s", "tx", "ty", "roll_deg", "cx", "cy"}`` (the
    pivot ``c_src`` as ``cx, cy``; :func:`warp_similarity` applies it), ``cost`` the final
    chamfer distance (px).  Raises :class:`InputError` for an empty mask."""
    S = _as_mask(src_mask, "source mask").astype(np.uint8)
    D = _as_mask(dst_mask, "target mask").astype(np.uint8)
    if not S.any() or not D.any():
        raise InputError("A silhouette is empty, so it cannot be lined up.", hint="Check the cut-outs.")
    max_roll = abs(float(max_roll_deg)) if allow_roll else 0.0
    es, ed = _edge(S), _edge(D)
    dt_s, dt_d = _edge_dt(es), _edge_dt(ed)

    def pts(edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.nonzero(edge)
        k = max(1, int(math.ceil(len(xs) / _CHAMFER_MAX_PTS)))
        return xs[::k].astype(np.float64), ys[::k].astype(np.float64)

    px, py = pts(es)
    qx, qy = pts(ed)
    ys, xs = np.nonzero(S)
    c = np.array([xs.mean(), ys.mean()])
    yd, xd = np.nonzero(D)
    h_s = float(ys.max() - ys.min() + 1)
    h_d = float(yd.max() - yd.min() + 1)
    if init is None:
        s0 = h_d / h_s
        cs = np.array([0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())])
        cd = np.array([0.5 * (xd.min() + xd.max()), 0.5 * (yd.min() + yd.max())])
        t0 = cd - c - s0 * (cs - c)
        r0 = 0.0
    else:
        s0 = float(init.get("s", 1.0))
        r0 = float(init.get("roll_deg", 0.0)) if allow_roll else 0.0
        c_in = np.array([float(init.get("cx", c[0])), float(init.get("cy", c[1]))])
        t_in = np.array([float(init.get("tx", 0.0)), float(init.get("ty", 0.0))])
        # same map about our pivot: t = t' + (I - s Rot) (c' - c)
        t0 = t_in + (np.eye(2) - s0 * _rot2(r0)) @ (c_in - c)
        if not (math.isfinite(s0) and s0 > 0 and np.isfinite(t0).all()):
            raise InputError(f"Invalid initial similarity {init!r}.")

    def unpack(x: np.ndarray) -> tuple[float, np.ndarray, float]:
        roll = max_roll * math.tanh(x[3] / max_roll) if max_roll > 0 and len(x) > 3 else 0.0
        return math.exp(x[0]), x[1:3], roll

    def cost(x: np.ndarray) -> float:
        s, t, roll = unpack(x)
        Rm = _rot2(roll)
        fx = s * (Rm[0, 0] * (px - c[0]) + Rm[0, 1] * (py - c[1])) + c[0] + t[0]
        fy = s * (Rm[1, 0] * (px - c[0]) + Rm[1, 1] * (py - c[1])) + c[1] + t[1]
        bx, by = qx - c[0] - t[0], qy - c[1] - t[1]
        ix = (Rm[0, 0] * bx + Rm[1, 0] * by) / s + c[0]  # Rot^T = Rot(-roll)
        iy = (Rm[0, 1] * bx + Rm[1, 1] * by) / s + c[1]
        return float(_sample_dt(dt_d, fx, fy).mean() + s * _sample_dt(dt_s, ix, iy).mean())

    if max_roll > 0:
        r_in = max(-0.999, min(0.999, r0 / max_roll))
        x0 = np.array([math.log(s0), t0[0], t0[1], max_roll * math.atanh(r_in)])
    else:
        x0 = np.array([math.log(s0), t0[0], t0[1]])
    tstep = max(1.0, _CHAMFER_STEP_FRAC * h_d)
    step = np.array([0.05, tstep, tstep, 2.0][: len(x0)])
    x, fx = nelder_mead(cost, x0, step, max_iter=300, tol=1e-4, cancel=cancel)
    _check(cancel)
    x, fx = nelder_mead(cost, x, 0.5 * step, max_iter=300, tol=1e-4, cancel=cancel)
    s, t, roll = unpack(x)
    params = {
        "s": float(s),
        "tx": float(t[0]),
        "ty": float(t[1]),
        "roll_deg": float(roll),
        "cx": float(c[0]),
        "cy": float(c[1]),
    }
    return params, float(fx)


def _apply_h(H: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x', y', w)`` of points under the 3 x 3 homography ``H`` (``w`` the projective
    denominator; the caller rejects ``w <= 0``, a point sent through the horizon)."""
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    ws = np.where(np.abs(w) > 1e-9, w, 1e-9)
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / ws, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / ws, w


def chamfer_perspective_fit(
    src_mask: Any,
    dst_mask: Any,
    *,
    max_keystone: float = 0.5,
    trunc_fracs: tuple[float, ...] = (0.10, 0.04),
    cancel: CancelToken | None = None,
) -> tuple[np.ndarray, float, float]:
    """Perspective refinement of two outlines already lined up in ONE frame (an addition to
    design V2, after :func:`chamfer_similarity_fit`): the 3 x 3 homography ``H`` that lays
    ``src_mask`` onto ``dst_mask`` (OpenCV coordinates; ``cv2.warpPerspective(src, H)``
    resamples the source into the target).

    A photo taken from above or below the subject's middle (a phone at eye level, tilted
    down) is a keystone of the square-on view: the head is nearer than the feet.  A
    similarity cannot absorb it, so a pose check that measures what is left after the
    similarity calls it a different pose.  For a flat subject a camera turn is exactly the
    homography ``K R K^-1``; to first order that is a similarity plus an aspect change, a
    translation and the two keystone terms, and only those are fitted:
    ``H = T^-1 [[1 + a, -b, tx], [b, 1 + e, ty], [g, h, 1]] T`` in coordinates normalised by
    the target's bounding box (centre, long side), ``g`` and ``h`` held within
    ``+-max_keystone`` by a smooth ``tanh`` bound.  A general homography (shear too) would
    also bend a raised arm onto a lowered one - a different pose must stay visible.

    The cost is the symmetric chamfer distance of :func:`chamfer_similarity_fit` (both masks
    in one frame, so no scale factor) with each distance truncated at ``trunc_fracs[k]``
    times the target's long side, one Nelder-Mead stage per value (the last restarted once): a
    robust fit follows the parts that match, so an arm in another pose does not pull the
    whole body toward it.  A mapping that sends a sample through the horizon costs ``inf``.
    Returns ``(H, cost, cost at the identity)`` (both with the last truncation; the
    identity when nothing better was found); raises :class:`InputError` for an empty mask
    or shapes that differ."""
    S = _as_mask(src_mask, "source mask").astype(np.uint8)
    D = _as_mask(dst_mask, "target mask").astype(np.uint8)
    if S.shape != D.shape:
        raise InputError(f"Both outlines must be in one frame (got {S.shape} and {D.shape}).")
    if not S.any() or not D.any():
        raise InputError("A silhouette is empty, so it cannot be lined up.", hint="Check the cut-outs.")
    k = abs(float(max_keystone))
    fracs = [float(f) for f in trunc_fracs] or [math.inf]
    es, ed = _edge(S), _edge(D)
    dt_s, dt_d = _edge_dt(es).astype(np.float32), _edge_dt(ed).astype(np.float32)

    def pts(edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.nonzero(edge)
        st = max(1, int(math.ceil(len(xs) / _PERSP_MAX_PTS)))
        return xs[::st].astype(np.float64), ys[::st].astype(np.float64)

    def sample(dt: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """:func:`_sample_dt` through ``cv2.remap`` (several times faster than the numpy
        lookup; its 1/32 px weights are plenty for a cost truncated at a few pixels): the
        distance map clamped at the border plus the distance to the image outside it."""
        H, W = dt.shape
        out = np.hypot(x - np.clip(x, 0.0, W - 1.0), y - np.clip(y, 0.0, H - 1.0))
        mx = np.clip(x, -1.0, W).astype(np.float32).reshape(1, -1)
        my = np.clip(y, -1.0, H).astype(np.float32).reshape(1, -1)
        v = cv2.remap(dt, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return v.ravel().astype(np.float64) + out

    px, py = pts(es)
    qx, qy = pts(ed)
    yd, xd = np.nonzero(D)
    c = np.array([0.5 * (xd.min() + xd.max()), 0.5 * (yd.min() + yd.max())])
    L = float(max(xd.max() - xd.min(), yd.max() - yd.min()) + 1)
    T = np.array([[1.0 / L, 0.0, -c[0] / L], [0.0, 1.0 / L, -c[1] / L], [0.0, 0.0, 1.0]])
    Ti = np.linalg.inv(T)

    def unpack(x: np.ndarray) -> np.ndarray:
        g = k * math.tanh(x[5] / k) if k > 0 else 0.0
        h = k * math.tanh(x[6] / k) if k > 0 else 0.0
        Hn = np.array([[1.0 + x[0], -x[1], x[2]], [x[1], 1.0 + x[3], x[4]], [g, h, 1.0]])
        return Ti @ Hn @ T

    def cost_at(tau: float) -> Callable[[np.ndarray], float]:
        def cost(x: np.ndarray) -> float:
            H = unpack(x)
            try:
                Hi = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                return math.inf
            fx, fy, w = _apply_h(H, px, py)
            bx, by, wb = _apply_h(Hi, qx, qy)
            if (w <= 0).any() or (wb <= 0).any():
                return math.inf
            d1 = np.minimum(sample(dt_d, fx, fy), tau)
            d2 = np.minimum(sample(dt_s, bx, by), tau)
            return float(d1.mean() + d2.mean())

        return cost

    x = np.zeros(7)
    step = np.array([0.02, 0.02, 0.01, 0.02, 0.01, 0.05, 0.05])
    for i, f in enumerate(fracs):
        cost = cost_at(f * L if math.isfinite(f) and f > 0 else math.inf)
        x, _ = nelder_mead(cost, x, step, max_iter=600, tol=1e-4, cancel=cancel)
        _check(cancel)
        if i == len(fracs) - 1:  # the last (tightest) stage is restarted once
            x, _ = nelder_mead(cost, x, 0.5 * step, max_iter=600, tol=1e-4, cancel=cancel)
    fx, c0 = cost(x), cost(np.zeros(7))
    if not fx < c0:  # never worse than leaving the outlines as they are
        return np.eye(3), float(c0), float(c0)
    return unpack(x), float(fx), float(c0)


def outline_gap(a: Any, b: Any, pct: float = 98.0) -> float:
    """How far apart two outlines in one frame are (px): the ``pct`` percentile of the
    distances from each mask's boundary pixels to the other mask's boundary, the larger of
    the two directions (``inf`` when a mask is empty).  After a global fit this is what a
    lock would still have to move whatever the pairing, so it is the pose check of the back
    lock: a lazy nearest-point pairing makes the spline's own move small even for an arm in
    another place."""
    A = _as_mask(a).astype(np.uint8)
    B = _as_mask(b).astype(np.uint8)
    if A.shape != B.shape:
        raise InputError(f"Both outlines must be in one frame (got {A.shape} and {B.shape}).")
    if not A.any() or not B.any():
        return math.inf
    ea, eb = _edge(A), _edge(B)
    da, db = _edge_dt(ea), _edge_dt(eb)
    q = float(pct)
    return float(max(np.percentile(db[ea], q), np.percentile(da[eb], q)))


_CV_NATIVE = (np.uint8, np.uint16, np.int16, np.float32)  # resampled as they are


def _cv_apply(img: Any, fn: Callable[[np.ndarray, Any], np.ndarray], border: Any) -> np.ndarray:
    """Run an OpenCV resampler ``fn(image, border_value)`` on any image: bool masks go
    through uint8 (a truthy ``border`` is 255, the result is thresholded at 128), float64 /
    float16 through float32 and back, other dtypes come back as float32; more than 4
    channels are resampled in groups of 4."""
    a = np.asarray(img)
    if a.ndim not in (2, 3) or a.size == 0:
        raise InputError(f"Expected an H x W or H x W x C image (got shape {a.shape}).")
    if a.dtype == bool:
        return _cv_apply(a.astype(np.uint8) * 255, fn, 255.0 if np.any(border) else 0.0) >= 128
    if a.dtype in (np.float64, np.float16):
        return _cv_apply(a.astype(np.float32), fn, border).astype(a.dtype)
    if a.dtype not in _CV_NATIVE:
        a = a.astype(np.float32)
    if a.ndim == 3 and a.shape[2] > 4:
        bv = (
            np.broadcast_to(np.asarray(border, dtype=np.float64).ravel(), (a.shape[2],))
            if np.ndim(border)
            else None
        )
        parts = [
            _cv_apply(a[..., i : i + 4], fn, border if bv is None else bv[i : i + 4])
            for i in range(0, a.shape[2], 4)
        ]
        return np.concatenate([p if p.ndim == 3 else p[..., None] for p in parts], axis=2)
    bv_cv = float(border) if np.ndim(border) == 0 else tuple(float(v) for v in np.ravel(border))
    out = fn(np.ascontiguousarray(a), bv_cv)
    if a.ndim == 3 and out.ndim == 2:
        out = out[..., None]
    return out


def warp_similarity(
    img: Any,
    params: dict[str, Any],
    out_shape: tuple[int, ...],
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: Any = 0,
) -> np.ndarray:
    """Resample ``img`` (source frame) into the target frame of a
    :func:`chamfer_similarity_fit` result (``cv2.warpAffine``); ``out_shape[:2]`` is the
    target ``(H, W)``.  Works for bool masks, uint8 / float images and any channel count;
    pixels from outside the source get ``border_value``.  Missing ``cx, cy`` mean a pivot at
    the origin."""
    H, W = int(out_shape[0]), int(out_shape[1])
    if H < 1 or W < 1:
        raise InputError(f"Invalid output size {H} x {W}.")
    M = _similarity_affine(params)
    return _cv_apply(
        img,
        lambda a, bv: cv2.warpAffine(
            a, M, (W, H), flags=int(interpolation), borderMode=cv2.BORDER_CONSTANT, borderValue=bv
        ),
        border_value,
    )


# =========================================================================== thin-plate spline
def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    """``U = r^2 log r`` from squared distances (0 at r = 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r2 > 0, 0.5 * r2 * np.log(np.maximum(r2, 1e-300)), 0.0)


@dataclass
class TPS:
    """Thin-plate spline from the TARGET (output) image to the SOURCE image: ``tps(q)`` is
    where output pixel ``q`` samples the source (OpenCV coordinates).

    Stored in normalised coordinates ``(q - center) / scale`` (``center`` / ``scale`` =
    the mean and RMS radius of the target control points): ``f(q) = a0 + a1 x + a2 y +
    sum_i w_i U(|q - c_i|)``, ``U = r^2 log r``."""

    ctrl: np.ndarray  # (n, 2) normalised target control points
    w: np.ndarray  # (n, 2) kernel weights
    a: np.ndarray  # (3, 2) affine part [1, x, y] -> (x', y') (normalised)
    center: np.ndarray  # (2,)
    scale: float

    def __call__(self, pts: Any) -> np.ndarray:
        """Source positions (``M x 2``) of target points ``pts`` (``M x 2`` or ``...x2``)."""
        q = np.asarray(pts, dtype=np.float64)
        shape = q.shape
        q = q.reshape(-1, 2)
        out = np.empty_like(q)
        c2 = (self.ctrl**2).sum(axis=1)
        for a in range(0, len(q), _TPS_CHUNK):
            qn = (q[a : a + _TPS_CHUNK] - self.center) / self.scale
            # |q - c|^2 expanded: M x n floats instead of an M x n x 2 difference array
            d2 = np.maximum((qn**2).sum(axis=1)[:, None] + c2[None, :] - 2.0 * (qn @ self.ctrl.T), 0.0)
            val = _tps_kernel(d2) @ self.w + self.a[0] + qn @ self.a[1:]
            out[a : a + _TPS_CHUNK] = val * self.scale + self.center
        return out.reshape(shape)

    def max_displacement(self, shape: tuple[int, ...], *, mask: Any = None) -> float:
        """Largest ``|tps(q) - q|`` (px) over a lattice of about 65 x 65 points spanning an
        image of ``shape[:2]``, plus the control points.

        With ``mask`` (``shape[:2]``, e.g. the target silhouette) only lattice points inside
        it count (at most 4096 of its pixels when the lattice misses it): a spline that
        interpolates contour points can swing far in empty background between distant
        parts (beside the legs, under a raised arm) where nothing is sampled, so a "pose too
        different" gate should look at the subject only."""
        H, W = int(shape[0]), int(shape[1])
        xs = np.unique(np.append(np.linspace(0.0, max(W - 1, 0), min(W, 65)), max(W - 1, 0)))
        ys = np.unique(np.append(np.linspace(0.0, max(H - 1, 0), min(H, 65)), max(H - 1, 0)))
        gx, gy = np.meshgrid(xs, ys)
        lat = np.stack([gx.ravel(), gy.ravel()], axis=1)
        if mask is not None:
            m = _as_mask(mask)
            if m.shape != (H, W):
                raise InputError(f"The mask ({m.shape}) must match the shape ({H}, {W}).")
            lat = lat[m[lat[:, 1].astype(np.int64), lat[:, 0].astype(np.int64)]]
            if not len(lat):
                yy, xx = np.nonzero(m)
                k = max(1, int(math.ceil(len(xx) / 4096)))
                lat = np.stack([xx[::k], yy[::k]], axis=1).astype(np.float64)
        q = np.vstack([lat, self.ctrl * self.scale + self.center])
        return float(np.linalg.norm(self(q) - q, axis=1).max())


def tps_fit(src_pts: Any, dst_pts: Any, *, lam: float = 1e-3) -> TPS:
    """Thin-plate spline with ``tps(dst_pts[i]) ~= src_pts[i]`` (a backward map for
    :func:`tps_warp`): the ``(n+3) x (n+3)`` system ``[[K + lam I, P], [P^T, 0]] [w; a] =
    [src; 0]``, ``K_ij = U(|d_i - d_j|)``, ``P = [1, x, y]``, solved in coordinates
    normalised by the target points' mean and RMS radius (so ``lam`` - smoothing, 0 =
    exact interpolation - does not depend on the image size).  Points are ``n x 2``
    ``(x, y)`` with ``n >= 3``; raises :class:`InputError` otherwise."""
    s = np.asarray(src_pts, dtype=np.float64)
    d = np.asarray(dst_pts, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 2 or s.shape != d.shape or len(s) < 3:
        raise InputError(f"TPS needs two n x 2 point lists with n >= 3 (got {s.shape} and {d.shape}).")
    if not (np.isfinite(s).all() and np.isfinite(d).all()):
        raise InputError("TPS control points must be finite.")
    lam_v = float(lam)
    if not (math.isfinite(lam_v) and lam_v >= 0):
        raise InputError(f"The TPS smoothing must be >= 0 (got {lam!r}).")
    center = d.mean(axis=0)
    scale = float(np.sqrt(((d - center) ** 2).sum(axis=1).mean()))
    scale = scale if scale > 1e-9 else 1.0
    dn = (d - center) / scale
    sn = (s - center) / scale
    n = len(dn)
    K = _tps_kernel(((dn[:, None, :] - dn[None, :, :]) ** 2).sum(-1))
    P = np.hstack([np.ones((n, 1)), dn])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K + lam_v * np.eye(n)
    L[:n, n:] = P
    L[n:, :n] = P.T
    rhs = np.zeros((n + 3, 2))
    rhs[:n] = sn
    try:
        sol = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:  # e.g. all control points on a line: least-squares instead
        sol = np.linalg.lstsq(L, rhs, rcond=None)[0]
    return TPS(ctrl=dn, w=sol[:n], a=sol[n:], center=center, scale=scale)


def tps_warp(
    img: Any,
    tps: TPS,
    out_shape: tuple[int, ...],
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: Any = 0,
) -> np.ndarray:
    """Output pixel ``q`` (``out_shape[:2]``) samples ``img`` at ``tps(q)`` with
    ``cv2.remap`` (bool / uint8 / float, any channel count).  Outputs above 256 x 256
    pixels evaluate the spline on a 4 px lattice and interpolate it bilinearly (sub-0.01 px
    difference for the smooth warps used here)."""
    H, W = int(out_shape[0]), int(out_shape[1])
    if H < 1 or W < 1:
        raise InputError(f"Invalid output size {H} x {W}.")
    if H * W <= _TPS_EXACT_PX:
        gy, gx = np.mgrid[0:H, 0:W]
        m = tps(np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64))
        mx = m[:, 0].reshape(H, W).astype(np.float32)
        my = m[:, 1].reshape(H, W).astype(np.float32)
    else:
        k = _TPS_LATTICE
        hc, wc = (H - 1) // k + 2, (W - 1) // k + 2
        gy, gx = np.mgrid[0:hc, 0:wc]
        m = tps(np.stack([gx.ravel() * k, gy.ravel() * k], axis=1).astype(np.float64))
        cx = m[:, 0].reshape(hc, wc).astype(np.float32)
        cy = m[:, 1].reshape(hc, wc).astype(np.float32)
        fy, fx = np.mgrid[0:H, 0:W].astype(np.float32) / np.float32(k)
        mx = cv2.remap(cx, fx, fy, cv2.INTER_LINEAR)
        my = cv2.remap(cy, fx, fy, cv2.INTER_LINEAR)
    return _cv_apply(
        img,
        lambda a, bv: cv2.remap(
            a, mx, my, int(interpolation), borderMode=cv2.BORDER_CONSTANT, borderValue=bv
        ),
        border_value,
    )


# =========================================================================== 3D registration
def _rot_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, 0.5 * (float(np.trace(R)) - 1.0)))))


@dataclass
class _Targets:
    """SHARP's front as ICP targets on the mould grid."""

    F: np.ndarray  # (Hm*Wm, 3) unprojected front points ray(cell) * D_f (NaN where invalid)
    N: np.ndarray  # (Hm*Wm, 3) unit front normals (facing the camera)
    valid: np.ndarray  # (Hm, Wm) bool: matte cells with a depth and a normal
    cell_m: float  # metric size of one cell at the median front depth
    size_m: float  # longer side of the matte's bounding box (m)


def _front_targets(front: FrontModel) -> _Targets:
    matte = np.asarray(front.matte, dtype=bool)
    D = np.asarray(front.D_f, dtype=np.float64)
    N = np.asarray(front.normals, dtype=np.float64)
    if matte.shape != D.shape or N.shape != (*D.shape, 3):
        raise InputError(f"The front maps do not match ({matte.shape}, {D.shape}, {N.shape}).")
    valid = matte & np.isfinite(D) & (D > 0) & np.isfinite(N).all(axis=-1)
    if not valid.any():
        raise InputError("SHARP's front is empty, so nothing can be lined up with it.")
    u, v = front.grid.centers()
    F = front.cam.rays(u, v) * np.where(valid, D, np.nan)[..., None]
    cell_m = float(np.median(D[valid])) * float(front.grid.cell_px) / float(front.cam.f)
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    size_m = max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1) * cell_m
    return _Targets(F.reshape(-1, 3), N.reshape(-1, 3), valid, cell_m, size_m)


def _icp_pairs(
    X: np.ndarray, front: FrontModel, tg: _Targets, kv: int, radius_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Projective association: the points of ``X`` that win a z-buffer cell (``kv x kv``
    mould cells) are paired with the nearest unprojected front point within ``radius_m``,
    searched in a disc of mould cells around their projection (every ``ks``-th cell when
    the disc would be wider than 10 cells).  Returns ``(i, q, n, d)``: point indices,
    partner points, the front normals there, 3D distances."""
    grid, cam = front.grid, front.cam
    Hm, Wm = tg.valid.shape
    u, v, z = cam.project(X)
    gx = u / grid.cell_px
    gy = v / grid.cell_px
    with np.errstate(invalid="ignore"):
        gx = np.where(z > 1e-6, gx, np.nan)
    hv, wv = -(-Hm // kv), -(-Wm // kv)
    _, idx = point_zbuffer(gx / kv, gy / kv, z, (hv, wv))
    win = idx[idx >= 0]
    col = np.floor(gx[win]).astype(np.int64)
    row = np.floor(gy[win]).astype(np.int64)
    p = X[win]
    ks = max(1, int(math.ceil(radius_m / (tg.cell_m * _ICP_MAX_WIN))))
    R = max(1, int(math.ceil(radius_m / (tg.cell_m * ks))))
    best_d = np.full(len(win), np.inf)
    best_c = np.zeros(len(win), np.int64)
    vflat = tg.valid.ravel()
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            if dr * dr + dc * dc > R * R + R:  # a disc, slightly rounded out
                continue
            rr = row + dr * ks
            cc = col + dc * ks
            sel = np.flatnonzero((rr >= 0) & (rr < Hm) & (cc >= 0) & (cc < Wm))
            cell = rr[sel] * Wm + cc[sel]
            ok = vflat[cell]
            sel, cell = sel[ok], cell[ok]
            if not len(sel):
                continue
            d = np.linalg.norm(p[sel] - tg.F[cell], axis=1)
            better = d < best_d[sel]
            best_d[sel[better]] = d[better]
            best_c[sel[better]] = cell[better]
    keep = best_d <= radius_m
    c = best_c[keep]
    return win[keep], tg.F[c], tg.N[c], best_d[keep]


def _plane_step(p: np.ndarray, q: np.ndarray, n: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """One Gauss-Newton step of the point-to-plane similarity: minimise ``sum (n_i . (p_i'
    - q_i))^2`` with ``p' = (1 + sigma) R(omega) (p - mu) + mu + t`` linearised in
    ``(omega, t, sigma)`` (7 x 7 normal equations, lever arms normalised by their RMS so
    the unknowns are comparable, a tiny Levenberg damping for unobservable directions)."""
    mu = p.mean(axis=0)
    a = p - mu
    L = max(float(np.sqrt((a * a).sum(axis=1).mean())), 1e-12)
    an = a / L
    J = np.hstack([np.cross(an, n), n, (n * an).sum(axis=1, keepdims=True)])
    r0 = ((p - q) * n).sum(axis=1)
    H = J.T @ J
    H += (1e-6 * float(np.trace(H)) / 7.0 + 1e-300) * np.eye(7)
    x = -np.linalg.solve(H, J.T @ r0)
    w = x[:3] / L  # rotation vector (the lever arm a = an * L)
    s = 1.0 + x[6] / L
    th = float(np.linalg.norm(w))
    if th > 1e-15:
        k = w / th
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        Rm = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)
    else:
        Rm = np.eye(3)
    return s, Rm, mu + x[3:6] - s * (Rm @ mu)


def projective_icp(
    P: Any,
    front: FrontModel,
    *,
    iters: int = 8,
    trim: float = 0.3,
    max_rot_deg: float = 10.0,
    metric: str = "plane",
    cancel: CancelToken | None = None,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Refine a similarity that moves points ``P`` (``N x 3``, frame C) onto SHARP's front
    surface: returns ``(s, R, t, diag)`` with ``s * R @ P + t`` on the front.

    Per iteration: transform, project with ``front.cam`` and keep the z-buffer winners
    (the points the photo's camera sees; the z-buffer cell is ``kv x kv`` mould cells with
    ``kv`` chosen so about two points land per cell), pair each with the nearest front
    point ``ray(cell) * D_f(cell)`` in a window of mould cells around its projection
    (radius 8 % of the subject's size, shrinking by 0.6 per iteration to ``max(1 %, 2
    cells)``; pairs farther apart are dropped), drop the worst ``trim`` share of the
    residuals and fit the similarity update.  A step that rotates more than
    ``max_rot_deg`` (or scales by more than 1.25x) is rejected and ends the loop; a tiny
    step ends it early.

    ``metric="plane"`` (default) fits the update to the point-to-PLANE residuals ``n .
    (p - q)`` (front normals ``front.normals``, one linearised Gauss-Newton step per
    iteration); ``"point"`` is the design's trimmed :func:`umeyama` on the point pairs.
    Deviations from the design, both measured on the synthetic person: (1) pairing along
    the point's own camera ray cannot see a sideways offset, so the window search
    generalises it (a window of 0 is the design's rule); (2) a person is thin along the
    view, so closest-point pairs pin the points sideways and point-to-point Umeyama
    corrects a 5 % scale error by only ~4 % of it per iteration (still 4 % off after 20),
    and the coarse grid samples bias it by ~1 %; point-to-plane slides along the surface
    and converges from 5 deg / 5 % / 3 cm to < 0.2 % in 8 iterations, without that bias.

    ``diag``: ``iters`` run, ``metric``, ``kv``, ``pairs``, ``resid_m`` (median
    point-to-plane distance) and ``dist_m`` (median distance to the nearest front sample)
    after the final transform, ``rejected``, ``converged``.  Raises :class:`InputError`
    for malformed points or an empty front."""
    if metric not in ("plane", "point"):
        raise InputError(f"Unknown ICP metric {metric!r} (expected 'plane' or 'point').")
    X0 = np.asarray(P, dtype=np.float64)
    if X0.ndim != 2 or X0.shape[1] != 3:
        raise InputError(f"ICP needs N x 3 points (got shape {X0.shape}).")
    X0 = X0[np.isfinite(X0).all(axis=1)]
    tg = _front_targets(front)
    n_valid = int(tg.valid.sum())
    kv = max(1, int(round(math.sqrt(n_valid / max(len(X0) / 4.0, 1.0)))))
    r0 = _ICP_R0_FRAC * tg.size_m
    r_min = max(2.0 * tg.cell_m, _ICP_RMIN_FRAC * tg.size_m)
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    diag: dict[str, Any] = {"iters": 0, "metric": metric, "rejected": False, "converged": False, "kv": kv}
    trim_v = min(max(float(trim), 0.0), 0.9)
    for k in range(max(0, int(iters)) if len(X0) >= _ICP_MIN_PAIRS else 0):
        _check(cancel)
        radius = max(r_min, r0 * _ICP_DECAY**k)
        X = s * (X0 @ R.T) + t
        i, q, nrm, d = _icp_pairs(X, front, tg, kv, radius)
        if len(i) < _ICP_MIN_PAIRS:
            diag["reason"] = "too few points line up with the front"
            break
        p = X[i]
        res = np.abs(((p - q) * nrm).sum(axis=1)) if metric == "plane" else d
        keep = res <= np.quantile(res, 1.0 - trim_v)
        if int(keep.sum()) < _ICP_MIN_PAIRS:
            keep = np.ones(len(res), bool)
        if metric == "plane":
            su, Ru, tu = _plane_step(p[keep], q[keep], nrm[keep])
        else:
            su, Ru, tu = umeyama(p[keep], q[keep], with_scale=True)
        diag["iters"] = k + 1
        ang = _rot_angle_deg(Ru)
        if not (ang <= max_rot_deg and 1.0 / _ICP_MAX_STEP_SCALE <= su <= _ICP_MAX_STEP_SCALE):
            diag["rejected"] = True
            log.info("Projective ICP: step %d rejected (%.1f deg, scale %.3f)", k + 1, ang, su)
            break
        s, R, t = su * s, Ru @ R, su * (Ru @ t) + tu
        if ang < 0.01 and abs(su - 1.0) < 1e-5 and float(np.linalg.norm(tu)) < 1e-4 * tg.size_m:
            diag["converged"] = True
            break
    if len(X0) < _ICP_MIN_PAIRS:
        diag["reason"] = "too few points"
    X = s * (X0 @ R.T) + t
    i, q, nrm, d = _icp_pairs(X, front, tg, kv, r_min) if len(X0) else (np.zeros(0),) * 4
    pl = np.abs(((X[i] - q) * nrm).sum(axis=1)) if len(i) else np.zeros(0)
    diag.update(
        pairs=int(len(i)),
        resid_m=float(np.median(pl)) if len(pl) else float("nan"),
        dist_m=float(np.median(d)) if len(d) else float("nan"),
    )
    return s, R, t, diag


# ---------------------------------------------------------------------- cloud registration helpers
def up_rotation(axis: str) -> np.ndarray:
    """Rotation ``A`` taking the cloud axis ``axis`` (a key of :data:`UP_AXES`) to the
    subject frame's up ``(0, -1, 0)`` (S has ``y`` down): the shortest rotation (Rodrigues
    about ``axis x up``), and the half turn about ``x`` for ``"+y"``."""
    if axis not in UP_AXES:
        raise InputError(f"Unknown axis {axis!r} (expected one of {', '.join(UP_AXES)}).")
    a = np.array(UP_AXES[axis])
    u = np.array([0.0, -1.0, 0.0])
    c = float(a @ u)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(a, u)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def yaw_rotation(yaw_deg: float) -> np.ndarray:
    """``R_y(yaw)`` in the subject frame S (about its vertical ``y^`` axis): ``[[c, 0, s],
    [0, 1, 0], [-s, 0, c]]``."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class _Work:
    """SHARP's front on a z-buffer grid over the photo (``long`` cells on its long side)."""

    h: int
    w: int
    cell: float  # photo pixels per cell
    cam_f: float
    half_w: float
    half_h: float
    matte: np.ndarray  # (h, w) bool: matte cells (area >= 0.5)
    D_m: np.ndarray  # (Hm, Wm) float32 front depth on the mould grid, filled (no NaN)
    mould_cell: float  # photo pixels per mould cell
    dt_out: np.ndarray  # (h, w) distance (cells) of outside cells to the matte
    n_matte: int
    h_px: float  # matte height (cells)
    centroid: np.ndarray  # (2,) matte centroid, continuous photo pixels

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continuous cell coordinates and depth of ``N x 3`` points in C."""
        z = X[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            zs = np.where(z > 1e-6, z, np.nan)
            x = (self.cam_f * X[:, 0] / zs + self.half_w) / self.cell
            y = (self.cam_f * X[:, 1] / zs + self.half_h) / self.cell
        return x, y, zs


def _work_grid(front: FrontModel, long_side: int) -> _Work:
    cam = front.cam
    W, H = int(cam.width), int(cam.height)
    cell = max(W, H) / float(long_side)
    w, h = max(1, round(W / cell)), max(1, round(H / cell))
    matte_m = np.asarray(front.matte, dtype=np.float32)
    if not matte_m.any():
        raise InputError("SHARP's front is empty, so nothing can be lined up with it.")
    interp = cv2.INTER_AREA if w < matte_m.shape[1] else cv2.INTER_LINEAR
    matte = cv2.resize(matte_m, (w, h), interpolation=interp) >= 0.5
    if not matte.any():  # a tiny subject: keep at least its nearest cell
        matte = cv2.resize(matte_m, (w, h), interpolation=cv2.INTER_NEAREST) >= 0.5
    D = np.asarray(front.D_f, dtype=np.float64)
    D = np.where(np.isfinite(D), D, np.asarray(front.D_ext, dtype=np.float64))
    fin = np.isfinite(D)
    D = np.where(fin, D, float(np.median(D[fin])) if fin.any() else 1.0).astype(np.float32)
    ys, xs = np.nonzero(matte)
    dt_out = cv2.distanceTransform((~matte).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return _Work(
        h=h,
        w=w,
        cell=cell,
        cam_f=float(cam.f),
        half_w=W / 2.0,
        half_h=H / 2.0,
        matte=matte,
        D_m=D,
        mould_cell=float(front.grid.cell_px),
        dt_out=dt_out.astype(np.float64),
        n_matte=int(matte.sum()),
        h_px=float(ys.max() - ys.min() + 1),
        centroid=np.array([(xs.mean() + 0.5) * cell, (ys.mean() + 0.5) * cell]),
    )


_CLOSE3 = np.ones((3, 3), np.uint8)
_COVER_SS = 3  # coverage supersampling of the working grid
_VISIBLE_TOL_M = 0.02  # a point within this of the z-buffer depth is on the visible surface


@dataclass
class _Eval:
    iou: float
    resid: float  # median |z - D_f(u, v)| (m) of visible points over the matte
    chamfer: float  # in units of 5 % of the matte height
    cov_centroid: np.ndarray | None  # (2,) covered-cell centroid (cells)
    z_near: float  # median depth of the visible points over the matte


def _disc_kernel(r: float) -> np.ndarray:
    reach = int(math.floor(r))
    k = np.zeros((2 * reach + 1, 2 * reach + 1), np.uint8)
    for dx, dy in _disc_offsets(r):
        k[dy + reach, dx + reach] = 1
    return k


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear lookup at continuous OpenCV coordinates (pixel centres at integers),
    clamped to the image."""
    H, W = img.shape
    xc = np.clip(x, 0.0, W - 1.0)
    yc = np.clip(y, 0.0, H - 1.0)
    x0 = np.minimum(np.floor(xc).astype(np.int64), max(W - 2, 0))
    y0 = np.minimum(np.floor(yc).astype(np.int64), max(H - 2, 0))
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = xc - x0
    fy = yc - y0
    return (
        img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
        + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy
    )  # fmt: skip


def _evaluate(X: np.ndarray, wk: _Work, radius: float, chamfer: bool = False) -> _Eval:
    """Silhouette and front-depth agreement of points ``X`` (C) with the front on ``wk``.

    Both measures are built to peak exactly at the true pose (measured on the synthetic
    person; the plain per-cell versions scored a 2 % too small cloud higher):

    * coverage: the points' cells at 3x the grid's resolution, dilated by the disc
      ``radius`` (fine cells) to close the gaps between points, closed 3 x 3, eroded by the
      disc again, then area-averaged back to the grid (>= 0.5).  "Any point in the cell"
      would grow the silhouette by up to a cell, because grazing points crowd at the
      outline.
    * residual: the median of ``|z - D_f(u, v)|`` over the visible points (within 2 cm of
      their cell's nearest point) inside the matte, ``D_f`` bilinear on the mould grid at the
      point's own position (a cell's nearest point compared with the cell's mean depth is
      biased by the depth spread inside the cell).

    Silhouette work is uint8 morphology only (float z-buffers at the fine resolution cost
    ~3x more for the same answer)."""
    ss = _COVER_SS
    x, y, z = wk.project(X)
    pad = int(math.floor(radius))
    Hs, Ws = wk.h * ss + 2 * pad, wk.w * ss + 2 * pad
    with np.errstate(invalid="ignore"):
        fx = np.floor(x * ss) + pad
        fy = np.floor(y * ss) + pad
        ok = (fx >= 0) & (fx < Ws) & (fy >= 0) & (fy < Hs) & np.isfinite(z)
    i = np.flatnonzero(ok)
    if not len(i):
        return _Eval(0.0, 1.0, 10.0, None, float("nan"))
    hit = np.zeros(Hs * Ws, np.uint8)
    hit[fy[i].astype(np.int64) * Ws + fx[i].astype(np.int64)] = 255
    hit = hit.reshape(Hs, Ws)
    if pad:
        k = _disc_kernel(radius)
        hit = cv2.dilate(hit, k)
        hit = cv2.morphologyEx(hit, cv2.MORPH_CLOSE, _CLOSE3)
        hit = cv2.erode(hit, k, borderType=cv2.BORDER_REPLICATE)[pad : pad + wk.h * ss, pad : pad + wk.w * ss]
    else:
        hit = cv2.morphologyEx(hit, cv2.MORPH_CLOSE, _CLOSE3)
    cov = cv2.resize(hit, (wk.w, wk.h), interpolation=cv2.INTER_AREA) >= 128
    if not cov.any():
        return _Eval(0.0, 1.0, 10.0, None, float("nan"))
    inter = int(np.count_nonzero(cov & wk.matte))
    union = int(np.count_nonzero(cov | wk.matte))
    iou = inter / union if union else 0.0
    # visible points: within 2 cm of the nearest point of their (working) cell
    cx = (fx[i] - pad) // ss
    cy = (fy[i] - pad) // ss
    inb = (cx >= 0) & (cx < wk.w) & (cy >= 0) & (cy < wk.h)
    i, cx, cy = i[inb], cx[inb].astype(np.int64), cy[inb].astype(np.int64)
    flat = cy * wk.w + cx
    zc = np.full(wk.h * wk.w, np.inf)
    np.minimum.at(zc, flat, z[i])
    keep = (z[i] <= zc[flat] + _VISIBLE_TOL_M) & wk.matte.ravel()[flat]
    i = i[keep]
    if len(i):
        gx = x[i] * (wk.cell / wk.mould_cell) - 0.5  # mould-grid OpenCV coordinates
        gy = y[i] * (wk.cell / wk.mould_cell) - 0.5
        resid = float(np.median(np.abs(z[i] - _bilinear(wk.D_m, gx, gy))))
        z_near = float(np.median(z[i]))
    else:
        fin = zc[np.isfinite(zc)]
        resid, z_near = 1.0, float(np.median(fin)) if len(fin) else float("nan")
    ys, xs = np.nonzero(cov)
    cen = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
    ch = 0.0
    if chamfer:
        out_t = float(wk.dt_out[cov].mean())
        dt_cov = cv2.distanceTransform((~cov).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        in_t = float(dt_cov[wk.matte].mean())
        ch = 0.5 * (out_t + in_t) / max(1.0, 0.05 * wk.h_px)
    return _Eval(iou, resid, ch, cen, z_near)


def _splat_radius(n_points: int, wk: _Work) -> float:
    """Disc radius (fine coverage cells) that closes the gaps between the visible half of
    ``n_points`` spread over the matte: 1.2 x their mean spacing, in [1, 4]."""
    per_cell = 0.5 * n_points / max(wk.n_matte * _COVER_SS**2, 1)
    return float(np.clip(1.2 / math.sqrt(max(per_cell, 1e-6)), 1.0, 4.0))


def _decompose(R_total: np.ndarray, canon: Canon, A: np.ndarray) -> tuple[float, float]:
    """``(yaw_deg in [0, 360), tilt_deg)`` of ``R_total = canon.R @ T @ R_y(yaw) @ A`` where
    ``T`` is the smallest rotation tilting the vertical (``tilt_deg`` its angle)."""
    M = canon.R.T @ R_total @ A.T
    y = np.array([0.0, 1.0, 0.0])
    my = M @ y
    c = float(np.clip(my @ y, -1.0, 1.0))
    tilt = math.degrees(math.acos(c))
    v = np.cross(y, my)
    if np.linalg.norm(v) > 1e-12 and c > -1.0 + 1e-12:
        K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        T = np.eye(3) + K + K @ K / (1.0 + c)
    else:
        T = np.eye(3)
    Ry = T.T @ M
    yaw = math.degrees(math.atan2(Ry[0, 2], Ry[0, 0])) % 360.0
    return yaw, tilt


def _fail(diag: dict[str, Any], reason: str) -> tuple[None, dict[str, Any]]:
    diag.update(ok=False, reason=reason)
    log.info("Cloud registration failed: %s", reason)
    return None, diag


def register_cloud_to_front(
    sc: SplatCloud,
    front: FrontModel,
    canon: Canon,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> tuple[SplatCloud | None, dict[str, Any]]:
    """Place a free 360 deg Gaussian cloud of unknown axis convention and scale (TRELLIS:
    a unit cube, any up axis) behind SHARP's front, in frame C (design 3.5.5 steps 1-6).

    1. Centres with opacity > 0.3, a fixed random subsample of 20k, centred on their mean.
    2. Candidates: the cloud's up is each of ``+-x, +-y, +-z`` (:func:`up_rotation` maps it
       to the subject's up) times 48 yaws (7.5 deg, :func:`yaw_rotation` about S's
       vertical), so the cloud's rotation into C is ``canon.R @ R_y(yaw) @ A_up``.  Scale
       ``s0 = canon.height_s / (P99.5 - P0.5 of the cloud along up)``; translation: on the
       ray through the matte centroid with the median depth of the visible points equal to
       the front's median depth, then shifted so the covered cells' centroid is the
       matte's.  Each is z-buffered on a 128-cell grid over the photo.
    3. Score ``IoU - 0.5 * min(1, med|z - D_f| / 0.1 m)``.
    4. The best three (at least 20 deg apart in rotation, so they are not one basin)
       are refined by Nelder-Mead over (yaw, pitch, roll within 15 deg, log s, t) on a
       192-cell grid; cost ``1 - IoU + 0.5 * resid / 0.1 m + 0.2 * chamfer`` (chamfer: the
       mean distance of covered cells to the matte and of matte cells to the coverage, in
       units of 5 % of the matte height).
    5. :func:`projective_icp` from the best (kept only when it does not raise that cost by
       more than 0.05).
    6. Gates: IoU >= 0.70, median depth residual <= 5 cm, ``s`` within ``[0.5, 2] * s0``.

    Returns ``(sc_C, diag)``: the whole cloud (every Gaussian, :func:`transform_splats`)
    in C, or None when it cannot be registered or a gate fails.  ``diag`` (JSON-ready):
    ``ok``, ``reason`` (plain words), ``iou``, ``resid_m``, ``scale`` (cloud units ->
    metres), ``s0``, ``up`` (cloud axis, e.g. ``"-z"``), ``yaw_deg``, ``tilt_deg``,
    ``transform`` (``s, R, t`` with ``P_C = s R P + t``), ``candidates``, ``icp``,
    ``seconds``.  Raises :class:`InputError` for a malformed cloud or front."""
    rep = _as_reporter(progress)
    t_start = time.perf_counter()
    _check(cancel)
    _validate(sc)
    diag: dict[str, Any] = {"ok": False, "reason": "", "n_in": len(sc)}
    rep(0.0, "Lining up the TRELLIS shape with the photo")
    P_all = np.asarray(sc.means, dtype=np.float64)
    op = np.nan_to_num(np.asarray(sc.opacity, dtype=np.float64), nan=0.0)
    ok = np.isfinite(P_all).all(axis=1) & (op > _REG_OPACITY)
    rows = np.flatnonzero(ok)
    diag["n_opaque"] = int(len(rows))
    if len(rows) < _REG_MIN_POINTS:
        return _fail(diag, "the generated shape is empty or almost transparent")
    if len(rows) > _REG_POINTS:
        rows = np.random.default_rng(0).choice(rows, _REG_POINTS, replace=False)
    else:
        rows = np.random.default_rng(0).permutation(rows)  # any prefix is a random subset
    Pt = P_all[rows]
    m_T = Pt.mean(axis=0)
    Pc = Pt - m_T
    if float(np.abs(Pc).max()) <= 1e-12:
        return _fail(diag, "the generated shape has no extent")
    wk_c = _work_grid(front, _REG_COARSE_LONG)
    wk_f = _work_grid(front, _REG_FINE_LONG)
    D_front = np.asarray(front.D_f, dtype=np.float64)[np.asarray(front.matte, dtype=bool)]
    D_front = D_front[np.isfinite(D_front)]
    z_ref = float(np.median(D_front)) if len(D_front) else float(np.median(wk_c.D_m))
    cam = front.cam
    d_c = cam.rays(wk_c.centroid[0], wk_c.centroid[1])  # z = 1 ray through the matte centroid
    # nested random subsets: ranking candidates and the simplex need fewer points than ICP
    P_co = Pc[:_REG_COARSE_POINTS]
    P_nm = Pc[:_REG_REFINE_POINTS]
    rad_c = _splat_radius(len(P_co), wk_c)
    rad_f = _splat_radius(len(P_nm), wk_f)
    rad_g = _splat_radius(len(Pc), wk_f)
    height = float(canon.height_s)
    if not (math.isfinite(height) and height > 0):
        raise InputError(f"Invalid subject height {canon.height_s!r}.")

    # ---- 2-3. discrete search over up axis x yaw
    cands: list[dict[str, Any]] = []
    yaws = np.arange(_REG_YAWS) * (360.0 / _REG_YAWS)
    for ai, (name, _) in enumerate(UP_AXES.items()):
        _check(cancel)
        rep(0.02 + 0.4 * ai / len(UP_AXES), "Lining up the TRELLIS shape: trying orientations")
        A = up_rotation(name)
        up_h = -(Pc @ A.T)[:, 1]  # height along the subject's up
        lo, hi = np.percentile(up_h, [0.5, 99.5])
        if hi - lo <= 1e-12:
            continue
        s0 = height / float(hi - lo)
        for yaw in yaws:
            Rk = canon.R @ yaw_rotation(float(yaw)) @ A
            X = s0 * (P_co @ Rk.T)
            p = d_c * (z_ref - float(np.percentile(X[:, 2], 25.0)))
            ev = _evaluate(X + p, wk_c, rad_c)
            if ev.cov_centroid is not None and math.isfinite(ev.z_near):
                shift = (wk_c.centroid / wk_c.cell - ev.cov_centroid) * wk_c.cell * ev.z_near / cam.f
                p = p + d_c * (z_ref - ev.z_near) + np.array([shift[0], shift[1], 0.0])
                ev = _evaluate(X + p, wk_c, rad_c)
            score = ev.iou - 0.5 * min(1.0, ev.resid / _REG_RESID_REF)
            cands.append({"up": name, "yaw": float(yaw), "s0": s0, "R": Rk, "p": p, "score": score, "A": A})
    if not cands:
        return _fail(diag, "the generated shape has no height")
    cands.sort(key=lambda c: -c["score"])
    top: list[dict[str, Any]] = []
    for c in cands:
        if all(_rot_angle_deg(c["R"] @ o["R"].T) >= _REG_DISTINCT_DEG for o in top):
            top.append(c)
        if len(top) == _REG_TOP:
            break
    diag["candidates"] = [
        {"up": c["up"], "yaw_deg": round(c["yaw"], 1), "score": round(c["score"], 4)} for c in top
    ]

    # ---- 4. Nelder-Mead refinement of the best three
    def pose(c: dict[str, Any], x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        R = canon.R @ _rot_x(x[1]) @ _rot_z(x[2]) @ yaw_rotation(c["yaw"] + x[0]) @ c["A"]
        return c["s0"] * math.exp(x[3]), R, c["p"] + x[4:7]

    def refine_cost(s: float, R: np.ndarray, p: np.ndarray) -> float:
        ev = _evaluate(s * (P_nm @ R.T) + p, wk_f, rad_f, chamfer=True)
        return 1.0 - ev.iou + 0.5 * ev.resid / _REG_RESID_REF + 0.2 * ev.chamfer

    tstep = 0.02 * height
    step = np.array([4.0, 3.0, 3.0, 0.04, tstep, tstep, tstep])
    best: tuple[float, float, np.ndarray, np.ndarray, dict[str, Any]] | None = None
    for ci, c in enumerate(top):
        _check(cancel)
        rep(0.45 + 0.35 * ci / len(top), f"Lining up the TRELLIS shape: refining ({ci + 1}/{len(top)})")

        def cost(x: np.ndarray, c: dict[str, Any] = c) -> float:
            excess = max(0.0, abs(x[1]) - _REG_TILT_MAX) + max(0.0, abs(x[2]) - _REG_TILT_MAX)
            return refine_cost(*pose(c, x)) + 0.1 * (excess / 5.0) ** 2

        x, fx = nelder_mead(cost, np.zeros(7), step, max_iter=_REG_NM_ITERS, tol=1e-4, cancel=cancel)
        diag["candidates"][ci]["cost"] = round(float(fx), 4)
        if best is None or fx < best[0]:
            s, R, p = pose(c, x)
            best = (fx, s, R, p, c)
    if best is None:  # cannot happen: there is at least one candidate
        return _fail(diag, "no orientation could be scored")
    cost0, s_b, R_b, p_b, c_b = best

    # ---- 5. projective ICP
    _check(cancel)
    rep(0.82, "Lining up the TRELLIS shape: fitting the front surface")
    X_b = s_b * (Pc @ R_b.T) + p_b
    si, Ri, ti, icp_diag = projective_icp(X_b, front, cancel=cancel)
    s_i, R_i, p_i = si * s_b, Ri @ R_b, si * (Ri @ p_b) + ti
    cost_i = refine_cost(s_i, R_i, p_i)
    icp_diag["kept"] = bool(cost_i <= cost0 + 0.05)
    if icp_diag["kept"]:
        s_b, R_b, p_b = s_i, R_i, p_i
    diag["icp"] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in icp_diag.items()}

    # ---- 6. gates
    ev = _evaluate(s_b * (Pc @ R_b.T) + p_b, wk_f, rad_g)
    t_b = p_b - s_b * (R_b @ m_T)  # P_C = s R P_T + t
    yaw, tilt = _decompose(R_b, canon, c_b["A"])
    ratio = s_b / c_b["s0"]
    diag.update(
        iou=round(float(ev.iou), 4),
        resid_m=round(float(ev.resid), 4),
        scale=round(float(s_b), 6),
        s0=round(float(c_b["s0"]), 6),
        up=c_b["up"],
        yaw_deg=round(float(yaw), 2),
        tilt_deg=round(float(tilt), 2),
        transform={"s": float(s_b), "R": R_b.tolist(), "t": t_b.tolist()},
        seconds=round(time.perf_counter() - t_start, 2),
    )
    reasons = []
    if ev.iou < _REG_GATE_IOU:
        reasons.append(f"its outline matches the photo only {100 * ev.iou:.0f} %, 70 % is needed")
    if not ev.resid <= _REG_GATE_RESID:
        reasons.append(
            f"its front is {100 * ev.resid:.0f} cm off the photo's surface, at most 5 cm is allowed"
        )
    if not (_REG_GATE_SCALE[0] <= ratio <= _REG_GATE_SCALE[1]):
        reasons.append(f"its size had to change {ratio:.2f}x to fit")
    log.info(
        "Cloud registration: up %s, yaw %.1f deg, tilt %.1f deg, scale %.4g (%.2f x s0), IoU %.3f, "
        "residual %.1f cm, ICP %s, %.1f s",
        c_b["up"],
        yaw,
        tilt,
        s_b,
        ratio,
        ev.iou,
        100 * ev.resid,
        "kept" if icp_diag["kept"] else "discarded",
        diag["seconds"],
    )
    if reasons:
        return _fail(diag, "; ".join(reasons))
    _check(cancel)
    out = transform_splats(sc, R_b, t_b, s_b)
    diag.update(ok=True, reason="")
    rep(1.0, "TRELLIS shape lined up")
    return out, diag


# =========================================================================== colour
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


_LIN_LUT = _srgb_to_linear(np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)


def _colors01(c: Any) -> np.ndarray:
    """``...x3`` colours as ``N x 3`` float64 in [0, 1] (uint8 is divided by 255)."""
    a = np.asarray(c)
    if a.size == 0:
        return np.zeros((0, 3))
    if a.shape[-1] != 3:
        raise InputError(f"Colours must be ...x3 RGB (got shape {a.shape}).")
    out = a.reshape(-1, 3).astype(np.float64)
    return out / 255.0 if a.dtype == np.uint8 else out


def _huber_cost(r: np.ndarray, delta: float) -> float:
    return float(np.where(r <= delta, 0.5 * r * r, delta * (r - 0.5 * delta)).sum())


def color_affine_fit(
    src: Any, dst: Any, *, lam: float = 0.1, iters: int = 5
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Affine colour map ``c_dst ~= A c_src + b`` in LINEAR RGB from paired sRGB colours
    (``N x 3`` - or any ``...x3`` - floats in [0, 1], or uint8) - design V4.

    Minimises ``sum rho_Huber(||A c_s + b - c_d||) + lam * n * sigma0^2 * ||A - I||_F^2``
    by IRLS (``iters`` reweightings after a least-squares start; Huber threshold ``1.345
    sigma`` with ``sigma = median |r| / 1.538``, the per-channel noise of 3-D Gaussian
    residual vectors, at least 1e-4).  ``sigma0 = 0.02`` (linear RGB) puts the
    design's ``lam = 0.1 n`` ridge on the scale of real colour spreads: a colour direction
    with less spread than ~0.02 keeps the identity gain, well-spread colours are fitted
    essentially unbiased (in 0-1 units the plain ``0.1 n`` would shrink every gain about
    halfway to 1).  Then clamped: ``||A - I||_F <= 0.35``, ``diag A`` in [0.6, 1.6], ``b``
    refitted for the clamped ``A`` and clipped to ``+-0.12``.

    Returns ``(A, b, ok)``; ``ok`` is False (and ``A = I, b = 0``) when there are fewer
    than 8 usable pairs or the fit's robust cost does not beat the identity's."""
    s, d = _colors01(src), _colors01(dst)
    if s.shape != d.shape:
        raise InputError(f"Colour pairs must have the same shape (got {s.shape} and {d.shape}).")
    eye = np.eye(3)
    ok = np.isfinite(s).all(axis=1) & np.isfinite(d).all(axis=1)
    if int(ok.sum()) < _COLOR_MIN_N:
        return eye, np.zeros(3), False
    X = _srgb_to_linear(s[ok])
    Y = _srgb_to_linear(d[ok])
    n = len(X)
    Phi = np.hstack([X, np.ones((n, 1))])
    X0 = np.vstack([eye, np.zeros((1, 3))])
    w = np.ones(n)
    lam_v = max(0.0, float(lam))
    delta = _HUBER_MIN
    sol = X0
    for it in range(max(0, int(iters)) + 1):
        lam_eff = lam_v * float(w.sum()) * _COLOR_SIGMA0**2
        reg = np.diag([lam_eff, lam_eff, lam_eff, 0.0])
        G = Phi.T @ (Phi * w[:, None]) + reg
        B = Phi.T @ (Y * w[:, None]) + reg @ X0
        try:
            sol = np.linalg.solve(G, B)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(G, B, rcond=None)[0]
        r = np.linalg.norm(Phi @ sol - Y, axis=1)
        delta = max(_HUBER_K * float(np.median(r)) / _CHI3_MEDIAN, _HUBER_MIN)
        if it < iters:
            w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-300))
    A = sol[:3].T.copy()
    dev = A - eye
    nf = float(np.linalg.norm(dev))
    if nf > _COLOR_MAX_DEV:
        A = eye + dev * (_COLOR_MAX_DEV / nf)
    di = np.clip(np.diag(A), *_COLOR_DIAG)
    A[np.diag_indices(3)] = di
    b = (w[:, None] * (Y - X @ A.T)).sum(axis=0) / max(float(w.sum()), 1e-300)
    b = np.clip(b, -_COLOR_MAX_B, _COLOR_MAX_B)
    cost_fit = _huber_cost(np.linalg.norm(X @ A.T + b - Y, axis=1), delta)
    cost_id = _huber_cost(np.linalg.norm(X - Y, axis=1), delta)
    if not cost_fit < cost_id:
        return eye, np.zeros(3), False
    return A, b, True


def apply_color_affine(rgb: Any, A: Any, b: Any) -> np.ndarray:
    """Apply a :func:`color_affine_fit` map: sRGB in, sRGB out, the affine in linear RGB.
    ``rgb`` is uint8 (``...x3``, e.g. an ``H x W x 3`` image) or float in [0, 1] (e.g.
    ``N x 3`` splat colours); the result has the same shape and dtype (clipped)."""
    a = np.asarray(rgb)
    Am = np.asarray(A, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32).ravel()
    if a.ndim < 1 or a.shape[-1] != 3 or Am.shape != (3, 3) or bv.shape != (3,):
        raise InputError(f"Colour map needs ...x3 colours, a 3x3 A and a 3-vector b (got {a.shape}).")
    if a.dtype != np.uint8 and a.dtype.kind != "f":
        raise InputError(f"Colours must be uint8 or float in [0, 1] (got {a.dtype}).")
    flat = a.reshape(-1, 3)
    out = np.empty(flat.shape, a.dtype)
    for i in range(0, len(flat), _COLOR_CHUNK):  # a full photo would need several 150 MB temporaries
        blk = flat[i : i + _COLOR_CHUNK]
        lin = _LIN_LUT[blk] if a.dtype == np.uint8 else _srgb_to_linear(blk.astype(np.float32))
        srgb = _linear_to_srgb(lin @ Am.T + bv)
        out[i : i + _COLOR_CHUNK] = np.clip(np.rint(srgb * 255.0), 0, 255) if a.dtype == np.uint8 else srgb
    return out.reshape(a.shape)


def _rgb_u8(img: Any, what: str) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise InputError(f"The {what} must be an H x W x 3 image (got shape {a.shape}).")
    a = a[..., :3]
    if a.dtype == np.uint8:
        return a
    if a.dtype.kind == "f":
        return np.clip(np.rint(np.nan_to_num(a.astype(np.float64)) * 255.0), 0, 255).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def reinhard_lab(img: Any, mask: Any, ref: Any, ref_mask: Any, *, strength: float = 0.5) -> np.ndarray:
    """Reinhard colour transfer: shift and scale ``img``'s CIELAB channels so their mean /
    standard deviation over ``mask`` move toward ``ref``'s over ``ref_mask``, by
    ``strength`` (0 = unchanged, 1 = full transfer).  The statistics come from the masks
    (e.g. the rim bands of a back view and of the photo); the mapping is applied to every
    pixel of ``img``.  The std ratio is clamped to [0.5, 2].  Images are uint8 (or float in
    [0, 1]) ``H x W x 3`` RGB; returns uint8.  Fewer than 16 pixels in either mask returns
    ``img`` unchanged (logged)."""
    a = _rgb_u8(img, "image")
    r = _rgb_u8(ref, "reference image")
    m = _as_mask(mask, "image mask")
    rm = _as_mask(ref_mask, "reference mask")
    if m.shape != a.shape[:2] or rm.shape != r.shape[:2]:
        raise InputError("Each mask must match its image's size.")
    k = float(strength)
    if not (math.isfinite(k) and 0.0 <= k <= 1.0):
        raise InputError(f"The colour transfer strength must be in [0, 1] (got {strength!r}).")
    if int(m.sum()) < _REINHARD_MIN_PX or int(rm.sum()) < _REINHARD_MIN_PX:
        log.info("Reinhard colour transfer skipped: a mask has fewer than %d pixels", _REINHARD_MIN_PX)
        return a.copy()
    lab = cv2.cvtColor(a.astype(np.float32) / np.float32(255.0), cv2.COLOR_RGB2Lab)
    lab_r = cv2.cvtColor(r.astype(np.float32) / np.float32(255.0), cv2.COLOR_RGB2Lab)
    mu_s, sd_s = lab[m].mean(axis=0, dtype=np.float64), lab[m].std(axis=0, dtype=np.float64)
    mu_r, sd_r = lab_r[rm].mean(axis=0, dtype=np.float64), lab_r[rm].std(axis=0, dtype=np.float64)
    ratio = np.clip(sd_r / np.maximum(sd_s, 1e-6), *_REINHARD_RATIO)
    # lab + k ((lab - mu_s) ratio + mu_r - lab) as one per-channel gain and offset
    gain = (1.0 + k * (ratio - 1.0)).astype(np.float32)
    offset = (k * (mu_r - mu_s * ratio)).astype(np.float32)
    rgb = cv2.cvtColor(lab * gain + offset, cv2.COLOR_Lab2RGB)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


__all__ = [
    "TPS",
    "UP_AXES",
    "apply_color_affine",
    "chamfer_perspective_fit",
    "chamfer_similarity_fit",
    "color_affine_fit",
    "contour_points",
    "nelder_mead",
    "outline_gap",
    "point_zbuffer",
    "projective_icp",
    "register_cloud_to_front",
    "reinhard_lab",
    "silhouette_iou",
    "tps_fit",
    "tps_warp",
    "umeyama",
    "up_rotation",
    "warp_similarity",
    "yaw_rotation",
]
