"""SHARP 360: build the hidden back of the subject as a "mould" behind SHARP's front.

Design sections 3.5.1 and 3.6 (docs/design/SHARP360.md).  The back is a height field
along the photo camera's rays on the isotropic mould grid (``splat_fusion.MouldGrid``):
for every matte cell ``p`` the back surface is at depth ``z_b(p) > D_f(p)``.  It closes
exactly at the photo silhouette because the target thickness is zero there; where the
silhouette edge has depth (``Z_rim > D_ext``), a "rim wall" of Gaussians closes the side.

* :func:`inflation_prior` - the billboard-style inflated thickness behind the front: the
  silhouette's circular cross-section profile (``billboard.inflation_profile``) in metres
  (``mpp = D_f / f_m`` per cell), flattened to an ellipse where a part is wide (a torso
  is not as deep as it is wide), scaled by ``kappa`` so the median thickness is
  ``thickness * height``, added to a smoothed copy of the front depth (so the back does
  not copy the nose or the folds of the front).
* :func:`poisson_heightfield` - screened Poisson solve of the back depth from a target,
  boundary values and edge weights.  Matrix-free: the 5-point operator is applied with
  shifted slices and edge masks on the mask's bounding box, Jacobi-preconditioned
  conjugate gradient, coarse to fine (2x2 block pyramid, warm start by upsampling).  It
  solves for the correction ``y = z - target``, which is smooth and small, so the coarse
  levels are good starting points and float64 never has to resolve metres-sized offsets.
* :func:`gaussians_from_heightfield` - one flat Gaussian per (strided) cell, oriented on
  the surface by Gram-Schmidt tangents (det +1) with the normal away from the camera.
  Steep cells near the outline get long tangent axes, which closes the sides.
* :func:`rim_wall` - Gaussians closing the side between the front edge and the back edge.
* :func:`shell_back` - the built-in back source (no AI): prior + colours from the outline.
* :func:`mould_back` - shared back builder for every source that provides a target depth.

All Gaussians are returned in SHARP's camera frame C (OpenCV axes, metres).  Mould cell
``(row, col)`` has its centre at photo pixel ``((col + 0.5) * cell_px, (row + 0.5) *
cell_px)`` and the ray ``((u - W/2) / f, (v - H/2) / f, 1)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..core.errors import InputError
from ..core.fill import push_pull_fill
from ..core.log import get_logger
from ..core.types import CancelToken, FloatMap, ProgressReporter, SplatCloud
from .billboard import _BACK_DARKEN, _BAND_PX, _KAPPA_MAX, BACK_MODES, _dist, _masked_blur, inflation_profile
from .mesh import _as_reporter, _check
from .splat import concat_splats, quat_to_rotmat, rotmat_to_quat
from .splat_fusion import Canon, FrontModel

log = get_logger(__name__)

#: Back opacity of mould Gaussians.
BACK_OPACITY = 0.95
#: The back stays at least this far (m) behind the front surface.
MIN_GAP_M = 0.005
#: Max relief detail (m) added from a depth map's high-pass when there are no side views.
DETAIL_M = 0.03
#: Lower clamp (m) of a solved mould back behind the front (design 3.6; the shell uses MIN_GAP_M).
MOULD_GAP_M = 0.01
#: A rim wall is only built where the outline's back edge is this far (m) behind its front edge.
RIM_MIN_M = 0.01

_SCALE_MIN, _SCALE_MAX = 0.001, 0.05  # clamp of the two surface axes (m)
_S3_MIN = 0.0005  # thinnest axis (m)
_HULL_SLACK_M = 0.02  # a solved back may exceed the hull exit by this much
_ANCHOR_LAM = 10.0  # weight of the clamped cells in the re-solve
_WALL_MAX_STEPS = 64  # Gaussians per wall ray, at most
_CHUNK = 1 << 18  # cells per vectorised chunk
_KAPPA_SAMPLES = 200_000  # matte cells used to fit kappa
_BAND_REF_LONG = 1024  # the billboard's 8 px colour band is defined at this grid size


# --------------------------------------------------------------------------- data
@dataclass
class HullIntervals:
    """Per mould-grid ray, the first contiguous ``[z_in, z_out]`` interval inside the visual
    hull of the accepted generated views (filled in by view-based sources, milestone M4).
    ``valid`` marks rays where the interval is known."""

    z_in: FloatMap
    z_out: FloatMap
    valid: np.ndarray


@dataclass
class BackPart:
    """A back made by one back source, in frame C, generated Gaussians only.

    ``kind`` is ``"mould"`` (a closed height-field back from this module) or ``"cloud"``
    (a free Gaussian cloud such as TRELLIS's, merged with a closure under-layer).
    ``ok=False`` with a plain-language ``reason`` makes the orchestrator fall back to the
    next source.  ``back_z`` is the back depth per mould cell (NaN where unknown), used by
    the under-layer and the floater filter.  ``views`` holds generated / user views
    (``views_lift.GenView``) for previews and the Brush polish (later milestones).
    ``diag["warning"]``, when present, is a plain sentence for the user (e.g. a fallback).
    ``spacing_m`` (an addition to the design) is the typical distance between neighbouring
    Gaussians of ``sc`` when the source knows it (a strided mould back far from the camera
    is sampled more coarsely than the floater filter's 1.5 cm voxels); None = unknown.
    """

    sc: SplatCloud
    kind: str = "mould"
    source: str = "shell"
    ok: bool = True
    reason: str = ""
    normals: np.ndarray | None = None  # (N, 3) outward normals of sc's Gaussians (C)
    back_z: FloatMap | None = None
    hull: HullIntervals | None = None
    views: list[Any] | None = None
    diag: dict[str, Any] | None = None
    spacing_m: float | None = None


# --------------------------------------------------------------------------- helpers
def _grid_shape(front: FrontModel) -> tuple[int, int]:
    return int(front.grid.Hm), int(front.grid.Wm)


def _as_map(x: Any, shape: tuple[int, int], what: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != shape:
        raise InputError(f"The {what} map ({a.shape}) does not match the mould grid {shape}.")
    return a


def _front_matte(front: FrontModel) -> np.ndarray:
    matte = np.asarray(front.matte, dtype=bool)
    shape = _grid_shape(front)
    if matte.shape != shape:
        raise InputError(f"The subject outline ({matte.shape}) does not match the mould grid {shape}.")
    if not matte.any():
        raise InputError(
            "The subject outline is empty, so no back can be built.",
            hint="Check that the subject is selected in the photo.",
        )
    return matte


def _front_maps(front: FrontModel) -> tuple[np.ndarray, np.ndarray]:
    """``(matte, D_f)``: ``D_f`` (float64) filled inside the matte, NaN outside."""
    matte = _front_matte(front)
    D_f = _as_map(front.D_f, matte.shape, "front depth")
    ok = matte & np.isfinite(D_f) & (D_f > 0)
    if not ok.any():
        raise InputError("The depth of the subject's front is missing, so no back can be built.")
    if not ok[matte].all():  # a few unknown cells: borrow from the neighbours
        D_f = push_pull_fill(np.where(ok, D_f, 0.0).astype(np.float32), ok).astype(np.float64)
    return matte, np.where(matte, D_f, np.nan)


def _front_ext(front: FrontModel, D_f: np.ndarray) -> np.ndarray:
    """``D_ext`` (float64) with any missing matte value replaced by ``D_f``."""
    D_ext = _as_map(front.D_ext, D_f.shape, "extrapolated front depth")
    return np.where(np.isfinite(D_ext) & (D_ext > 0), D_ext, D_f)


def _color_map(colors: Any, shape: tuple[int, int], what: str = "colour") -> np.ndarray:
    """``Hm x Wm x 3`` uint8 or float [0, 1] colours as float32 in [0, 1]."""
    c = np.asarray(colors)
    if c.shape[:2] != shape or c.ndim != 3 or c.shape[2] < 3:
        raise InputError(f"The {what} map ({c.shape}) must be {shape[0]}x{shape[1]}x3.")
    c = c[..., :3]
    if c.dtype.kind in "ui":
        return c.astype(np.float32) / 255.0
    return np.clip(np.nan_to_num(c.astype(np.float32)), 0.0, 1.0)


def _nonneg(x: Any, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = math.nan
    if not (math.isfinite(v) and v >= 0):
        raise InputError(f"{what} must be a non-negative number (got {x!r}).")
    return v


def _stride(stride: Any) -> int:
    try:
        s = int(stride)
    except (TypeError, ValueError):
        s = 0
    if s < 1 or s != stride:
        raise InputError(f"The back Gaussian stride must be a whole number >= 1 (got {stride!r}).")
    return s


def _ray_xy(
    front: FrontModel,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """x / y components of the ``z = 1`` ray through mould column / row indices (also
    valid one cell outside the grid)."""
    f = float(front.cam.f)
    if not (math.isfinite(f) and f > 0):
        raise InputError("The camera focal length is unknown, so no back can be built.")
    cell = float(front.grid.cell_px)
    w2, h2 = front.cam.width / 2.0, front.cam.height / 2.0
    return (lambda c: ((c + 0.5) * cell - w2) / f), (lambda r: ((r + 0.5) * cell - h2) / f)


def _empty_cloud() -> SplatCloud:
    f32 = np.float32
    return SplatCloud(
        means=np.zeros((0, 3), f32),
        scales=np.zeros((0, 3), f32),
        quats=np.zeros((0, 4), f32),
        colors=np.zeros((0, 3), f32),
        opacity=np.zeros(0, f32),
    )


def _normals_of(sc: SplatCloud) -> np.ndarray:
    """Third local axis of every Gaussian (the surface normal of the flat ones built here)."""
    if len(sc) == 0:
        return np.zeros((0, 3), np.float32)
    return quat_to_rotmat(sc.quats)[:, :, 2].astype(np.float32)


def _normalize(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1)
    return v / np.maximum(n, 1e-300)[..., None], n


# --------------------------------------------------------------------------- prior
def _fit_kappa(D_s: np.ndarray, P: np.ndarray, D_f: np.ndarray, target: float) -> float:
    """``kappa`` in ``[0, _KAPPA_MAX]`` with ``median(max(D_s + kappa P, D_f + MIN_GAP_M) -
    D_f) == target`` (bisection: the median is non-decreasing in ``kappa``)."""
    if target <= 0 or not (P > 0).any():
        return 0.0
    if len(P) > _KAPPA_SAMPLES:  # a regular subsample has the same median for practical purposes
        k = math.ceil(len(P) / _KAPPA_SAMPLES)
        D_s, P, D_f = D_s[::k], P[::k], D_f[::k]
    floor = D_f + MIN_GAP_M

    def med(kappa: float) -> float:
        return float(np.median(np.maximum(D_s + kappa * P, floor) - D_f))

    if med(_KAPPA_MAX) < target:
        log.info("Back inflation clamped (kappa > %.1f); the silhouette is thin for its height", _KAPPA_MAX)
        return _KAPPA_MAX
    if med(0.0) >= target:
        return 0.0
    lo, hi = 0.0, _KAPPA_MAX
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if med(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class _Prior:
    Z: np.ndarray  # float64 back depth D_smooth + T (NaN outside the matte)
    T: np.ndarray  # float64 thickness term (NaN outside the matte)
    matte: np.ndarray
    D_f: np.ndarray  # float64, filled in the matte, NaN outside
    info: dict[str, Any]


def _inflation(front: FrontModel, canon: Canon, thickness: Any) -> _Prior:
    """The inflation prior and the checked front maps, see :func:`inflation_prior`."""
    thickness = _nonneg(thickness, "thickness (a ratio of the height)")
    matte, D_f = _front_maps(front)
    height = float(canon.height_s)
    if not (math.isfinite(height) and height > 0):
        raise InputError("The subject's height could not be measured, so no back can be built.")
    target = thickness * height
    prof32, dt32 = inflation_profile(matte)
    prof, dte = prof32.astype(np.float64), dt32.astype(np.float64)
    f_m = float(front.cam.f) / float(front.grid.cell_px)  # focal length in mould cells
    dfz = np.where(matte, D_f, 0.0)
    mpp = dfz / f_m
    # radius (cells) of the inscribed disc behind each cell, from prof^2 = dt (2 R - dt)
    R_m = np.where(dte > 0, 0.5 * (prof**2 / np.maximum(dte, 1e-12) + dte), 0.0) * mpp
    # A circular cross-section makes depth grow with width: a torso came out ~3x as thick as
    # it is.  Capping the depth semi-axis at half the median thickness gives wide parts an
    # elliptical section while round limbs keep the circle.
    cap = 0.5 * target
    squash = np.where(R_m > cap, cap / np.maximum(R_m, 1e-12), 1.0)
    P = squash * prof * mpp  # thickness per unit kappa (m)
    rows = np.flatnonzero(matte.any(axis=1))
    sigma = max(1.0, 0.04 * float(rows[-1] - rows[0] + 1))
    D_s = _masked_blur(dfz, matte, sigma).astype(np.float64)
    kappa = _fit_kappa(D_s[matte], P[matte], D_f[matte], target)
    T = np.where(matte, kappa * P, np.nan)
    return _Prior(
        Z=np.where(matte, D_s + kappa * P, np.nan),
        T=T,
        matte=matte,
        D_f=D_f,
        info={"kappa": float(kappa), "sigma_cells": float(sigma), "target_m": float(target)},
    )


def inflation_prior(front: FrontModel, canon: Canon, *, thickness: float) -> tuple[FloatMap, FloatMap]:
    """``(Z_prior, T)`` on the mould grid (float32, NaN outside the matte).

    ``T = kappa * prof * mpp`` where ``prof`` comes from ``billboard.inflation_profile(matte)``
    and ``mpp = D_f / f_m`` (metres per cell, ``f_m = cam.f / grid.cell_px``), except that
    the cross-section's depth semi-axis ``R * mpp`` (``R`` = the local inscribed-disc
    radius) is capped at ``thickness * height_s / 2``: wide parts get an ellipse instead
    of a circle.  ``Z_prior = D_smooth + T`` with ``D_smooth`` the masked blur of ``D_f``
    (sigma = 4 % of the silhouette height).  ``kappa`` (<= 4) makes the median of the
    thickness the shell actually gets, ``max(Z_prior, D_f + MIN_GAP_M) - D_f``, equal
    ``thickness * canon.height_s`` (the design fits ``median(T)``; fitting the clamped
    thickness keeps the user's "thickness" exact although ``D_smooth`` differs from
    ``D_f``)."""
    p = _inflation(front, canon, thickness)
    return p.Z.astype(np.float32), p.T.astype(np.float32)


# --------------------------------------------------------------------------- Poisson
def _boundary(m: np.ndarray) -> np.ndarray:
    """Cells of ``m`` with a 4-neighbour outside ``m`` (the grid border counts as outside)."""
    p = np.pad(m, 1)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def _block_sum(x: np.ndarray) -> np.ndarray:
    """2x2 block sums (odd sizes padded with zeros)."""
    h, w = x.shape
    x = np.pad(x, ((0, h % 2), (0, w % 2)))
    return x.reshape((h + 1) // 2, 2, (w + 1) // 2, 2).sum(axis=(1, 3))


def _block_mean(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Mean of ``x`` over the ``m`` cells of each 2x2 block (NaN where there are none)."""
    s = _block_sum(np.where(m, x, 0.0))
    n = _block_sum(m.astype(np.float64))
    return np.where(n > 0, s / np.maximum(n, 1.0), np.nan)


def _pcg(
    apply_a: Callable[[np.ndarray, np.ndarray], np.ndarray],
    b: np.ndarray,
    inv_diag: np.ndarray,
    x: np.ndarray,
    iters: int,
    cancel: CancelToken | None,
    tol: float = 1e-7,
) -> tuple[np.ndarray, int]:
    """Jacobi-preconditioned conjugate gradient for the SPD operator ``apply_a(x, out)``."""
    buf = np.empty_like(x)
    r = b - apply_a(x, buf)
    z = r * inv_diag
    p = z.copy()
    rz = float(np.vdot(r, z))
    stop = (tol * max(float(np.sqrt(np.vdot(b, b))), 1e-9)) ** 2
    done = 0
    while done < iters:
        if rz <= 0.0 or float(np.vdot(r, r)) <= stop:
            break
        if done % 16 == 0:
            _check(cancel)
        ap = apply_a(p, buf)
        pap = float(np.vdot(p, ap))
        if not pap > 0.0:
            break
        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap
        np.multiply(r, inv_diag, out=z)
        rz_new = float(np.vdot(r, z))
        p *= rz_new / rz
        p += z
        rz = rz_new
        done += 1
    return x, done


def _solve_level(
    m: np.ndarray,
    w: np.ndarray,
    rim: np.ndarray,
    rim_val: np.ndarray,
    anc: np.ndarray,
    anc_val: np.ndarray,
    y0: np.ndarray,
    lam_r: float,
    lam_s: float,
    lam_a: float,
    iters: int,
    cancel: CancelToken | None,
) -> tuple[np.ndarray, int]:
    """Solve ``(L_w + diag(lam_s + lam_r rim + lam_a anc)) y = lam_r rim rim_val + lam_a anc
    anc_val`` on the cells ``m`` (``y`` = 0 elsewhere)."""
    mf = m.astype(np.float64)
    wh = np.where(m[:, :-1] & m[:, 1:], np.minimum(w[:, :-1], w[:, 1:]), 0.0)
    wv = np.where(m[:-1, :] & m[1:, :], np.minimum(w[:-1, :], w[1:, :]), 0.0)
    d_extra = mf * lam_s + np.where(rim, lam_r, 0.0) + np.where(anc, lam_a, 0.0)
    rhs = np.where(rim, lam_r * rim_val, 0.0) + np.where(anc, lam_a * anc_val, 0.0)
    deg = d_extra.copy()
    deg[:, :-1] += wh
    deg[:, 1:] += wh
    deg[:-1, :] += wv
    deg[1:, :] += wv
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-300), 0.0)

    def apply_a(y: np.ndarray, out: np.ndarray) -> np.ndarray:
        np.multiply(d_extra, y, out=out)
        dh = wh * (y[:, 1:] - y[:, :-1])  # flux along each edge; zero across the mask border
        out[:, :-1] -= dh
        out[:, 1:] += dh
        dv = wv * (y[1:, :] - y[:-1, :])
        out[:-1, :] -= dv
        out[1:, :] += dv
        return out

    return _pcg(apply_a, rhs, inv, np.where(m, y0, 0.0), iters, cancel)


def poisson_heightfield(
    target: FloatMap,
    mask: np.ndarray,
    z_rim: FloatMap,
    *,
    weights: FloatMap | None = None,
    lam_rim: float = 10.0,
    lam_target: float = 0.01,
    iters: int = 150,
    levels: int = 3,
    anchor: FloatMap | None = None,
    lam_anchor: float = _ANCHOR_LAM,
    init: FloatMap | None = None,
    cancel: CancelToken | None = None,
) -> FloatMap:
    """Minimise ``sum_(p,q) w_pq (z_q - z_p - (T_q - T_p))^2 + lam_rim sum_boundary (z_p -
    z_rim_p)^2 + lam_target sum (z_p - T_p)^2`` over ``mask`` cells (4-neighbour edges
    inside the mask, ``w_pq = min(w_p, w_q)`` from ``weights``, default 1).  The boundary is
    the set of mask cells with a 4-neighbour outside the mask (the grid border counts as
    outside); boundary cells where ``z_rim`` is NaN get no rim term.  Several disconnected
    components and single cells are fine.

    Matrix-free Jacobi-preconditioned conjugate gradient, coarse to fine: ``levels``
    resolution levels (each a 2x2-block coarsening of the next, as long as the coarse grid
    keeps >= 4 cells across), at most ``iters`` iterations per level, warm start by
    upsampling.  The lambdas are rescaled per level so every level discretises the same
    continuous energy.

    Additions to the design signature (all optional): ``anchor`` - extra soft constraints
    ``lam_anchor * (z_p - anchor_p)^2`` on the cells where ``anchor`` is finite (the
    re-solve after clamping); ``init`` - start from this depth on the finest level and
    skip the coarse levels.  Non-finite ``target`` cells inside ``mask`` are filled from
    their neighbours.  Returns ``z`` (float32, NaN outside ``mask``)."""
    m_full = np.asarray(mask, dtype=bool)
    if m_full.ndim != 2:
        raise InputError("The back mask must be a 2-D map.")
    shape = (int(m_full.shape[0]), int(m_full.shape[1]))
    T_full = _as_map(target, shape, "back depth target")
    g_full = _as_map(z_rim, shape, "rim depth")
    w_full = None if weights is None else _as_map(weights, shape, "edge weight")
    a_full = None if anchor is None else _as_map(anchor, shape, "anchor depth")
    i_full = None if init is None else _as_map(init, shape, "initial depth")
    lam_rim = _nonneg(lam_rim, "lam_rim")
    lam_target = _nonneg(lam_target, "lam_target")
    lam_anchor = _nonneg(lam_anchor, "lam_anchor")
    if int(iters) < 1 or int(levels) < 1:
        raise InputError("iters and levels must be at least 1.")
    out = np.full(shape, np.nan, np.float32)
    if not m_full.any():
        return out
    _check(cancel)

    # everything happens on the mask's bounding box: a person covers a fraction of the photo
    rr = np.flatnonzero(m_full.any(axis=1))
    cc = np.flatnonzero(m_full.any(axis=0))
    box = (slice(int(rr[0]), int(rr[-1]) + 1), slice(int(cc[0]), int(cc[-1]) + 1))
    m = m_full[box]
    T = T_full[box]
    ok = m & np.isfinite(T)
    if not ok.any():
        raise InputError("The back depth target is empty, so no back can be built.")
    if not ok[m].all():
        T = push_pull_fill(np.where(ok, T, 0.0).astype(np.float32), ok).astype(np.float64)
    T = np.where(m, T, 0.0)
    w = np.ones(m.shape) if w_full is None else np.clip(np.nan_to_num(w_full[box], nan=1.0), 0.0, None)
    g = g_full[box]
    a = np.full(m.shape, np.nan) if a_full is None else a_full[box]

    # pyramid of (mask, target, weights, rim depth, anchor depth); coarse rim / anchor
    # values are block means over the fine cells that carry one
    n_lev = 1
    if i_full is None:
        while n_lev < int(levels) and min(m.shape) >= 8 * 2 ** (n_lev - 1):
            n_lev += 1
    pyr = [(m, T, w, g, a)]
    for _ in range(1, n_lev):
        m0, T0, w0, g0, a0 = pyr[-1]
        rim0 = _boundary(m0) & np.isfinite(g0)
        anc0 = m0 & np.isfinite(a0)
        m1 = _block_sum(m0.astype(np.float64)) > 0
        T1 = np.nan_to_num(_block_mean(T0, m0))
        w1 = np.nan_to_num(_block_mean(w0, m0), nan=1.0)
        pyr.append((m1, T1, w1, _block_mean(g0, rim0), _block_mean(a0, anc0)))

    if i_full is not None:
        ini = i_full[box]
        y = np.where(m & np.isfinite(ini), ini - T, 0.0)
    else:
        y = np.zeros(pyr[-1][0].shape)
    total_it = 0
    for k in range(n_lev - 1, -1, -1):
        mk, Tk, wk, gk, ak = pyr[k]
        if y.shape != mk.shape:  # nearest upsampling of the coarser correction
            y = np.repeat(np.repeat(y, 2, axis=0), 2, axis=1)[: mk.shape[0], : mk.shape[1]]
        rim = _boundary(mk) & np.isfinite(gk)
        anc = mk & np.isfinite(ak)
        s = float(2**k)  # per-cell terms scale with the cell area, boundary terms with its side
        y, it = _solve_level(
            mk,
            wk,
            rim,
            np.where(rim, gk - Tk, 0.0),
            anc,
            np.where(anc, ak - Tk, 0.0),
            y,
            lam_rim * s,
            lam_target * s * s,
            lam_anchor * s * s,
            int(iters),
            cancel,
        )
        total_it += it
    log.debug("Poisson back: %d cells, %d levels, %d CG iterations", int(m.sum()), n_lev, total_it)
    out[box] = np.where(m, T + y, np.nan).astype(np.float32)
    return out


# --------------------------------------------------------------------------- Gaussians
def gaussians_from_heightfield(
    z: FloatMap,
    mask: np.ndarray,
    front: FrontModel,
    colors: np.ndarray,
    *,
    stride: int = 1,
    opacity: float = BACK_OPACITY,
    cancel: CancelToken | None = None,
) -> tuple[SplatCloud, np.ndarray]:
    """One flat Gaussian per ``stride``-th mask cell at ``ray(p) * z(p)`` (frame C).

    Tangents by central differences (one-sided where a neighbour is outside the mask, flat
    at constant depth where both are), Gram-Schmidt: ``t1 = t_u^``, ``t2 = (t_v - (t_v.t1)
    t1)^``, ``n = t1 x t2``; if ``n . ray < 0`` negate ``t2`` and ``n`` so the normal points
    away from the camera (det stays +1).  Scales ``0.6*stride*|t_u|``, ``0.6*stride*|t_v_perp|``
    clamped to [1 mm, 5 cm], ``s3 = max(0.5 mm, 0.1*min)``.  ``colors`` is ``Hm x Wm x 3``
    uint8 (or float [0, 1]).  Cells are taken row-major where ``row % stride == col %
    stride == 0``.  ``cancel`` (optional addition) is checked per chunk of cells.
    Returns ``(cloud, normals)`` (normals float32 ``N x 3``)."""
    m = np.asarray(mask, dtype=bool)
    shape = _grid_shape(front)
    if m.shape != shape:
        raise InputError(f"The back mask ({m.shape}) does not match the mould grid {shape}.")
    zz = _as_map(z, shape, "back depth")
    s = _stride(stride)
    col = _color_map(colors, shape)
    op = float(opacity)
    if not (math.isfinite(op) and 0.0 <= op <= 1.0):
        raise InputError(f"opacity must be between 0 and 1 (got {opacity!r}).")
    rx, ry = _ray_xy(front)
    H, W = shape
    valid = m & np.isfinite(zz) & (zz > 0)
    sel = valid & (np.arange(H) % s == 0)[:, None] & (np.arange(W) % s == 0)[None, :]
    idx = np.flatnonzero(sel)
    n = len(idx)
    means = np.empty((n, 3), np.float32)
    scales = np.empty((n, 3), np.float32)
    quats = np.empty((n, 4), np.float32)
    normals = np.empty((n, 3), np.float32)

    def point(r: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        return np.stack([rx(c) * d, ry(r) * d, d], axis=-1)

    def tangent(r: np.ndarray, c: np.ndarray, zc: np.ndarray, dr: int, dc: int) -> np.ndarray:
        """d P / d cell along (dr, dc), from the valid neighbours."""
        r1, c1, r0, c0 = r + dr, c + dc, r - dr, c - dc
        in1 = (r1 < H) & (c1 < W)
        in0 = (r0 >= 0) & (c0 >= 0)
        ok1 = in1 & valid[np.minimum(r1, H - 1), np.minimum(c1, W - 1)]
        ok0 = in0 & valid[np.maximum(r0, 0), np.maximum(c0, 0)]
        z1 = np.where(ok1, zz[np.minimum(r1, H - 1), np.minimum(c1, W - 1)], zc)
        z0 = np.where(ok0, zz[np.maximum(r0, 0), np.maximum(c0, 0)], zc)
        flat = ~ok0 & ~ok1  # isolated along this axis: a step at constant depth
        hi = np.where(ok1 | flat, 1, 0)
        lo = np.where(ok0, -1, 0)
        p1 = point(r + hi * dr, c + hi * dc, z1)
        p0 = point(r + lo * dr, c + lo * dc, z0)
        return (p1 - p0) / (hi - lo)[:, None]

    for i0 in range(0, n, _CHUNK):
        _check(cancel)
        part = idx[i0 : i0 + _CHUNK]
        r, c = np.divmod(part, W)
        zc = zz[r, c]
        P = point(r, c, zc)
        t1, len_u = _normalize(tangent(r, c, zc, 0, 1))
        tv = tangent(r, c, zc, 1, 0)
        tvp = tv - (tv * t1).sum(1, keepdims=True) * t1
        t2, len_v = _normalize(tvp)
        ray_hat, _ = _normalize(P)
        bad = len_v <= 1e-9 * np.maximum(np.linalg.norm(tv, axis=1), 1e-300)
        if bad.any():  # t_v along t_u (degenerate fold): any tangent across t_u will do
            t2[bad] = _normalize(np.cross(ray_hat[bad], t1[bad]))[0]
            len_v = np.where(bad, np.linalg.norm(tv, axis=1), len_v)
        nrm = np.cross(t1, t2)
        flip = (nrm * ray_hat).sum(1) < 0
        t2[flip] *= -1.0
        nrm[flip] *= -1.0
        s1 = np.clip(0.6 * s * len_u, _SCALE_MIN, _SCALE_MAX)
        s2 = np.clip(0.6 * s * len_v, _SCALE_MIN, _SCALE_MAX)
        sl = slice(i0, i0 + len(part))
        means[sl] = P
        scales[sl] = np.stack([s1, s2, np.maximum(_S3_MIN, 0.1 * np.minimum(s1, s2))], axis=1)
        quats[sl] = rotmat_to_quat(np.stack([t1, t2, nrm], axis=2))
        normals[sl] = nrm
    sc = SplatCloud(
        means=means,
        scales=scales,
        quats=quats,
        colors=np.ascontiguousarray(col.reshape(-1, 3)[idx], dtype=np.float32),
        opacity=np.full(n, op, np.float32),
    )
    return sc, normals


# --------------------------------------------------------------------------- rim wall
def _contour_points(matte: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Boundary cells along every contour of ``matte`` (outer outlines and holes), about
    one per ``spacing`` cells of arc length: ``(points (K, 2) as (col, row), outward
    image-plane normals (K, 2) as (du, dv))``."""
    contours, _ = cv2.findContours(matte.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), np.int64), np.zeros((0, 2))
    pts = np.concatenate([c.reshape(-1, 2) for c in contours]).astype(np.int64)
    lens = np.array([len(c) for c in contours], np.int64)
    starts = np.cumsum(lens) - lens
    cid = np.repeat(np.arange(len(lens)), lens)
    local = np.arange(len(pts)) - starts[cid]
    L, S = lens[cid], starts[cid]

    # tangent over +-2 chain steps (smooths the 8-connected staircase), normal = its perpendicular
    tan = (pts[S + (local + 2) % L] - pts[S + (local - 2) % L]).astype(np.float64)
    nrm = np.stack([tan[:, 1], -tan[:, 0]], axis=1)
    nrm, ln = _normalize(nrm)
    nrm[ln < 1e-9] = (1.0, 0.0)  # single-cell contour: any direction
    # orient outward: the blurred matte is lower on the outside
    blur = cv2.GaussianBlur(matte.astype(np.float32), (0, 0), 1.0)
    h, w = matte.shape

    def sample(q: np.ndarray) -> np.ndarray:
        qi = np.rint(q).astype(np.int64)
        return blur[np.clip(qi[:, 1], 0, h - 1), np.clip(qi[:, 0], 0, w - 1)]

    flip = sample(pts + 1.5 * nrm) > sample(pts - 1.5 * nrm)
    nrm[flip] *= -1.0

    # about one point per `spacing` cells of arc length along each contour
    prev = S + (local - 1) % L
    step = np.linalg.norm((pts - pts[prev]).astype(np.float64), axis=1)
    step[local == 0] = 0.0
    cum = np.cumsum(step)
    cum -= cum[S]
    b = np.floor(cum / spacing + 1e-9)
    keep = (local == 0) | (b != b[prev])
    return pts[keep], nrm[keep]


def rim_wall(
    front: FrontModel,
    z_rim: FloatMap,
    *,
    colors_front: np.ndarray,
    colors_back: np.ndarray | None = None,
    spacing_cells: float = 1.0,
    cancel: CancelToken | None = None,
) -> SplatCloud:
    """Gaussians along the matte contour (``cv2.findContours`` on the mould matte, outer
    outlines and holes) from ``D_ext`` to ``z_rim`` along each boundary ray, every
    ~``spacing_cells`` cells of metric size (both along the contour and along the ray,
    including both ends).  Axes along the ray and the contour (0.6 x spacing); normal =
    outward contour normal (perpendicular to the ray, det +1); colour lerps from the front
    edge colour (``colors_front`` at the boundary cell) to the back edge colour
    (``colors_back``, default the front colour).  No Gaussians where ``z_rim - D_ext <= 1
    cm`` or ``z_rim`` is NaN.  ``cancel`` is an optional addition."""
    matte = _front_matte(front)
    shape = matte.shape
    zr = _as_map(z_rim, shape, "rim depth")
    D_ext = _as_map(front.D_ext, shape, "extrapolated front depth")
    cf = _color_map(colors_front, shape, "front colour")
    cb = cf if colors_back is None else _color_map(colors_back, shape, "back colour")
    spacing = float(spacing_cells)
    if not (math.isfinite(spacing) and spacing > 0):
        raise InputError(f"The rim wall spacing must be a positive number of cells (got {spacing_cells!r}).")
    rx, ry = _ray_xy(front)
    f, cell = float(front.cam.f), float(front.grid.cell_px)
    _check(cancel)

    pts, n2 = _contour_points(matte, spacing)
    c, r = pts[:, 0], pts[:, 1]
    z0, z1 = D_ext[r, c], zr[r, c]
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(z0) & np.isfinite(z1) & (z0 > 0) & (z1 - z0 > RIM_MIN_M)
    if not ok.any():
        return _empty_cloud()
    c, r, z0, z1, n2 = c[ok], r[ok], z0[ok], z1[ok], n2[ok]

    # frame per boundary ray: a1 = the ray, a2 = the contour direction, n = outward
    rvec = np.stack([rx(c), ry(r), np.ones(len(c))], axis=1)
    a1, rlen = _normalize(rvec)
    out3 = np.stack([n2[:, 0], n2[:, 1], np.zeros(len(c))], axis=1)
    tan3 = np.stack([-n2[:, 1], n2[:, 0], np.zeros(len(c))], axis=1)
    a2, _ = _normalize(tan3 - (tan3 * a1).sum(1, keepdims=True) * a1)
    nrm = np.cross(a1, a2)
    flip = (nrm * out3).sum(1) < 0
    a2[flip] *= -1.0
    nrm[flip] *= -1.0
    q = rotmat_to_quat(np.stack([a1, a2, nrm], axis=2))

    # samples along each ray, both ends included
    length = (z1 - z0) * rlen
    pitch = spacing * cell * 0.5 * (z0 + z1) / f  # metric size of `spacing` cells there
    k_n = np.clip(np.ceil(length / pitch).astype(np.int64) + 1, 2, _WALL_MAX_STEPS)
    pid = np.repeat(np.arange(len(c)), k_n)
    first = np.cumsum(k_n) - k_n
    t = (np.arange(len(pid)) - first[pid]) / (k_n[pid] - 1)
    zs = z0[pid] + t * (z1 - z0)[pid]
    s_ray = np.clip(0.6 * length / (k_n - 1), _SCALE_MIN, _SCALE_MAX)[pid]
    s_con = np.clip(0.6 * spacing * cell * zs / f, _SCALE_MIN, _SCALE_MAX)
    col = (1.0 - t)[:, None] * cf[r, c][pid] + t[:, None] * cb[r, c][pid]
    return SplatCloud(
        means=(rvec[pid] * zs[:, None]).astype(np.float32),
        scales=np.stack([s_ray, s_con, np.maximum(_S3_MIN, 0.1 * np.minimum(s_ray, s_con))], 1).astype(
            np.float32
        ),
        quats=q[pid].astype(np.float32),
        colors=np.clip(col, 0.0, 1.0).astype(np.float32),
        opacity=np.full(len(pid), BACK_OPACITY, np.float32),
    )


# --------------------------------------------------------------------------- colours
def _colors(
    front: FrontModel, matte: np.ndarray, dt: np.ndarray, back: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(back colours, edge colours, band width)``, uint8 ``Hm x Wm x 3``.

    The band is the ``_BAND_PX`` cells (at a 1024-cell grid, i.e. a fixed fraction of the
    photo, so the colours do not change with the quality) just inside the silhouette,
    without the outermost ring of cells, which mixes in the background.  Edge colours are
    the photo colours with that ring (and the outside) push-pull filled from the band only,
    so nothing from the middle of the front (a face) leaks into the rim.  Back colours:
    ``"mirror"`` = the edge colours, ``"diffuse"`` = the band's push-pull fill everywhere,
    darkened (billboard's ``_BAND_PX`` / ``_BACK_DARKEN``), so no face appears on the back
    of a head."""
    rgb = np.asarray(front.rgb)
    if rgb.ndim != 3 or rgb.shape[:2] != matte.shape or rgb.shape[2] < 3:
        raise InputError(
            f"The photo colours on the mould grid ({rgb.shape}) do not match the grid {matte.shape}."
        )
    rgb = np.ascontiguousarray(rgb[..., :3])
    if rgb.dtype != np.uint8:
        rgb = (_color_map(rgb, matte.shape) * 255.0 + 0.5).astype(np.uint8)
    band_w = max(2.0, _BAND_PX * max(matte.shape) / _BAND_REF_LONG)
    inner = matte & (dt > 1.5)
    band = inner & (dt <= band_w + 1.0)
    for cand in (band, inner, matte):  # thin subjects: whatever there is
        if cand.any():
            band = cand
            break
    fill = push_pull_fill(rgb, band)
    edge = np.where(inner[..., None], rgb, fill)
    if back == "mirror":
        return edge, edge, band_w
    return np.clip(fill.astype(np.float32) * _BACK_DARKEN + 0.5, 0, 255).astype(np.uint8), edge, band_w


# --------------------------------------------------------------------------- back sources
def _spacing(front: FrontModel, z_b: np.ndarray, matte: np.ndarray, stride: int) -> float | None:
    """Typical distance (m) between neighbouring back Gaussians: ``stride`` cells at the back's
    median depth (None when there is no back)."""
    z = z_b[matte & np.isfinite(z_b)]
    if not len(z):
        return None
    return float(stride * float(front.grid.cell_px) * float(np.median(z)) / float(front.cam.f))


def _finish(
    front: FrontModel,
    z_b: np.ndarray,
    matte: np.ndarray,
    colors: np.ndarray,
    colors_front: np.ndarray,
    z_rim: np.ndarray,
    stride: int,
    cancel: CancelToken | None,
) -> tuple[SplatCloud, np.ndarray, int, int]:
    """Back Gaussians + rim wall, concatenated, with their normals."""
    sc_b, n_b = gaussians_from_heightfield(z_b, matte, front, colors, stride=stride, cancel=cancel)
    _check(cancel)
    wall = rim_wall(
        front, z_rim, colors_front=colors_front, colors_back=colors, spacing_cells=stride, cancel=cancel
    )
    sc = concat_splats([sc_b, wall]) if len(wall) else sc_b
    normals = np.concatenate([n_b, _normals_of(wall)]) if len(wall) else n_b
    return sc, normals, len(sc_b), len(wall)


def mould_back(
    front: FrontModel,
    canon: Canon,
    target: FloatMap,
    *,
    colors: np.ndarray,
    z_rim: FloatMap | None = None,
    clip: FloatMap | None = None,
    stride: int = 1,
    weights: FloatMap | None = None,
    source: str = "mould",
    hull: HullIntervals | None = None,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> BackPart:
    """Shared back builder (design 3.6): Poisson-solve ``target`` on the matte with rim
    values ``z_rim`` (default ``D_ext``), clamp to ``[D_f + MOULD_GAP_M, min(clip -
    MIN_GAP_M, hull z_out + 2 cm)]`` (the front bound wins where they cross), re-solve once
    from there with the clamped cells as soft constraints (lambda 10) and clamp again.
    Without a ``hull``, the median thickness ``z_b - D_f`` must lie in ``[0.5, 2] *
    canon.T_prior``; otherwise the inflation prior is used instead (``diag["fallback"]``
    and a plain ``diag["warning"]``).  Then back Gaussians (``colors``, ``Hm x Wm x 3``) +
    a rim wall where ``z_rim - D_ext > 1 cm``.  ``kind="mould"``, ``back_z`` = the final
    depth (NaN outside the matte).

    ``hull`` and ``progress`` are optional additions to the skeleton (view-based sources,
    milestone M4); ``spacing_m`` is set as in :func:`shell_back`."""
    rep = _as_reporter(progress)
    s = _stride(stride)
    matte, D_f = _front_maps(front)
    shape = matte.shape
    D_ext = _front_ext(front, D_f)
    g = D_ext if z_rim is None else _as_map(z_rim, shape, "rim depth")
    _color_map(colors, shape)  # fail before the solve
    hi = np.full(shape, np.inf)
    if clip is not None:
        cl = _as_map(clip, shape, "clip depth")
        hi = np.where(np.isfinite(cl), np.minimum(hi, cl - MIN_GAP_M), hi)
    if hull is not None:
        z_out = _as_map(hull.z_out, shape, "hull exit depth")
        hv = np.asarray(hull.valid, dtype=bool) & np.isfinite(z_out)
        hi = np.where(hv, np.minimum(hi, z_out + _HULL_SLACK_M), hi)
    lo = D_f + MOULD_GAP_M
    _check(cancel)

    def clamp(z: np.ndarray) -> np.ndarray:
        return np.where(matte, np.maximum(np.minimum(z, hi), lo), np.nan)

    rep(0.05, "Building the back surface")
    z1 = poisson_heightfield(target, matte, g, weights=weights, cancel=cancel).astype(np.float64)
    zc = clamp(z1)
    clamped = matte & (np.abs(zc - z1) > 1e-6)
    resolved = False
    if clamped.any():
        rep(0.4, "Building the back surface")
        z2 = poisson_heightfield(
            target,
            matte,
            g,
            weights=weights,
            anchor=np.where(clamped, zc, np.nan),
            init=zc,
            cancel=cancel,
        ).astype(np.float64)
        zc = clamp(z2)
        resolved = True
    diag: dict[str, Any] = {
        "source": source,
        "clamped_cells": int(clamped.sum()),
        "resolved": resolved,
        "fallback": False,
    }
    T_prior = float(canon.T_prior)
    med = float(np.median((zc - D_f)[matte]))
    diag["median_thickness_m"] = med
    height = float(canon.height_s)
    sane = 0.5 * T_prior <= med <= 2.0 * T_prior
    if hull is None and math.isfinite(T_prior) and T_prior > 0 and height > 0 and not sane:
        prior = _inflation(front, canon, T_prior / height)
        zc = clamp(prior.Z)
        diag.update(
            fallback=True, kappa=prior.info["kappa"], median_thickness_m=float(np.median((zc - D_f)[matte]))
        )
        diag["warning"] = (
            f"The predicted back was implausibly {'thin' if med < 0.5 * T_prior else 'thick'} "
            f"({med * 100:.0f} cm instead of about {T_prior * 100:.0f} cm); the built-in rounded back was used."
        )
        log.warning(
            "Mould back (%s): median thickness %.3f m vs prior %.3f m; using the prior", source, med, T_prior
        )
    _check(cancel)
    rep(0.7, "Filling the sides")
    _, edge, _ = _colors(front, matte, _dist(matte), "mirror")
    z_rim_eff = np.where(_boundary(matte), g, np.nan)
    sc, normals, n_back, n_wall = _finish(front, zc, matte, colors, edge, z_rim_eff, s, cancel)
    spacing = _spacing(front, zc, matte, s)
    diag.update(n_back=n_back, n_wall=n_wall, stride=s, spacing_m=spacing)
    rep(1.0, "Back ready")
    return BackPart(
        sc=sc,
        kind="mould",
        source=source,
        ok=len(sc) > 0,
        reason="" if len(sc) else "the back has no Gaussians",
        normals=normals,
        back_z=zc.astype(np.float32),
        hull=hull,
        diag=diag,
        spacing_m=spacing,
    )


def shell_back(
    front: FrontModel,
    canon: Canon,
    *,
    thickness: float = 0.11,
    back: str = "diffuse",
    stride: int = 1,
    clip: FloatMap | None = None,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> BackPart:
    """The built-in back (no AI, design 3.5.1): ``inflation_prior`` used directly as the back
    depth (no solve), clamped to ``>= D_f + 5 mm`` (and ``<= clip - 5 mm`` where a ``clip``
    map is given - an optional addition for the TRELLIS closure under-layer; the front
    bound wins where they cross); colours ``"diffuse"`` = push-pull fill of a band just
    inside the silhouette, x 0.9 (no face on the back of a head), or ``"mirror"`` = the
    photo colours.  Plus the rim wall from the front's edge (``D_ext``) to the shell's own
    edge depth, where they are more than 1 cm apart.  ``source="shell"``; ``spacing_m`` =
    ``stride`` mould cells at the back's median depth (for the floater filter)."""
    rep = _as_reporter(progress)
    if back not in BACK_MODES:
        raise InputError(f"Unknown back colour mode {back!r}; use 'diffuse' or 'mirror'.")
    s = _stride(stride)
    _check(cancel)
    rep(0.0, "Building the back")
    prior = _inflation(front, canon, thickness)
    matte, D_f, info = prior.matte, prior.D_f, prior.info
    z_b = prior.Z
    if clip is not None:
        cl = _as_map(clip, matte.shape, "clip depth")
        z_b = np.where(np.isfinite(cl), np.minimum(z_b, cl - MIN_GAP_M), z_b)
    z_b = np.where(matte, np.maximum(z_b, D_f + MIN_GAP_M), np.nan)
    _check(cancel)

    rep(0.35, "Colouring the back")
    colors, edge, band_w = _colors(front, matte, _dist(matte), back)
    _check(cancel)

    rep(0.6, "Building the back")
    z_rim = np.where(_boundary(matte), z_b, np.nan)  # the shell's own edge depth
    sc, normals, n_back, n_wall = _finish(front, z_b, matte, colors, edge, z_rim, s, cancel)
    med = float(np.median((z_b - D_f)[matte]))
    spacing = _spacing(front, z_b, matte, s)
    diag = {
        "source": "shell",
        "kappa": info["kappa"],
        "sigma_cells": info["sigma_cells"],
        "target_thickness_m": info["target_m"],
        "median_thickness_m": med,
        "back": back,
        "band_cells": band_w if back == "diffuse" else 0.0,
        "stride": s,
        "spacing_m": spacing,
        "n_back": n_back,
        "n_wall": n_wall,
        "fallback": False,
    }
    log.info(
        "Shell back: %d Gaussians + %d rim wall, kappa %.2f, median thickness %.3f m (%s)",
        n_back,
        n_wall,
        info["kappa"],
        med,
        back,
    )
    rep(1.0, "Back ready")
    return BackPart(
        sc=sc,
        kind="mould",
        source="shell",
        ok=len(sc) > 0,
        reason="" if len(sc) else "the subject is too small to build a back for",
        normals=normals,
        back_z=z_b.astype(np.float32),
        diag=diag,
        spacing_m=spacing,
    )


__all__ = [
    "BACK_OPACITY",
    "DETAIL_M",
    "MIN_GAP_M",
    "MOULD_GAP_M",
    "RIM_MIN_M",
    "BackPart",
    "HullIntervals",
    "gaussians_from_heightfield",
    "inflation_prior",
    "mould_back",
    "poisson_heightfield",
    "rim_wall",
    "shell_back",
]
