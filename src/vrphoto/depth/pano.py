"""Depth for 360 equirect panoramas from a perspective depth estimator.

The panorama is split into six overlapping cube faces (``90 + overlap_deg`` degrees each),
each face is run through the estimator, every face's disparity is aligned to its
neighbours with a per-face affine map (scale + shift, least squares on the overlap
regions, in disparity space) and the aligned faces are blended back to the equirect with
raised-cosine weights that fall to zero at each face's border (so there are no seams).

Perspective estimators return inverse *z* (planar) depth, while a panorama needs inverse
*distance*; the two differ by ``cos(theta)`` (angle from the face axis).  Two alignment
models are therefore fitted and the one that explains the overlaps better is used:

* ``plain``:  ``s = a_f * d + b_f`` (the face output already is a radial quantity);
* ``radial``: ``s = cos(theta) * (a_f * d + b_f)`` (the face output is affine in 1/z).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..core.errors import InferenceError, InputError
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, DepthResult, Intrinsics, ProgressReporter
from ..geometry.projection import (
    CUBE_FACES,
    _as_float,
    _as_int,
    _eq_dirs_tile,
    _equirect_to_persp_R,
    _face_K,
    _fit_equirect_source,
    _resample_equirect,
    _rotate,
    _to_rgb_u8,
    cube_face_rotation,
)

if TYPE_CHECKING:  # pragma: no cover
    from .base import DepthEstimator

log = get_logger(__name__)

__all__ = ["estimate_equirect"]

_FIT_W = 720  # equirect width of the alignment grid (2 px / deg)
_MIN_PAIR_PX = 24  # minimum shared pixels for a face pair to constrain the fit
_RIDGE = 1e-4  # relative ridge regularisation toward identity
_RADIAL_GAIN = 0.8  # radial model must reduce the overlap residual by 20% to be chosen
_TILE_PX = 1 << 19


class _FaceGeom:
    """Projection of world directions into one overlapping cube face."""

    def __init__(self, key: str, K: Intrinsics, fov_deg: float, overlap_deg: float):
        self.key = key
        self.K = K
        self.Rt = cube_face_rotation(key).T
        self.n = K.width
        self.tmax = math.tan(math.radians(fov_deg) / 2.0)
        self.edge0 = math.radians(45.0 - overlap_deg / 2.0)
        self.edge1 = math.radians(45.0 + overlap_deg / 2.0)

    def project(self, x: np.ndarray, y: np.ndarray, z: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (u, v, cos_theta, blend weight, inside-rendered-face mask) for directions."""
        xc, yc, zc = _rotate(x, y, z, self.Rt)
        front = zc > 1e-6
        zs = np.where(front, zc, 1.0)
        a = np.where(front, xc / zs, 0.0)
        b = np.where(front, yc / zs, 0.0)
        inside = front & (np.abs(a) <= self.tmax) & (np.abs(b) <= self.tmax)
        w = self._ramp(np.arctan(np.abs(a))) * self._ramp(np.arctan(np.abs(b)))
        w = np.where(inside, w, 0.0)
        ct = 1.0 / np.sqrt(1.0 + a * a + b * b)
        u = (self.K.fx * a + self.K.cx).astype(np.float32)
        v = (self.K.fy * b + self.K.cy).astype(np.float32)
        return u, v, ct, w, inside

    def _ramp(self, ang: np.ndarray) -> np.ndarray:
        if self.edge1 - self.edge0 < 1e-9:
            return (ang <= self.edge0 + 1e-12).astype(np.float64)
        t = np.clip((ang - self.edge0) / (self.edge1 - self.edge0), 0.0, 1.0)
        return 0.5 * (1.0 + np.cos(math.pi * t))


def _sample(d: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear lookup of a face map at face pixel coordinates (float64 result)."""
    return cv2.remap(d, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)


def _features(model: str, d: np.ndarray, ct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel features (f1, f2) such that s = a * f1 + b * f2."""
    if model == "radial":
        return ct * d, ct
    return d, np.ones_like(d)


def _fit(model: str, samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], area: np.ndarray,
         ref: int) -> tuple[np.ndarray, float]:
    """Least-squares per-face (a, b) making overlapping faces agree.

    ``samples[f] = (d, cos_theta, inside)`` on the fit grid.  Face ``ref`` has ``a = 1``
    (and ``b = 0`` for the plain model, whose shift is otherwise free); a small ridge toward
    identity keeps everything well posed.  Returns (params n x 2, relative residual)."""
    n = len(samples)
    A = np.zeros((2 * n, 2 * n))
    feats = [_features(model, d, ct) for d, ct, _ in samples]
    pairs: list[tuple[int, int, np.ndarray]] = []
    for f in range(n):
        for g in range(f + 1, n):
            both = samples[f][2] & samples[g][2]
            if int(both.sum()) < _MIN_PAIR_PX:
                continue
            wt = area[both]
            pf = np.stack([feats[f][0][both], feats[f][1][both]], axis=1)
            pg = np.stack([feats[g][0][both], feats[g][1][both]], axis=1)
            sf, sg = slice(2 * f, 2 * f + 2), slice(2 * g, 2 * g + 2)
            A[sf, sf] += (pf * wt[:, None]).T @ pf
            A[sg, sg] += (pg * wt[:, None]).T @ pg
            cross = (pf * wt[:, None]).T @ pg
            A[sf, sg] -= cross
            A[sg, sf] -= cross.T
            pairs.append((f, g, both))
    if not pairs:
        raise InferenceError("Panorama depth: the cube faces do not overlap; increase overlap_deg")
    x0 = np.tile([1.0, 0.0], n)
    scale = float(np.mean(np.diag(A))) or 1.0
    mu = _RIDGE * scale
    fixed = [2 * ref] + ([2 * ref + 1] if model == "plain" else [])
    free = [i for i in range(2 * n) if i not in fixed]
    Aff = A[np.ix_(free, free)] + mu * np.eye(len(free))
    rhs = mu * x0[free] - A[np.ix_(free, fixed)] @ x0[fixed]
    x = x0.copy()
    try:
        x[free] = np.linalg.solve(Aff, rhs)
    except np.linalg.LinAlgError:
        x[free] = np.linalg.lstsq(Aff, rhs, rcond=None)[0]
    params = x.reshape(n, 2)
    # relative residual: overlap disagreement / spread of the aligned values
    res, wsum, vals = 0.0, 0.0, []
    for f, g, both in pairs:
        wt = area[both]
        sf = params[f, 0] * feats[f][0][both] + params[f, 1] * feats[f][1][both]
        sg = params[g, 0] * feats[g][0][both] + params[g, 1] * feats[g][1][both]
        res += float((wt * (sf - sg) ** 2).sum())
        wsum += float(wt.sum())
        vals.append(sf)
    allv = np.concatenate(vals)
    spread = float(np.std(allv)) or 1e-12
    return params, math.sqrt(res / max(wsum, 1e-12)) / spread


def _blend(disp: list[np.ndarray], geoms: list[_FaceGeom], params: np.ndarray, model: str,
           out_w: int, out_h: int, cancel: CancelToken | None) -> np.ndarray:
    """Weighted blend of the aligned faces onto an ``out_w x out_h`` equirect (row tiles)."""
    lon = (-math.pi, math.pi)
    lat = (-math.pi / 2, math.pi / 2)
    out = np.empty((out_h, out_w), np.float32)
    rows = max(1, _TILE_PX // max(1, out_w))
    for j0 in range(0, out_h, rows):
        if cancel is not None:
            cancel.check()
        j1 = min(out_h, j0 + rows)
        x, y, z = _eq_dirs_tile(out_w, out_h, lon, lat, j0, j1, 0, out_w)
        num = np.zeros((j1 - j0, out_w))
        den = np.zeros((j1 - j0, out_w))
        for f, g in enumerate(geoms):
            u, v, ct, w, _ = g.project(x, y, z)
            if not (w > 0).any():
                continue
            f1, f2 = _features(model, _sample(disp[f], u, v), ct)
            num += w * (params[f, 0] * f1 + params[f, 1] * f2)
            den += w
        out[j0:j1] = num / np.maximum(den, 1e-12)
    return out


def _normalize(s: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(s, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full(s.shape, 0.5, np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _clean_disparity(res: object, face: int, key: str) -> np.ndarray:
    """Face disparity from an estimator result (DepthResult or bare array) as a finite
    float32 ``face x face`` map (resized if needed, non-finite values -> median)."""
    d = getattr(res, "disparity", res)
    try:
        d = np.asarray(d, dtype=np.float32)
    except (TypeError, ValueError):
        raise InferenceError(f"Depth estimator returned an invalid map for cube face {key}") from None
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2 or d.size == 0:
        raise InferenceError(f"Depth estimator returned an invalid map for cube face {key}")
    if d.shape != (face, face):
        d = cv2.resize(d, (face, face), interpolation=cv2.INTER_LINEAR)
    bad = ~np.isfinite(d)
    if bad.all():
        raise InferenceError(f"Depth estimator returned no valid values for cube face {key}")
    if bad.any():
        d = np.where(bad, float(np.median(d[~bad])), d).astype(np.float32)
    return np.ascontiguousarray(d)


def estimate_equirect(eq: RGB, est: DepthEstimator, *, face: int = 518, overlap_deg: float = 10.0,
                      progress: ProgressReporter | None = None,
                      cancel: CancelToken | None = None) -> DepthResult:
    """Estimate a disparity map (1 = nearest) for a full 360x180 equirect panorama.

    ``est`` is any perspective :class:`~vrphoto.depth.base.DepthEstimator`; it is run on six
    cube faces of ``face`` x ``face`` pixels and ``90 + overlap_deg`` degrees FOV (face
    orientation as :func:`vrphoto.geometry.projection.equirect_to_cubemap`; at least a few
    pixels of overlap are always rendered for the alignment).  Faces are aligned with
    per-face affine maps fitted on the overlaps and blended with cosine weights over the
    ``overlap_deg`` band around each cube edge.
    Returns a :class:`DepthResult` at the equirect resolution with ``model_id``
    ``"pano:<est.id>"``; the disparity is proportional to inverse distance up to an affine
    map, normalised to [0, 1].
    """
    pr = progress or ProgressReporter.null()
    if not isinstance(eq, np.ndarray):
        raise InputError("Panorama depth needs an HxWx3 RGB equirect image")
    # gray / RGBA / uint16 / float [0,1] are converted (the estimators take uint8 RGB)
    eq = _to_rgb_u8(eq, "Panorama")
    h, w = eq.shape[:2]
    if h < 8 or w < 16:
        raise InputError(f"Panorama is too small ({w}x{h})")
    if abs(w / h - 2.0) > 0.1:
        raise InputError(f"Panorama depth needs a full 360x180 equirect (2:1); got {w}x{h}",
                         hint="Use normal photo depth for non-360 images.")
    face = _as_int(face, "Cube face size")
    if face < 16:
        raise InputError(f"Cube face size must be at least 16 px (got {face})")
    ov = _as_float(overlap_deg, "overlap_deg")
    if not (math.isfinite(ov) and 0.0 <= ov <= 60.0):
        raise InputError(f"overlap_deg must be between 0 and 60 (got {overlap_deg})")
    est_id = str(getattr(est, "id", type(est).__name__))
    # faces need a few pixels of overlap for the alignment even when blending is (almost) hard
    fov = 90.0 + max(ov, 1.0, 4.0 * 90.0 / face)
    K = _face_K(face, fov)
    geoms = [_FaceGeom(k, K, fov, ov) for k in CUBE_FACES]

    # anti-alias prefilter once (each face render would otherwise area-downscale a large
    # panorama again); the per-face renders then see an already small enough source
    src = _fit_equirect_source(eq, min(K.fx, K.fy), 2.0 * math.pi)
    disp: list[np.ndarray] = []
    n = len(CUBE_FACES)
    for i, key in enumerate(CUBE_FACES):
        if cancel is not None:
            cancel.check()
        pr(0.75 * i / n, f"Depth: cube face {i + 1}/{n}")
        img = _equirect_to_persp_R(src, K, cube_face_rotation(key))
        res = est.estimate(img, progress=pr.sub(0.75 * i / n, 0.75 * (i + 1) / n), cancel=cancel)
        disp.append(_clean_disparity(res, face, key))
    del src
    if cancel is not None:
        cancel.check()

    pr(0.76, "Depth: aligning faces")
    # the alignment grid samples the FACE maps, so its size is independent of the equirect
    # (tying it to a small panorama's width left too few overlap samples to fit)
    fit_w = _FIT_W
    fit_h = fit_w // 2
    lon = (-math.pi, math.pi)
    lat = (-math.pi / 2, math.pi / 2)
    x, y, z = _eq_dirs_tile(fit_w, fit_h, lon, lat, 0, fit_h, 0, fit_w)
    area = np.broadcast_to(np.cos(np.radians(90.0 - (np.arange(fit_h) + 0.5) / fit_h * 180.0))[:, None],
                           (fit_h, fit_w))
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for f, g in enumerate(geoms):
        u, v, ct, _, inside = g.project(x, y, z)
        # strictly inside (half a pixel margin) so samples never extrapolate
        inside &= (u >= 0) & (u <= face - 1) & (v >= 0) & (v <= face - 1)
        samples.append((_sample(disp[f], u, v), ct, inside))
    ref = CUBE_FACES.index("+z")
    fits: dict[str, tuple[np.ndarray, float]] = {}
    for model in ("plain", "radial"):
        fits[model] = _fit(model, samples, area, ref)
    model = "radial" if fits["radial"][1] < _RADIAL_GAIN * fits["plain"][1] else "plain"
    params = fits[model][0]
    log.info("Panorama depth: %s alignment (overlap residual plain %.4f, radial %.4f)",
             model, fits["plain"][1], fits["radial"][1])
    if np.any(params[:, 0] <= 0):
        log.warning("Panorama depth: face alignment produced a non-positive scale; using identity "
                    "for those faces")
        bad = params[:, 0] <= 0
        params[bad] = (1.0, 0.0)

    pr(0.8, "Depth: blending faces")
    f_px = K.fx
    work_w = int(min(w, max(64, round(2.0 * math.pi * 2.0 * f_px))))
    work_h = max(4, int(round(work_w * h / w)))
    s = _blend(disp, geoms, params, model, work_w, work_h, cancel)
    d = _normalize(s)
    pr(0.95, "Depth: resampling")
    disparity = np.clip(_resample_equirect(d, w, h, (-180.0, 180.0)), 0.0, 1.0).astype(np.float32)
    pr(1.0, "Depth: done")
    return DepthResult(disparity=disparity, model_id=f"pano:{est_id}")
