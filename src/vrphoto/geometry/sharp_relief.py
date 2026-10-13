"""SHARP relief: a back view's surface detail from a SHARP run on that view (M6, ``back_relief="sharp"``).

SHARP 360 can run SHARP (the ``sharp`` add-on, list input) on the back view it got (an AI
view or the user's back photo).  SHARP's cloud of that picture is in the picture's own
camera frame (OpenCV axes, metres, ``u = f*X/Z + W/2``, ``v = f*Y/Z + H/2``), so its nearest
surface per pixel is a depth map of the back as the back camera saw it.
:func:`sharp_relief` turns it into the disparity map ``views_lift.view_based_back`` takes
from ``depth_fn`` (``S x S`` in [0, 1], 1 = nearest the back camera) - the same contract as
the app's depth model, with SHARP's sharper geometry.

Steps (numpy / OpenCV, CPU, about a second for SHARP's 1.2 M Gaussians at 512 px):

1. The Gaussians in front of the camera with opacity >= ``min_opacity`` are projected into
   the output grid (the picture's frame resampled to ``out_size``: output pixel ``(i, j)``
   covers picture pixels ``u in [j W/Wo, (j+1) W/Wo)``, like ``cv2.resize`` of the picture).
2. A point z-buffer keeps the nearest centre per pixel (``register.point_zbuffer``; each
   centre covers a disc of 1.5x the median on-screen size of the Gaussians, at most 3 px, so
   a sparse cloud closes up) - SHARP's first layer, the surface it saw.
3. Region: ``alpha >= 0.5`` when the view's matte is given (strongly recommended: SHARP also
   models the background), else the covered pixels (closed 5 x 5).  Pixels without a hit
   in the region and pixels lying more than ``max(6 cm, 4 % z)`` behind their 5 x 5 median
   (background seen through a gap between the Gaussians) are push-pull filled, then a 3 x 3
   median removes single-pixel spikes.
4. Disparity ``1 / z`` normalised by its 0.5 / 99.5 percentiles inside the region and
   clipped to [0, 1]; outside the region it is push-pull extrapolated from the region (so a
   consumer whose matte is slightly larger sees no cliff).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.errors import InputError
from ..core.fill import push_pull_fill
from ..core.log import get_logger
from ..core.types import CancelToken, SplatCloud
from .mesh import _check
from .register import point_zbuffer
from .splat import read_splats

log = get_logger(__name__)

MIN_OPACITY = 0.2  # translucent Gaussians do not define the surface
MAX_RADIUS_PX = 3.0
RADIUS_GAIN = 1.5  # disc radius = this x the median on-screen 1-sigma size
GAP_TOL_M = 0.06  # "seen through a gap": this far behind the 5 x 5 median (or 4 % of z)
GAP_TOL_FRAC = 0.04
Z_MIN = 1e-3


def _out_shape(out_size: Any) -> tuple[int, int]:
    if isinstance(out_size, (tuple, list)) and len(out_size) == 2:
        h, w = out_size
    else:
        h = w = out_size
    try:
        H, W = int(h), int(w)
    except (TypeError, ValueError) as e:
        raise InputError(f"Invalid relief size {out_size!r}: give a whole number of pixels.") from e
    if H < 2 or W < 2 or H > 8192 or W > 8192:
        raise InputError(f"Invalid relief size {out_size!r}: use 2-8192 pixels.")
    return H, W


def _positive(value: Any, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if not math.isfinite(v) or v <= 0:
        raise InputError(f"SHARP's {what} must be a positive number (got {value!r}).")
    return v


def _region_from_alpha(alpha: Any, shape: tuple[int, int]) -> np.ndarray:
    a = np.asarray(alpha)
    if a.ndim == 3:
        a = a[..., 0]
    if a.ndim != 2 or a.size == 0:
        raise InputError(f"The view's matte must be an H x W map (got shape {a.shape}).")
    a = np.nan_to_num(a.astype(np.float32), nan=0.0)
    if a.size and float(a.max()) > 1.0 + 1e-6:
        a = a / 255.0
    H, W = shape
    if a.shape != (H, W):
        interp = cv2.INTER_AREA if a.shape[0] > H and a.shape[1] > W else cv2.INTER_LINEAR
        a = cv2.resize(a, (W, H), interpolation=interp)
    return a >= 0.5


def sharp_relief(
    ply_path: str | Path | SplatCloud,
    *,
    f_px: float,
    width: int,
    height: int,
    out_size: int | tuple[int, int],
    alpha: Any = None,
    min_opacity: float = MIN_OPACITY,
    diag: dict | None = None,
    cancel: CancelToken | None = None,
) -> np.ndarray:
    """Disparity map ``(Ho, Wo)`` float32 in [0, 1] (1 = nearest the view's camera) of the SHARP
    cloud of one back view.

    ``ply_path``: SHARP's PLY of the view (or the cloud itself); ``f_px``, ``width``,
    ``height``: SHARP's outputs for that picture (``outs["f_pxs"][i]`` ...); ``out_size``:
    the output size (``S`` -> ``S x S``, or ``(H, W)``); the output is the picture's frame
    resampled to it.  ``alpha``: the view's matte (``HxW`` in the picture's or the output's
    frame, any size - resized; float [0, 1] or 0-255) restricting the surface and the
    normalisation to the subject.  ``diag`` (a dict) receives JSON-ready statistics.
    Raises :class:`InputError` for bad arguments, an unreadable file or a cloud with no
    usable surface in the picture."""
    f = _positive(f_px, "focal length (f_px)")
    W, H = _positive(width, "picture width"), _positive(height, "picture height")
    Ho, Wo = _out_shape(out_size)
    _check(cancel)
    sc = ply_path if isinstance(ply_path, SplatCloud) else read_splats(ply_path)
    _check(cancel)
    P = np.asarray(sc.means, np.float64).reshape(-1, 3)
    op = np.asarray(sc.opacity, np.float64).ravel()
    sx, sy = Wo / W, Ho / H
    with np.errstate(invalid="ignore", divide="ignore"):
        keep = np.isfinite(P).all(axis=1) & (P[:, 2] > Z_MIN) & (op >= float(min_opacity))
        Pk = P[keep]
        z = Pk[:, 2]
        u = (f * Pk[:, 0] / z + W / 2.0) * sx
        v = (f * Pk[:, 1] / z + H / 2.0) * sy
        inside = (u >= 0) & (u < Wo) & (v >= 0) & (v < Ho)
    if not inside.any():
        raise InputError("SHARP's cloud of the back view has no visible surface in the picture.",
                         hint="Check that the SHARP outputs (f_px, width, height) belong to this picture.")
    # disc radius from the Gaussians' on-screen size (middle scale: the in-plane size of a flat
    # Gaussian and the across-ray size of a ray-aligned "card")
    s_mid = np.sort(np.asarray(sc.scales, np.float64)[keep][inside], axis=1)[:, 1]
    sigma_px = s_mid * f / z[inside] * math.sqrt(sx * sy)
    sig = float(np.median(sigma_px[np.isfinite(sigma_px)])) if np.isfinite(sigma_px).any() else 0.0
    radius = float(np.clip(RADIUS_GAIN * sig, 0.0, MAX_RADIUS_PX))
    zbuf, _ = point_zbuffer(u, v, z, (Ho, Wo), radius=radius)
    _check(cancel)
    hit = np.isfinite(zbuf)
    if alpha is not None:
        region = _region_from_alpha(alpha, (Ho, Wo))
        if not region.any():
            raise InputError("The back view's matte is empty, so it has no relief.")
    else:
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        region = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_CLOSE, k5).astype(bool)
    known = hit & region
    if not known.any():
        raise InputError("SHARP's cloud does not cover the subject in the back view.",
                         hint="Check that the matte and the SHARP outputs belong to this picture.")
    depth = np.where(known, zbuf, 0.0).astype(np.float32)
    filled = push_pull_fill(depth, known)
    med = cv2.medianBlur(filled, 5)
    gap = known & (filled > med + np.maximum(GAP_TOL_M, GAP_TOL_FRAC * med))
    # the outline band: its 5 x 5 median is half background, so compare it with the surface
    # extrapolated from the interior instead (cards and background just outside the subject)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    interior = known & ~gap & cv2.erode(region.astype(np.uint8), k5, borderType=cv2.BORDER_CONSTANT,
                                        borderValue=0).astype(bool)
    if interior.sum() >= 16:
        ref = push_pull_fill(np.where(interior, depth, 0.0).astype(np.float32), interior)
        gap |= known & ~interior & (depth > ref + np.maximum(GAP_TOL_M, GAP_TOL_FRAC * ref))
    known_ok = known & ~gap
    if not known_ok.any():  # pragma: no cover - a 5x5 median always leaves some pixels
        known_ok = known
    filled = cv2.medianBlur(push_pull_fill(np.where(known_ok, depth, 0.0).astype(np.float32), known_ok), 3)
    _check(cancel)
    disp = 1.0 / np.maximum(filled.astype(np.float64), Z_MIN)
    vals = disp[region]
    lo, hi = (float(t) for t in np.percentile(vals, [0.5, 99.5]))
    flat = not (hi - lo > 1e-9 * max(1.0, abs(hi)))
    d = np.zeros((Ho, Wo), np.float32) if flat else np.clip((disp - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if not region.all():
        d = push_pull_fill(np.where(region, d, 0.0).astype(np.float32), region)
    d = np.clip(d, 0.0, 1.0).astype(np.float32)
    if diag is not None:
        zr = filled[region]
        diag.update(
            n_points=int(len(P)),
            n_used=int(inside.sum()),
            radius_px=round(radius, 3),
            region_px=int(region.sum()),
            coverage=round(float(known.sum()) / max(1, int(region.sum())), 4),
            gap_px=int(gap.sum()),
            z_range_m=[round(float(np.percentile(zr, 0.5)), 4), round(float(np.percentile(zr, 99.5)), 4)],
            flat=bool(flat),
            shape=[Ho, Wo],
            matte=alpha is not None,
        )
    log.debug("SHARP relief %dx%d: %d of %d Gaussians, radius %.2f px, %d gap px, coverage %.3f",
              Wo, Ho, int(inside.sum()), len(P), radius, int(gap.sum()),
              float(known.sum()) / max(1, int(region.sum())))
    return d


__all__ = ["MIN_OPACITY", "sharp_relief"]
