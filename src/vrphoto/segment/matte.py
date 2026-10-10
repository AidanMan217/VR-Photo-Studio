"""Matte post-processing: threshold, keep the main subject, fill holes, feather, refine.

All functions take and return ``float32 HxW`` alpha in [0, 1] (uint8 0..255 and bool
inputs are accepted and converted).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.errors import InputError
from ..core.types import FloatMap

__all__ = ["clean_matte", "refine_matte", "largest_component", "fill_mask_holes", "feather_mask", "to_alpha"]


def to_alpha(alpha: np.ndarray) -> FloatMap:
    """Convert a bool / uint8 (0..255) / float alpha map to ``float32`` in [0, 1]."""
    if not isinstance(alpha, np.ndarray):
        raise InputError(f"Alpha must be a numpy array, got {type(alpha).__name__}.")
    a = alpha
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise InputError(f"Alpha must be a single-channel HxW map, got shape {alpha.shape}.")
    if a.dtype == np.bool_:
        return a.astype(np.float32)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if np.issubdtype(a.dtype, np.integer):
        return np.clip(a.astype(np.float32), 0.0, 1.0)
    return np.clip(np.nan_to_num(a.astype(np.float32), nan=0.0), 0.0, 1.0)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected component of a bool mask."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return mask.astype(bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def fill_mask_holes(
    mask: np.ndarray, max_hole_frac: float | None = None, min_hole_px: int = 16
) -> np.ndarray:
    """Fill background regions fully enclosed by the mask (not touching the image border).

    With ``max_hole_frac`` only holes of at most ``max(min_hole_px, max_hole_frac * mask
    area)`` pixels are filled, so real gaps (e.g. between an arm and the body) survive."""
    m = mask.astype(bool)
    inv = (~m).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if n <= 1:
        return m
    border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    is_hole = np.ones(n, bool)
    is_hole[0] = False  # label 0 = the mask itself
    is_hole[border] = False
    if max_hole_frac is not None:
        limit = max(float(min_hole_px), float(max_hole_frac) * float(m.sum()))
        is_hole &= stats[:, cv2.CC_STAT_AREA] <= limit
    if not is_hole.any():
        return m
    return m | is_hole[labels]


def feather_mask(mask: np.ndarray, feather_px: float) -> FloatMap:
    """Soft alpha from a bool mask: a linear ramp ``2*feather_px`` wide centred on the
    mask boundary (the 0.5 iso-line stays on the boundary)."""
    m = mask.astype(np.uint8)
    if feather_px <= 0 or not m.any() or m.all():
        return m.astype(np.float32)
    dt_in = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dt_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sd = np.where(m > 0, dt_in - 0.5, 0.5 - dt_out)
    return np.clip(0.5 + sd / (2.0 * float(feather_px)), 0.0, 1.0).astype(np.float32)


def clean_matte(
    alpha: FloatMap,
    *,
    threshold: float = 0.5,
    feather_px: float = 2,
    fill_holes: bool = True,
    largest_only: bool = True,
    max_hole_frac: float | None = 0.02,
) -> FloatMap:
    """Clean a raw segmentation matte.

    1. threshold (``alpha >= threshold``);
    2. optionally keep only the largest connected component;
    3. optionally fill enclosed holes (only holes up to ``max_hole_frac`` of the subject
       area, so real gaps such as between arm and body stay open; ``None`` fills all);
    4. feather the edge with a ``2*feather_px`` wide linear ramp centred on the boundary.

    Returns ``float32 HxW`` in [0, 1].  An empty matte returns zeros.
    """
    a = to_alpha(alpha)
    if not 0.0 <= float(threshold) <= 1.0:
        raise InputError(f"Matte threshold must be in [0, 1], got {threshold}.")
    m = a >= float(threshold) if threshold > 0 else a > 0
    if not m.any():
        return np.zeros(a.shape, np.float32)
    if largest_only:
        m = largest_component(m)
    if fill_holes:
        m = fill_mask_holes(m, max_hole_frac)
    return feather_mask(m, float(feather_px))


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, cv2.CV_32F, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


#: Above this many pixels the guided filter runs subsampled ("fast guided filter").
GUIDED_MAX_PX = 4_000_000


def _guided_coeffs(i: np.ndarray, p: np.ndarray, r: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mi, mp = _box(i, r), _box(p, r)
    cov = _box(i * p, r) - mi * mp
    var = _box(i * i, r) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return _box(a, r), _box(b, r)


def guided_filter_gray(
    guide: np.ndarray, src: np.ndarray, r: int, eps: float, *, max_px: int | None = GUIDED_MAX_PX
) -> np.ndarray:
    """He et al. guided filter with a single-channel float32 guide in [0, 1].

    Images larger than ``max_px`` use the fast guided filter (He & Sun 2015): the linear
    coefficients are computed on a copy subsampled by ``k`` (radius ``r/k``) and upsampled
    bilinearly, so peak memory stays at a few full-size arrays even for 50+ MP photos."""
    i = guide.astype(np.float32, copy=False)
    p = src.astype(np.float32, copy=False)
    h, w = i.shape[:2]
    k = 1
    if max_px is not None and h * w > max_px:
        k = int(np.ceil(np.sqrt(h * w / float(max_px))))
    if k <= 1 or r < 2 * k:
        a, b = _guided_coeffs(i, p, r, eps)
        return a * i + b
    sw, sh = max(1, int(round(w / k))), max(1, int(round(h / k)))
    a, b = _guided_coeffs(
        cv2.resize(i, (sw, sh), interpolation=cv2.INTER_AREA),
        cv2.resize(p, (sw, sh), interpolation=cv2.INTER_AREA),
        max(1, int(round(r / k))),
        eps,
    )
    out = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    out *= i
    out += cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def refine_matte(
    alpha: FloatMap, img: np.ndarray, *, radius: int | None = None, eps: float = 1e-3
) -> FloatMap:
    """Edge-aware refinement of a (coarse or upsampled) matte with a guided filter so its
    edges snap to the image edges.  ``radius`` defaults to ~0.4% of the long side."""
    a = to_alpha(alpha)
    if img.ndim != 3 or img.shape[:2] != a.shape:
        raise InputError("refine_matte: image and alpha sizes differ.")
    h, w = a.shape
    r = int(radius) if radius is not None else max(2, int(round(0.004 * max(h, w))))
    gray = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    out = guided_filter_gray(gray, a, r, eps)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
