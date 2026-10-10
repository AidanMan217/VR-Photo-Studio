"""Depth-map post-processing: normalisation, guided filtering, temporal alignment, edges.

All functions take and return ``float32`` maps; disparity is in [0, 1] with 1 = nearest.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.types import RGB, FloatMap


def normalize(raw: FloatMap, lo_pct: float = 2.0, hi_pct: float = 98.0) -> FloatMap:
    """Robustly rescale ``raw`` to [0, 1] using its ``lo_pct``/``hi_pct`` percentiles.

    Non-finite values are treated as the minimum (far).  A (nearly) constant map
    returns 0.5 everywhere.  Orientation is preserved (larger in -> larger out).
    """
    a = np.asarray(raw, dtype=np.float32)
    if a.ndim != 2:
        a = np.squeeze(a)
        if a.ndim != 2:
            raise ValueError(f"normalize expects a 2-D map, got shape {np.shape(raw)}")
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, np.float32)
    vals = a[finite]
    if vals.size > 2_000_000:  # percentiles on a strided sample are plenty accurate
        vals = vals[:: vals.size // 1_000_000]
    lo, hi = np.percentile(vals, [lo_pct, hi_pct]).astype(np.float64)
    if not hi - lo > 1e-12 * max(1.0, abs(hi)):
        lo, hi = float(vals.min()), float(vals.max())
        if not hi - lo > 1e-12 * max(1.0, abs(hi)):
            return np.full(a.shape, 0.5, np.float32)
    out = np.where(finite, a, lo).astype(np.float64)
    out = (out - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _guide_float(guide: np.ndarray) -> np.ndarray:
    g = np.asarray(guide)
    if g.dtype == np.uint8:
        g = g.astype(np.float32) / 255.0
    elif g.dtype == np.uint16:
        g = g.astype(np.float32) / 65535.0
    else:
        g = g.astype(np.float32)
    if g.ndim == 3 and g.shape[2] == 4:
        g = g[..., :3]
    if g.ndim == 3 and g.shape[2] == 1:
        g = g[..., 0]
    return g


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


FAST_GF_MIN_PIXELS = 4_000_000  # above this, refine/finalize use the fast guided filter
FAST_GF_TARGET_PIXELS = 2_000_000  # ... computing coefficients at about this many pixels


def auto_subsample(n_pixels: int, r: int) -> int:
    """Subsampling factor for the fast guided filter (1 = exact) for an image size.

    Large photos (12+ MP) would need >1 GB of float32 temporaries for the exact colour
    guided filter; computing the linear coefficients on a ~2 MP grid gives visually
    identical depth (He & Sun 2015) with a fraction of the memory and time.
    """
    if n_pixels <= FAST_GF_MIN_PIXELS:
        return 1
    s = int(np.ceil(np.sqrt(n_pixels / FAST_GF_TARGET_PIXELS)))
    return max(1, min(s, int(r)))


def _gf_coeffs(p: np.ndarray, g: np.ndarray, r: int, eps: float) -> tuple[list[np.ndarray], np.ndarray]:
    """Box-averaged linear coefficients ``(mean_a per guide channel, mean_b)``."""
    if g.ndim == 2:
        mean_i = _box(g, r)
        mean_p = _box(p, r)
        cov_ip = _box(g * p, r) - mean_i * mean_p
        var_i = _box(g * g, r) - mean_i * mean_i
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        return [_box(a, r)], _box(b, r)

    # colour guide
    ir, ig, ib = g[..., 0], g[..., 1], g[..., 2]
    mean_r, mean_g, mean_b = _box(ir, r), _box(ig, r), _box(ib, r)
    mean_p = _box(p, r)
    cov_rp = _box(ir * p, r) - mean_r * mean_p
    cov_gp = _box(ig * p, r) - mean_g * mean_p
    cov_bp = _box(ib * p, r) - mean_b * mean_p
    var_rr = _box(ir * ir, r) - mean_r * mean_r + eps
    var_rg = _box(ir * ig, r) - mean_r * mean_g
    var_rb = _box(ir * ib, r) - mean_r * mean_b
    var_gg = _box(ig * ig, r) - mean_g * mean_g + eps
    var_gb = _box(ig * ib, r) - mean_g * mean_b
    var_bb = _box(ib * ib, r) - mean_b * mean_b + eps
    # inverse of the symmetric 3x3 covariance via cofactors
    inv_rr = var_gg * var_bb - var_gb * var_gb
    inv_rg = var_gb * var_rb - var_rg * var_bb
    inv_rb = var_rg * var_gb - var_gg * var_rb
    inv_gg = var_rr * var_bb - var_rb * var_rb
    inv_gb = var_rb * var_rg - var_rr * var_gb
    inv_bb = var_rr * var_gg - var_rg * var_rg
    det = var_rr * inv_rr + var_rg * inv_rg + var_rb * inv_rb
    det = np.where(np.abs(det) < 1e-20, 1e-20, det)
    a_r = (inv_rr * cov_rp + inv_rg * cov_gp + inv_rb * cov_bp) / det
    a_g = (inv_rg * cov_rp + inv_gg * cov_gp + inv_gb * cov_bp) / det
    a_b = (inv_rb * cov_rp + inv_gb * cov_gp + inv_bb * cov_bp) / det
    b = mean_p - a_r * mean_r - a_g * mean_g - a_b * mean_b
    return [_box(a_r, r), _box(a_g, r), _box(a_b, r)], _box(b, r)


def guided_filter(
    src: FloatMap, guide: RGB | FloatMap, r: int = 8, eps: float = 1e-3, *, subsample: int = 1
) -> FloatMap:
    """Edge-preserving guided filter (He et al. 2010) of ``src`` steered by ``guide``.

    ``guide`` may be an RGB image (uint8 or float in [0,1]) -> colour guided filter with a
    per-pixel 3x3 covariance, or a single-channel map -> grey guided filter.  ``r`` is the
    window radius in pixels, ``eps`` the regularisation (in guide units squared; larger =
    smoother).  Implemented with ``cv2.boxFilter`` (O(1) per pixel in ``r``).

    ``subsample`` > 1 (optional extension) selects the *fast* guided filter (He & Sun
    2015): the coefficients are computed on a grid ``subsample`` times coarser and
    bilinearly upsampled, which cuts memory and time by ~``subsample**2`` on big images.
    """
    p = np.asarray(src, dtype=np.float32)
    if p.ndim != 2:
        raise ValueError(f"guided_filter: src must be HxW, got {p.shape}")
    g = _guide_float(guide)
    if g.shape[:2] != p.shape:
        raise ValueError(f"guided_filter: guide {g.shape[:2]} and src {p.shape} sizes differ")
    r = int(max(1, min(r, max(p.shape) // 2)))
    eps = float(max(eps, 1e-12))
    h, w = p.shape
    s = int(max(1, min(int(subsample), r)))
    if s > 1:
        lw, lh = max(1, int(round(w / s))), max(1, int(round(h / s)))
        p_lo = cv2.resize(p, (lw, lh), interpolation=cv2.INTER_AREA)
        g_lo = cv2.resize(g, (lw, lh), interpolation=cv2.INTER_AREA)
        r_lo = int(max(1, min(round(r / s), max(lh, lw) // 2)))
        mean_a, mean_b = _gf_coeffs(p_lo, g_lo, r_lo, eps)
        del p_lo, g_lo

        def up(x: np.ndarray) -> np.ndarray:
            return cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mean_a, mean_b = _gf_coeffs(p, g, r, eps)

        def up(x: np.ndarray) -> np.ndarray:
            return x

    q = up(mean_b).astype(np.float32, copy=True)
    channels = [g] if g.ndim == 2 else [g[..., c] for c in range(3)]
    for a, gc in zip(mean_a, channels):
        q += up(a) * gc
    return q


def gray_guide(img: RGB) -> FloatMap:
    """Luma of an RGB image as float32 in [0, 1] (guide for :func:`guided_filter`)."""
    g = _guide_float(img)
    if g.ndim == 2:
        return g
    return (0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]).astype(np.float32)


def refine(disparity: FloatMap, img: RGB, strength: float = 1.0) -> FloatMap:
    """Edge-aware smoothing of a [0,1] disparity map guided by the colour image.

    ``strength`` 0 returns the input unchanged, 1 is the default (colour guided filter,
    radius ~0.6% of the long side), 2 smooths about twice as wide.  The result snaps
    depth discontinuities to colour edges and removes blocky upsampling artefacts.
    """
    d = np.asarray(disparity, dtype=np.float32)
    s = float(max(0.0, min(strength, 4.0)))
    if s <= 1e-3:
        return d.copy()
    h, w = d.shape
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    r = max(1, int(round(0.006 * max(h, w) * s)))
    eps = 1e-4 * (1.0 + 4.0 * s)
    out = guided_filter(d, img, r=r, eps=eps, subsample=auto_subsample(h * w, r))
    if s < 1.0:
        out = d + s * (out - d)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def temporal_align(prev: FloatMap | None, cur: FloatMap, ema: float = 0.6) -> FloatMap:
    """Stabilise per-frame relative disparity for video.

    ``cur`` is mapped onto ``prev`` with a robust least-squares scale + shift (relative
    depth models drift in scale/offset between frames), then blended with an
    exponential moving average where ``ema`` is the weight of the previous frame.  The
    blend weight falls off where the aligned frames disagree strongly (moving objects),
    so motion does not leave trails.  Returns ``cur`` unchanged when ``prev`` is None.
    """
    c = np.asarray(cur, dtype=np.float32)
    if prev is None:
        return c.copy()
    p = np.asarray(prev, dtype=np.float32)
    if p.shape != c.shape:
        p = cv2.resize(p, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
    ema = float(min(max(ema, 0.0), 0.99))

    step = max(1, int(np.sqrt(c.size / 65536)))
    cs = c[::step, ::step].ravel().astype(np.float64)
    ps = p[::step, ::step].ravel().astype(np.float64)
    ok = np.isfinite(cs) & np.isfinite(ps)
    cs, ps = cs[ok], ps[ok]
    a, b = 1.0, 0.0
    if cs.size >= 16 and cs.std() > 1e-6:
        sel = np.ones(cs.size, bool)
        for _ in range(3):  # IRLS-style trimming of the worst 20% residuals
            A = np.stack([cs[sel], np.ones(int(sel.sum()))], 1)
            sol, *_ = np.linalg.lstsq(A, ps[sel], rcond=None)
            a, b = float(sol[0]), float(sol[1])
            res = np.abs(a * cs + b - ps)
            sel = res <= np.quantile(res, 0.8)
            if sel.sum() < 16:
                break
        if not np.isfinite(a) or a <= 1e-3:  # degenerate / inverted fit: match moments instead
            a = float(ps.std() / max(cs.std(), 1e-6))
            b = float(ps.mean() - a * cs.mean())
    aligned = a * c + b
    diff = np.abs(aligned - p)
    w = ema * np.exp(-((diff / 0.1) ** 2))
    out = w * p + (1.0 - w) * aligned
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0).astype(np.float32)


def depth_edges(inv_z: FloatMap, thresh: float) -> np.ndarray:
    """Boolean map of depth discontinuities.

    A pixel is an edge if the absolute difference to any 4-neighbour exceeds ``thresh``
    (in the units of ``inv_z``); both sides of a discontinuity are marked.
    """
    d = np.asarray(inv_z, dtype=np.float32)
    e = np.zeros(d.shape, bool)
    dx = np.abs(np.diff(d, axis=1)) > thresh
    dy = np.abs(np.diff(d, axis=0)) > thresh
    e[:, :-1] |= dx
    e[:, 1:] |= dx
    e[:-1, :] |= dy
    e[1:, :] |= dy
    return e


def upsample(d: FloatMap, size_hw: tuple[int, int]) -> FloatMap:
    """Bicubic resize of a map to ``(H, W)`` (area when shrinking)."""
    h, w = size_hw
    if d.shape == (h, w):
        return d.astype(np.float32, copy=False)
    interp = cv2.INTER_AREA if (d.shape[0] > h and d.shape[1] > w) else cv2.INTER_CUBIC
    return cv2.resize(d.astype(np.float32), (w, h), interpolation=interp)


def finalize_disparity(rel: FloatMap, img: RGB, *, refine_guided: bool = True) -> tuple[FloatMap, FloatMap]:
    """Algorithm A for relative disparity at model resolution.

    ``d = normalize(rel)``; bicubic upsample to the image size; grey guided filter
    (r ~ 0.006 W, eps 1e-3).  Returns ``(disparity [0,1] at image res, raw upsampled)``.
    """
    h, w = img.shape[:2]
    d = normalize(rel)
    d = upsample(d, (h, w))
    raw = upsample(np.nan_to_num(np.asarray(rel, np.float32), nan=0.0, posinf=0.0, neginf=0.0), (h, w))
    if refine_guided:
        r = max(1, int(round(0.006 * w)))
        d = guided_filter(d, gray_guide(img), r=r, eps=1e-3, subsample=auto_subsample(h * w, r))
    return np.clip(d, 0.0, 1.0).astype(np.float32), raw.astype(np.float32)


__all__ = [
    "normalize",
    "guided_filter",
    "refine",
    "temporal_align",
    "depth_edges",
    "gray_guide",
    "upsample",
    "auto_subsample",
    "finalize_disparity",
]
