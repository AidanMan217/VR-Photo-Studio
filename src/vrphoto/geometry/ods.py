"""Omnidirectional stereo (ODS) from a mono 360 equirect plus a per-pixel distance map.

Algorithm G (docs/CONTRACTS.md): a scene point at longitude ``lambda``, latitude ``phi`` and
distance ``D`` has horizontal radius ``rho = D cos(phi)`` and height ``y = D sin(phi)``.
Each eye looks along rays tangent to a circle of radius ``r = (ipd/2) * w(phi)`` with the
pole fade ``w = clip((phi_hi - |phi|) / (phi_hi - phi_lo), 0, 1)`` (default 45..75 deg), so
the point appears at

    lambda' = lambda + e * asin(min(r / rho, .99))       e = +1 left eye, -1 right eye
    phi'    = atan2(y, sqrt(max(rho^2 - r^2, eps)))

A point straight ahead therefore appears shifted RIGHT in the left eye and LEFT in the right
eye, and there is no parallax at the poles.  Each eye is rendered by a backward warp in
equirect space: the fixed-point iteration ``p <- u - delta(p)`` (``delta`` = forward
displacement sampled bilinearly with longitude wrap) from two initialisations, keeping the
converged solution nearest to the viewer; the residual becomes a coverage alpha
``clip(1.5 - r, 0, 1)`` and disocclusions are filled from the background side with the
given inpainter (or push-pull).
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from ..core.errors import InferenceError, InputError
from ..core.fill import push_pull_fill
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, FloatMap, ProgressReporter
from .projection import _resample_equirect

log = get_logger(__name__)

__all__ = ["ods_pair", "ods_displacement"]

_ITERS = 8  # fixed-point iterations per initialisation
_CONVERGED_PX = 0.5
_TILE_PX = 1 << 19  # target pixels per warp tile
_FAR_M = 1.0e6  # distance used for invalid / infinite depth
_EPS = 1e-12
_ASIN_CAP = 0.99


def _pole_weight(phi: np.ndarray, pole_fade: tuple[float, float]) -> np.ndarray:
    lo, hi = (math.radians(a) for a in pole_fade)
    return np.clip((hi - np.abs(phi)) / (hi - lo), 0.0, 1.0)


def _check_pole_fade(pole_fade: Any) -> tuple[float, float]:
    try:
        lo, hi = float(pole_fade[0]), float(pole_fade[1])
    except (TypeError, ValueError, IndexError):
        raise InputError("pole_fade must be a (start, end) pair of latitudes in degrees") from None
    if not (0.0 <= lo < hi <= 90.0):
        raise InputError(f"pole_fade must satisfy 0 <= start < end <= 90 (got {lo}, {hi})")
    return lo, hi


def _check_ipd(ipd_m: Any) -> float:
    try:
        ipd = float(ipd_m)
    except (TypeError, ValueError):
        raise InputError(f"ipd_m must be a number of metres (got {ipd_m!r})") from None
    if not (math.isfinite(ipd) and 0.0 <= ipd <= 1.0):
        raise InputError(f"ipd_m must be between 0 and 1 metre (got {ipd_m})")
    return ipd


def _prepare_distance(dist_m: np.ndarray, h: int, w: int, min_m: float) -> np.ndarray:
    """Validated float32 distance map at ``h x w``: NaN / <= 0 / inf -> far, resized on
    inverse distance with longitude wrap, clamped to ``>= min_m``."""
    try:
        d = np.array(dist_m, dtype=np.float32)  # always a private copy (modified below)
    except (TypeError, ValueError):
        raise InputError("The distance map must be a numeric HxW array of metres") from None
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[..., 0]
    if d.ndim != 2 or d.size == 0:
        raise InputError("The distance map must be an HxW array of metres")
    bad = np.isnan(d) | (d <= 0.0)  # +inf is valid: infinitely far
    if bad.all():
        raise InputError("The distance map has no valid (positive) values")
    if bad.any():
        d[bad] = _FAR_M
    np.minimum(d, _FAR_M, out=d)
    if d.shape != (h, w):
        log.debug("Resizing distance map %s to %dx%d", d.shape, w, h)
        # resample INVERSE distance (depth edges stay sharp-ish, far stays far) and wrap
        # longitude so the +-180 deg seam gets no parallax discontinuity
        inv = _resample_equirect(np.ascontiguousarray(1.0 / d), w, h, (-180.0, 180.0))
        d = (1.0 / np.maximum(inv, 1.0 / _FAR_M)).astype(np.float32)
    n_near = int((d < min_m).sum())
    if n_near:
        log.info("ODS: clamping %d pixel(s) nearer than %.3f m", n_near, min_m)
        np.maximum(d, min_m, out=d)
    return d


def ods_displacement(dist_m: FloatMap, ipd_m: float = 0.064, *,
                     pole_fade: tuple[float, float] = (45.0, 75.0)) -> tuple[np.ndarray, np.ndarray]:
    """Forward ODS displacement of every equirect pixel for the LEFT eye, in pixels.

    Returns ``(dx, dy)`` float32 HxW: a source pixel at ``(i, j)`` appears at
    ``(i + dx, j + dy)`` in the left eye and at ``(i - dx, j + dy)`` in the right eye
    (``dx`` > 0 = shifted right).  ``dist_m`` is the distance along each pixel's ray
    (metres, the equirect resolution defines the grid; inf / NaN / <= 0 = infinitely far,
    nearer than ``ipd_m`` is clamped).
    """
    lo, hi = _check_pole_fade(pole_fade)
    ipd = _check_ipd(ipd_m)
    d = np.asarray(dist_m)
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise InputError("The distance map must be an HxW array of metres")
    h, w = d.shape
    d = _prepare_distance(d, h, w, min_m=max(ipd, 1e-3))
    dx = np.empty((h, w), np.float32)
    dy = np.empty((h, w), np.float32)
    rows = max(1, _TILE_PX // max(1, w))
    for j0 in range(0, h, rows):
        j1 = min(h, j0 + rows)
        _displacement_rows(d[j0:j1], j0, h, w, ipd, (lo, hi), dx[j0:j1], dy[j0:j1])
    return dx, dy


def _displacement_rows(d: np.ndarray, j0: int, h: int, w: int, ipd: float, fade: tuple[float, float],
                       dx_out: np.ndarray, dy_out: np.ndarray) -> None:
    """Algorithm G forward displacement (left eye, pixels) for rows ``j0..j0+len(d)``."""
    j1 = j0 + d.shape[0]
    phi = (math.pi / 2 - (np.arange(j0, j1, dtype=np.float64) + 0.5) / h * math.pi)[:, None]
    r = 0.5 * ipd * _pole_weight(phi, fade)  # (R, 1)
    D = d.astype(np.float64)
    rho = D * np.cos(phi)
    yv = D * np.sin(phi)
    ratio = np.minimum(r / np.maximum(rho, _EPS), _ASIN_CAP)
    dlam = np.arcsin(ratio)
    phi2 = np.arctan2(yv, np.sqrt(np.maximum(rho * rho - r * r, _EPS)))
    has_r = np.broadcast_to(r > 0.0, D.shape)
    dx_out[:] = np.where(has_r, dlam * (w / (2.0 * math.pi)), 0.0)
    dy_out[:] = np.where(has_r, -(phi2 - phi) * (h / math.pi), 0.0)


def _accepts_kw(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _call_inpainter(inpainter: Any, img: np.ndarray, mask: np.ndarray, cancel: CancelToken | None) -> np.ndarray:
    """Run an Inpainter object (``.inpaint(img, mask)``) or a plain callable ``(img, mask)``."""
    if hasattr(inpainter, "inpaint"):
        fn: Callable[..., Any] = inpainter.inpaint
    elif callable(inpainter):
        fn = inpainter
    else:
        raise InputError("inpainter must provide inpaint(img, mask) or be a callable")
    out = fn(img, mask, cancel=cancel) if _accepts_kw(fn, "cancel") else fn(img, mask)
    out = np.asarray(out)
    if out.shape != img.shape:
        raise InferenceError(f"The inpainter returned an image of shape {out.shape}, expected {img.shape}")
    if out.dtype != np.uint8:
        out = np.clip(np.rint(out.astype(np.float64)), 0, 255).astype(np.uint8)
    return out


def _warp_eye(eq: np.ndarray, disp: np.ndarray, invd_map: np.ndarray, e: float, cancel: CancelToken | None,
              progress: ProgressReporter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-warp one eye.  ``disp`` HxWx2 float32 = left-eye (dx, dy); ``invd_map`` = 1/D.
    Returns (rgb uint8, alpha float32, inverse distance float32 at the chosen sources)."""
    h, w = disp.shape[:2]
    out = np.empty_like(eq)
    alpha = np.empty((h, w), np.float32)
    invd = np.empty((h, w), np.float32)
    rows = max(1, _TILE_PX // max(1, w))
    W = np.float32(w)
    half = np.float32(w / 2.0)
    ymax = np.float32(h - 1)
    ef = np.float32(e)
    xs = np.arange(w, dtype=np.float32)[None, :]

    def step_to(ux: np.ndarray, uy: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # keep x in [0, W): |delta| <= asin(.99)/2pi * W < W so one wrap suffices.  BORDER_WRAP
        # would wrap any x, but OpenCV 4.x saturates remap coordinates to int16, which breaks
        # x > 32767 (reachable for W > ~26600 without this reduction)
        nx = ux - ef * s[..., 0]
        nx[nx < 0.0] += W
        nx[nx >= W] -= W
        ny = uy - s[..., 1]
        np.clip(ny, 0.0, ymax, out=ny)
        return nx, ny

    def max_step(nx: np.ndarray, px: np.ndarray, ny: np.ndarray, py: np.ndarray) -> float:
        ax = np.abs(nx - px)
        ax = np.minimum(ax, W - ax)  # longitude distance across the seam
        return max(float(ax.max()), float(np.abs(ny - py).max()))

    def solve(ux: np.ndarray, uy: np.ndarray, px: np.ndarray, py: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # fixed point of p = u - delta(p)  (i.e. p <- p + (u - q(p)) with q(p) = p + delta(p))
        for it in range(_ITERS):
            nx, ny = step_to(ux, uy, cv2.remap(disp, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))
            done = it % 2 == 1 and max_step(nx, px, ny, py) < 1e-3
            px, py = nx, ny
            if done:
                break
        s = cv2.remap(disp, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        rx = ux - px - ef * s[..., 0] + half
        rx = np.remainder(rx, W) - half  # longitude residual modulo the width
        ry = uy - py - s[..., 1]
        inv = cv2.remap(invd_map, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return px, py, np.hypot(rx, ry), inv

    for j0 in range(0, h, rows):
        if cancel is not None:
            cancel.check()
        j1 = min(h, j0 + rows)
        ux = np.ascontiguousarray(np.broadcast_to(xs, (j1 - j0, w)))
        uy = np.ascontiguousarray(np.broadcast_to(np.arange(j0, j1, dtype=np.float32)[:, None], (j1 - j0, w)))
        pa = solve(ux, uy, ux, uy)
        bx, by = step_to(ux, uy, disp[j0:j1])  # second start: 2u - q(u) = u - delta(u)
        pb = solve(ux, uy, bx, by)
        ca, cb = pa[2] < _CONVERGED_PX, pb[2] < _CONVERGED_PX
        use_b = (cb & (~ca | (pb[3] > pa[3]))) | (~ca & ~cb & (pb[2] < pa[2]))
        px = np.where(use_b, pb[0], pa[0])
        py = np.where(use_b, pb[1], pa[1])
        res = np.where(use_b, pb[2], pa[2])
        invd[j0:j1] = np.where(use_b, pb[3], pa[3])
        alpha[j0:j1] = np.clip(1.5 - res, 0.0, 1.0)
        tile = cv2.remap(eq, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out[j0:j1] = tile.reshape(out[j0:j1].shape)
        progress(j1 / h, "Warping")
    return out, alpha, invd


def _nearest_in_rows(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: index of the nearest True pixel at/left of each pixel (-1 if none) and
    at/right of it (W if none)."""
    w = mask.shape[1]
    idx = np.arange(w, dtype=np.int32)[None, :]
    left = np.maximum.accumulate(np.where(mask, idx, np.int32(-1)), axis=1)
    right = np.minimum.accumulate(np.where(mask, idx, np.int32(w))[:, ::-1], axis=1)[:, ::-1]
    return left, right


def _bg_streak_rows(img: np.ndarray, hole: np.ndarray, invd: np.ndarray, reach: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Background-side propagation for a block of rows.

    Each hole pixel takes the colour of the nearest known pixel (one pixel further in, to
    skip mixed edge samples) on whichever side of its hole run is FARTHER away.  Returns
    (prefilled image, foreground mask) where the foreground mask marks known pixels within
    ``reach`` of a hole that are clearly nearer than that hole's background side."""
    h, w = hole.shape
    rows = np.arange(h)[:, None]
    known = ~hole
    kl, kr = _nearest_in_rows(known)
    kl_c, kr_c = np.clip(kl, 0, w - 1), np.clip(kr, 0, w - 1)
    inv_l = np.where(kl >= 0, invd[rows, kl_c], np.inf)
    inv_r = np.where(kr < w, invd[rows, kr_c], np.inf)
    use_left = inv_l <= inv_r
    bg_inv = np.minimum(inv_l, inv_r)
    src_l = np.where((kl_c >= 1) & known[rows, np.maximum(kl_c - 1, 0)], kl_c - 1, kl_c)
    src_r = np.where((kr_c <= w - 2) & known[rows, np.minimum(kr_c + 1, w - 1)], kr_c + 1, kr_c)
    src = np.where(use_left, src_l, src_r)
    out = img.copy()
    hr, hc = np.nonzero(hole & np.isfinite(bg_inv))
    out[hr, hc] = img[hr, src[hr, hc]]
    # foreground band next to each hole run
    hl, hrr = _nearest_in_rows(hole)
    hl_c, hr_c = np.clip(hl, 0, w - 1), np.clip(hrr, 0, w - 1)
    cols = np.arange(w)[None, :]
    near_l = (hl >= 0) & (cols - hl <= reach)
    near_r = (hrr < w) & (hrr - cols <= reach)
    thr = 1.19
    fg = known & ((near_l & (invd > thr * bg_inv[rows, hl_c])) | (near_r & (invd > thr * bg_inv[rows, hr_c])))
    return out, fg


def _push_pull_crops(img: np.ndarray, mask: np.ndarray, margin: int, cancel: CancelToken | None
                     ) -> np.ndarray:
    """``core.fill.push_pull_fill`` applied per group of nearby holes (crop + ``margin``):
    same result near the holes, but cost scales with the hole area, not the image."""
    out = img.copy()
    k = 2 * int(margin) + 1
    grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    for i in range(1, n):
        if cancel is not None and i % 64 == 0:
            cancel.check()
        x, y, cw, ch = (int(v) for v in stats[i, :4])
        sl = (slice(y, y + ch), slice(x, x + cw))
        sub_mask = mask[sl] & (labels[sl] == i)
        if not sub_mask.any():
            continue
        known = ~mask[sl]
        if not known.any():
            continue
        filled = push_pull_fill(img[sl], known)
        o = out[sl]
        o[sub_mask] = filled[sub_mask]
    return out


def _max_hole_run(hole: np.ndarray) -> int:
    """Longest horizontal run of hole pixels (row-chunked to bound memory)."""
    h, w = hole.shape
    best = 0
    rows = max(1, _TILE_PX // max(1, w))
    for j0 in range(0, h, rows):
        blk = hole[j0:j0 + rows]
        if not blk.any():
            continue
        kl, kr = _nearest_in_rows(~blk)
        best = max(best, int((kr - kl - 1)[blk].max()))
    return best


def _fill_disocclusions(img: np.ndarray, alpha: np.ndarray, invd: np.ndarray, inpainter: Any,
                        cancel: CancelToken | None) -> np.ndarray:
    """Fill pixels with alpha < 1 from the BACKGROUND side, with longitude wrap.

    ODS disocclusions are horizontal gaps between a near object and what is behind it.
    Known pixels near a gap that are clearly nearer than the
    gap's farther side are treated as foreground and hidden from the fill (masked for the
    inpainter / zero weight for push-pull), so the occluder is not smeared into the gap.
    Gap pixels are pre-filled by propagating the farther side's colour (what an inpainter
    sees under its mask, and the result if everything around is foreground)."""
    hole = alpha < 0.999
    if not hole.any():
        return img
    h, w = hole.shape
    gap = max(4, w // 1024) + _max_hole_run(hole) + 2
    # push-pull averages ever larger neighbourhoods, so it needs foreground hidden much
    # farther from the gap than an inpainter (which only needs the adjacent band); capped so
    # the morphology stays cheap on 8K+ panoramas
    reach = int(min(max(64, w // 64), gap if inpainter is not None else 4 * gap))
    pad = min(w, max(16, reach + gap))

    def wrap(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.concatenate([a[:, w - pad:], a, a[:, :pad]], axis=1))

    pre = np.empty_like(img)
    fg = np.zeros((h, w), bool)
    rows = max(1, (_TILE_PX // 2) // max(1, w + 2 * pad))
    for j0 in range(0, h, rows):
        if cancel is not None:
            cancel.check()
        j1 = min(h, j0 + rows)
        p_img, p_fg = _bg_streak_rows(wrap(img[j0:j1]), wrap(hole[j0:j1]), wrap(invd[j0:j1]), reach)
        pre[j0:j1] = p_img[:, pad:pad + w]
        fg[j0:j1] = p_fg[:, pad:pad + w]
    mask = hole | fg
    if mask.all():
        return pre
    if inpainter is None:
        filled = _push_pull_crops(wrap(pre), wrap(mask), reach + 8, cancel)[:, pad:pad + w]
    else:
        filled = _call_inpainter(inpainter, wrap(pre), wrap(mask), cancel)[:, pad:pad + w]
    a = np.clip(2.0 * alpha[hole] - 1.0, 0.0, 1.0)[:, None]  # alpha < 0.5 -> fully filled
    out = img.copy()
    out[hole] = np.clip(np.rint(img[hole] * a + filled[hole] * (1.0 - a)), 0, 255).astype(np.uint8)
    return out


def ods_pair(eq: RGB, dist_m: FloatMap, ipd_m: float = 0.064, *, inpainter: Any = None,
             pole_fade: tuple[float, float] = (45.0, 75.0), progress: ProgressReporter | None = None,
             cancel: CancelToken | None = None) -> tuple[RGB, RGB]:
    """Render an omnidirectional-stereo pair from a mono 360 equirect (algorithm G).

    ``eq``: full 360x180 equirect, HxWx3 uint8 RGB.  ``dist_m``: per-pixel distance along
    the ray in metres (resized to ``eq`` if needed; inf, NaN and <= 0 = infinitely far;
    distances nearer than ``ipd_m`` are clamped).  ``inpainter``: object with
    ``inpaint(img, mask_bool)`` (or a callable ``(img, mask_bool) -> img``) used for
    disocclusions, default push-pull.  ``pole_fade``: latitudes (deg) between which the
    parallax fades to zero toward the poles.  Returns ``(left, right)`` equirects of the
    input size; compose top/bottom (left on top) for Quest 360 stereo players.
    """
    pr = progress or ProgressReporter.null()
    if not isinstance(eq, np.ndarray) or eq.ndim != 3 or eq.shape[2] != 3 or eq.dtype != np.uint8:
        raise InputError("ODS input must be an HxWx3 uint8 RGB equirect")
    h, w = eq.shape[:2]
    if h < 4 or w < 8:
        raise InputError(f"ODS input is too small ({w}x{h})")
    if w >= 32767 or h >= 32767:
        raise InputError(f"ODS input is too large ({w}x{h}); the maximum is 32766 px per side")
    if abs(w / h - 2.0) > 0.1:
        log.warning("ODS input is %dx%d, not 2:1; treating it as a full 360x180 equirect", w, h)
    ipd = _check_ipd(ipd_m)
    fade = _check_pole_fade(pole_fade)
    if cancel is not None:
        cancel.check()
    eq = np.ascontiguousarray(eq)
    if ipd == 0.0:
        return eq.copy(), eq.copy()
    dist = _prepare_distance(dist_m, h, w, min_m=max(ipd, 1e-3))
    pr(0.0, "Computing stereo displacement")
    disp = np.empty((h, w, 2), np.float32)
    invd_map = np.empty((h, w), np.float32)
    rows = max(1, _TILE_PX // max(1, w))
    for j0 in range(0, h, rows):
        if cancel is not None:
            cancel.check()
        j1 = min(h, j0 + rows)
        _displacement_rows(dist[j0:j1], j0, h, w, ipd, fade, disp[j0:j1, :, 0], disp[j0:j1, :, 1])
        invd_map[j0:j1] = 1.0 / dist[j0:j1]
    del dist
    eyes: list[np.ndarray] = []
    for k, (e, name) in enumerate(((1.0, "left"), (-1.0, "right"))):
        base = 0.05 + 0.475 * k
        rgb, alpha, invd = _warp_eye(eq, disp, invd_map, e, cancel, pr.sub(base, base + 0.35))
        if cancel is not None:
            cancel.check()
        n_holes = int((alpha < 0.5).sum())
        pr(base + 0.35, f"Filling {name} eye disocclusions")
        log.debug("ODS %s eye: %d disoccluded px (%.3f%%)", name, n_holes, 100.0 * n_holes / alpha.size)
        rgb = _fill_disocclusions(rgb, alpha, invd, inpainter, cancel)
        eyes.append(rgb)
        pr(base + 0.475, f"{name.capitalize()} eye done")
    pr(1.0, "ODS done")
    return eyes[0], eyes[1]
