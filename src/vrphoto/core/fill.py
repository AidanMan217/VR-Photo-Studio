"""Push-pull hole filling: fast, deterministic, smooth fill of unknown pixels.

Used for hole filling after warping, VR180/360 surroundings, billboard back textures and
as the no-model inpainting fallback.
"""

from __future__ import annotations

import cv2
import numpy as np


def push_pull_fill(img: np.ndarray, known: np.ndarray, *, min_size: int = 1) -> np.ndarray:
    """Fill pixels where ``known`` is False/0 from their surroundings.

    ``img``: HxW or HxWxC (uint8, uint16 or float).  ``known``: HxW bool or float weights in
    [0,1].  Builds a pyramid of (img*w, w) by area downsampling, then on the way back up
    blends each level with the upsampled coarser estimate: ``est = num + up * (1 - w)``
    (premultiplied form of ``img*w + up*(1-w)``).  Pixels with weight 1 are returned
    unchanged.  Returns the same dtype and shape as ``img``.

    Only the bounding box of the unknown pixels (plus a generous margin of known context)
    is processed, with per-channel OpenCV ops, so small holes in big images are cheap.
    """
    if img.ndim not in (2, 3):
        raise ValueError("img must be HxW or HxWxC")
    if known.shape[:2] != img.shape[:2]:
        raise ValueError("known mask shape must match image")
    if known.dtype == bool:
        w = known
        unknown = ~known
    else:
        w = np.clip(known.astype(np.float32), 0.0, 1.0)
        unknown = w < 1.0
    if not unknown.any():
        return img.copy()
    if not (w if w.dtype == bool else w > 0).any():
        return np.zeros_like(img)

    h, wd = unknown.shape
    rows = np.flatnonzero(unknown.any(axis=1))
    cols = np.flatnonzero(unknown.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    margin = max(32, ((y1 - y0) + (x1 - x0)) // 4)
    y0, y1 = max(0, y0 - margin), min(h, y1 + margin)
    x0, x1 = max(0, x0 - margin), min(wd, x1 + margin)

    out = img.copy()
    crop_img = img[y0:y1, x0:x1]
    crop_w = w[y0:y1, x0:x1]
    if crop_w.dtype == bool and not crop_w.any():  # crop has no context: use whole image
        y0, y1, x0, x1 = 0, h, 0, wd
        crop_img, crop_w = img, w
    filled = _push_pull_core(crop_img, crop_w, min_size)
    if w.dtype == bool:
        region = unknown[y0:y1, x0:x1]
        out[y0:y1, x0:x1][region] = filled[region]
    else:
        out[y0:y1, x0:x1] = filled
        full = w >= 1.0
        out[full] = img[full]
    return out


def _push_pull_core(img: np.ndarray, known: np.ndarray, min_size: int) -> np.ndarray:
    dtype = img.dtype
    w = known.astype(np.float32)
    if img.ndim == 2:
        chans = [img]
    else:
        chans = [img[..., c] for c in range(img.shape[2])]

    dens: list[np.ndarray] = [w]
    nums: list[list[np.ndarray]] = [[cv2.multiply(np.ascontiguousarray(c), w, dtype=cv2.CV_32F) for c in chans]]
    while True:
        h, wd = dens[-1].shape
        if max(h, wd) <= min_size or min(h, wd) <= 1:
            break
        size = (max(1, (wd + 1) // 2), max(1, (h + 1) // 2))
        dens.append(cv2.resize(dens[-1], size, interpolation=cv2.INTER_AREA))
        nums.append([cv2.resize(n, size, interpolation=cv2.INTER_AREA) for n in nums[-1]])

    d = np.maximum(dens[-1], 1e-8)
    est = [cv2.divide(n, d) for n in nums[-1]]
    for lvl in range(len(dens) - 2, -1, -1):
        den = dens[lvl]
        h, wd = den.shape
        inv = 1.0 - den
        est = [
            cv2.add(n, cv2.multiply(cv2.resize(e, (wd, h), interpolation=cv2.INTER_LINEAR), inv))
            for n, e in zip(nums[lvl], est)
        ]

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        est = [np.clip(np.rint(e), info.min, info.max).astype(dtype) for e in est]
    else:
        est = [e.astype(dtype, copy=False) for e in est]
    if img.ndim == 2:
        return est[0]
    return np.stack(est, axis=-1)
