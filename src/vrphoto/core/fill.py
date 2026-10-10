"""Push-pull hole filling: fast, deterministic, smooth fill of unknown pixels.

Used for hole filling after warping, VR180/360 surroundings, billboard back textures and
as the no-model inpainting fallback.
"""

from __future__ import annotations

import cv2
import numpy as np


def push_pull_fill(img: np.ndarray, known: np.ndarray, *, min_size: int = 1) -> np.ndarray:
    """Fill pixels where ``known`` is False/0 from their surroundings.

    ``img``: HxW or HxWxC (uint8 or float).  ``known``: HxW bool or float weights in [0,1].
    Builds a pyramid of (img*w, w) by area downsampling, then on the way back up replaces
    unknown pixels with the upsampled weighted average.  Known pixels are returned unchanged.
    Returns the same dtype as ``img``.
    """
    if img.ndim not in (2, 3):
        raise ValueError("img must be HxW or HxWxC")
    dtype = img.dtype
    x = img.astype(np.float32)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[..., None]
    w = known.astype(np.float32)
    if w.shape != x.shape[:2]:
        raise ValueError("known mask shape must match image")
    if w.max() <= 0:
        out = np.zeros_like(x)
        return out[..., 0].astype(dtype) if squeeze else out.astype(dtype)

    # push: build pyramid
    levels: list[tuple[np.ndarray, np.ndarray]] = []
    num = x * w[..., None]
    den = w.copy()
    while True:
        levels.append((num, den))
        h, wd = den.shape
        if max(h, wd) <= min_size or min(h, wd) <= 1:
            break
        nh, nw = max(1, (h + 1) // 2), max(1, (wd + 1) // 2)
        num = cv2.resize(num, (nw, nh), interpolation=cv2.INTER_AREA)
        if num.ndim == 2:
            num = num[..., None]
        den = cv2.resize(den, (nw, nh), interpolation=cv2.INTER_AREA)

    # pull: from coarsest to finest
    num_c, den_c = levels[-1]
    est = num_c / np.maximum(den_c[..., None], 1e-8)
    for num_l, den_l in reversed(levels[:-1]):
        h, wd = den_l.shape
        up = cv2.resize(est, (wd, h), interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2:
            up = up[..., None]
        cur = num_l / np.maximum(den_l[..., None], 1e-8)
        a = np.clip(den_l, 0.0, 1.0)[..., None]
        est = cur * a + up * (1.0 - a)

    out = np.where(w[..., None] >= 0.999, x, est)
    if squeeze:
        out = out[..., 0]
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(dtype)
