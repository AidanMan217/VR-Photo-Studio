"""Model-free depth heuristic (instant, offline fallback).

Disparity = ``0.55 * vertical ramp`` (bottom of the frame is near, as for most
ground-level photos) ``+ 0.45 * blurred spectral-residual saliency`` (salient subjects
pop forward), normalised and refined with a guided filter so the depth follows image
edges.  Crude, but always available and produces pleasant parallax for portraits and
landscapes.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.types import RGB, CancelToken, DepthResult, FloatMap, ProgressReporter
from .base import as_rgb, check, reporter
from .post import auto_subsample, guided_filter, normalize, upsample

RAMP_WEIGHT = 0.55
SALIENCY_WEIGHT = 0.45


def spectral_residual_saliency(img: RGB, size: int = 64) -> FloatMap:
    """Spectral-residual saliency (Hou & Zhang 2007) at ``size`` px long side, in [0,1].

    Returns zeros for (near-)uniform images.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    s = size / max(h, w)
    sw, sh = max(4, int(round(w * s))), max(4, int(round(h * s)))
    g = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    if float(g.std()) < 1e-3:
        return np.zeros((sh, sw), np.float32)
    f = np.fft.fft2(g.astype(np.float64))
    amp = np.abs(f)
    log_amp = np.log(amp + 1e-4 * float(amp.max()))  # relative floor: exact spectral zeros
    phase = np.angle(f)
    residual = log_amp - cv2.blur(log_amp, (3, 3), borderType=cv2.BORDER_REPLICATE)
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), sigmaX=max(1.0, 0.04 * max(sw, sh)))
    lo, hi = float(sal.min()), float(sal.max())
    if hi - lo < 1e-12:
        return np.zeros_like(sal, dtype=np.float32)
    return ((sal - lo) / (hi - lo)).astype(np.float32)


class HeuristicEstimator:
    """No-model estimator: vertical ramp + saliency, guided-filter refined."""

    id = "heuristic"

    def __init__(self, work_long: int = 384) -> None:
        self.work_long = int(max(32, work_long))

    def estimate(
        self, img: RGB, *, progress: ProgressReporter | None = None, cancel: CancelToken | None = None
    ) -> DepthResult:
        """Estimate a relative disparity map (1 = near) at the input resolution."""
        rep = reporter(progress)
        rgb = as_rgb(img)
        h, w = rgb.shape[:2]
        rep(0.0, "Estimating depth (heuristic)")
        check(cancel)
        s = min(1.0, self.work_long / max(h, w))
        sw, sh = max(2, int(round(w * s))), max(2, int(round(h * s)))
        small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA) if s < 1.0 else rgb

        ramp = np.repeat(np.linspace(0.0, 1.0, sh, dtype=np.float32)[:, None], sw, axis=1)
        sal = spectral_residual_saliency(small)
        sal = cv2.resize(sal, (sw, sh), interpolation=cv2.INTER_CUBIC)
        sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=max(1.0, 0.03 * max(sw, sh)))
        sal = np.clip(sal, 0.0, None)
        if sal.max() > 0:
            sal = sal / sal.max()
        raw = RAMP_WEIGHT * ramp + SALIENCY_WEIGHT * sal
        check(cancel)
        rep(0.5, "Refining depth")
        d = normalize(raw, 0.5, 99.5)
        # edge-aware refinement at working resolution (colour guide), then upsample + grey guide
        d = guided_filter(d, small, r=max(2, int(round(0.03 * max(sw, sh)))), eps=1e-2)
        d = upsample(np.clip(d, 0.0, 1.0), (h, w))
        r = max(1, int(round(0.006 * w)))
        d = guided_filter(
            d, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), r=r, eps=1e-3, subsample=auto_subsample(h * w, r)
        )
        d = np.clip(d, 0.0, 1.0).astype(np.float32)
        check(cancel)
        rep(1.0, "Depth ready")
        return DepthResult(disparity=d, model_id=self.id, raw=upsample(raw.astype(np.float32), (h, w)))


__all__ = ["HeuristicEstimator", "spectral_residual_saliency"]
