"""Model-free inpainting: OpenCV Telea / Navier-Stokes and push-pull.

* ``telea`` (id ``"opencv"``): fast-marching inpainting; good for thin bands and small
  holes, smears large holes.
* ``ns`` (id ``"opencv-ns"``): Navier-Stokes based; similar quality, slower.
* ``pushpull`` (id ``"pushpull"``): smooth pyramid fill (``core.fill.push_pull_fill``);
  instant, blurry, never streaky.  Best for large holes when no model is available.

Telea / NS run time grows with hole area x radius^2 and cannot be cancelled while OpenCV
runs (a 2 MP hole takes ~10 s, 0.3 MP about 1 s), so holes larger than ``max_hole_px``
are filled on a downscaled copy (known pixels averaged without hole content), upscaled
and blended into the full-resolution surroundings.  At that size the fill is a smooth
smear either way.
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from ..core.errors import InferenceError, InputError
from ..core.fill import push_pull_fill
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, ProgressReporter
from ..segment import _runtime as rt
from .crops import masked_downscale, seam_blend

log = get_logger(__name__)

__all__ = ["OpenCVInpainter"]

Method = Literal["telea", "ns", "pushpull"]
_IDS: dict[str, str] = {"telea": "opencv", "ns": "opencv-ns", "pushpull": "pushpull"}

#: Holes with more pixels than this are filled at reduced resolution by Telea / NS.
MAX_FULLRES_HOLE_PX = 300_000


class OpenCVInpainter:
    """Inpainting without AI models (see module docstring)."""

    def __init__(
        self, method: Method = "telea", radius: int | None = None, *, max_hole_px: int = MAX_FULLRES_HOLE_PX
    ) -> None:
        if method not in _IDS:
            raise InputError(
                f"Unknown OpenCV inpainting method '{method}'.", hint="Use telea, ns or pushpull."
            )
        self.method: Method = method
        self.radius = radius
        self.max_hole_px = max(1024, int(max_hole_px))
        self.id = _IDS[method]

    def _radius(self, h: int, w: int) -> int:
        if self.radius is not None:
            return max(1, int(self.radius))
        return int(np.clip(round(max(h, w) / 400.0), 3, 9))

    def inpaint(
        self,
        img: RGB,
        mask: np.ndarray,
        *,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> RGB:
        """Fill ``mask`` (bool HxW, True = fill).  Unmasked pixels are returned unchanged."""
        rgb = rt.as_rgb(img)
        h, w = rgb.shape[:2]
        m = rt.as_bool_mask(mask, (h, w))
        if not m.any():
            return rgb.copy()
        if cancel is not None:
            cancel.check()
        rt.report(progress, 0.0, "Filling holes")
        if m.all():
            log.warning("Inpainting mask covers the whole image; filling with grey")
            out = np.full_like(rgb, 128)
        elif self.method == "pushpull":
            out = push_pull_fill(rgb, ~m)
        else:
            area = int(np.count_nonzero(m))
            if area > self.max_hole_px:
                out = self._fill_downscaled(rgb, m, area, cancel)
            else:
                out = self._cv_inpaint(rgb, m, self._radius(h, w))
        if cancel is not None:
            cancel.check()
        res = rgb.copy()
        res[m] = out[m]
        rt.report(progress, 1.0, "Holes filled")
        return res

    def _cv_inpaint(self, rgb: RGB, m: np.ndarray, radius: int) -> RGB:
        flags = cv2.INPAINT_TELEA if self.method == "telea" else cv2.INPAINT_NS
        try:
            return cv2.inpaint(np.ascontiguousarray(rgb), m.astype(np.uint8) * 255, radius, flags)
        except cv2.error as e:  # pragma: no cover - OpenCV rejects only malformed input
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
            raise InferenceError(f"OpenCV inpainting failed: {msg}") from e

    def _fill_downscaled(self, rgb: RGB, m: np.ndarray, area: int, cancel: CancelToken | None) -> RGB:
        """Telea / NS on a copy shrunk so the hole has about ``max_hole_px`` pixels."""
        h, w = m.shape
        s = math.sqrt(self.max_hole_px / float(area))
        sw, sh = max(2, int(round(w * s))), max(2, int(round(h * s)))
        log.debug("Large hole (%d px): OpenCV %s at %dx%d", area, self.method, sw, sh)
        small, small_hole = masked_downscale(rgb, m, sw, sh)
        if small_hole.all():
            return np.full_like(rgb, 128)
        filled = self._cv_inpaint(small, small_hole, self._radius(sh, sw))
        if cancel is not None:
            cancel.check()
        big = rt.resize_to(filled, w, h)
        return seam_blend(rgb, big, m, max(2.0, 1.0 / s))
