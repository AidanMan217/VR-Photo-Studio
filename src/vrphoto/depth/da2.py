"""Depth Anything V2 (small / base / large) through ONNX Runtime.

Pre-processing follows the reference implementation (``Resize(518, keep_aspect_ratio,
ensure_multiple_of=14, resize_method="lower_bound")`` + ImageNet normalisation): the
SHORT side becomes 518 and both sides are multiples of 14 (patch size).  Very wide
images are capped at :data:`MAX_LONG` on the long side.

DirectML note: the DML execution provider re-plans the graph whenever the input shape
changes (first run per shape costs seconds), so on DML the long side is snapped to a
small fixed set of :data:`LONG_BUCKETS`.  We deliberately keep ONE session per model
rather than one per shape: a session per shape would duplicate the weights in VRAM
(1.3 GB for the large model).  Videos keep a constant shape anyway.

Output ``predicted_depth`` is relative inverse depth (disparity: larger = nearer); it is
post-processed with algorithm A (normalise, bicubic upsample, guided filter).
"""

from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np

from ..core.types import RGB, CancelToken, DepthResult, ProgressReporter
from .base import as_rgb, check, reporter
from .post import finalize_disparity

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 14
SHORT_SIDE = 518
MAX_LONG = 1792  # 128 patches
LONG_BUCKETS = (518, 616, 700, 784, 924, 1036, 1176, 1372, 1568, 1792)


def round_multiple(x: float, m: int = PATCH) -> int:
    """Nearest multiple of ``m`` (at least ``m``)."""
    return max(m, int(round(x / m)) * m)


def input_size(
    h: int,
    w: int,
    *,
    short: int = SHORT_SIDE,
    multiple: int = PATCH,
    max_long: int = MAX_LONG,
    buckets: tuple[int, ...] | None = None,
) -> tuple[int, int]:
    """Network input ``(H, W)`` for an ``h`` x ``w`` image.

    Short side -> ``short`` keeping aspect (long side capped at ``max_long``), both sides
    rounded to multiples of ``multiple``; with ``buckets`` the long side snaps to the
    nearest bucket value (slight aspect change, fewer distinct shapes).
    """
    if h <= 0 or w <= 0:
        raise ValueError("image size must be positive")
    s = short / min(h, w)
    th, tw = h * s, w * s
    if max(th, tw) > max_long:
        c = max_long / max(th, tw)
        th, tw = th * c, tw * c
    th_i, tw_i = round_multiple(th, multiple), round_multiple(tw, multiple)
    if buckets:
        if th_i >= tw_i:
            th_i = min(buckets, key=lambda b: (abs(b - th_i), b))
            th_i = max(th_i, tw_i)
        else:
            tw_i = min(buckets, key=lambda b: (abs(b - tw_i), b))
            tw_i = max(tw_i, th_i)
    return th_i, tw_i


def preprocess(rgb: RGB, size_hw: tuple[int, int]) -> np.ndarray:
    """Resize + ImageNet-normalise to a float32 ``[1, 3, H, W]`` tensor."""
    th, tw = size_hw
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if (th < h and tw < w) else cv2.INTER_CUBIC
    x = cv2.resize(rgb, (tw, th), interpolation=interp) if (th, tw) != (h, w) else rgb
    x = x.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def squeeze_map(a: np.ndarray) -> np.ndarray:
    """Reduce a model output like ``[1,H,W]`` / ``[1,1,H,W]`` to ``[H,W]``."""
    a = np.asarray(a)
    while a.ndim > 2:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"unexpected depth output shape {a.shape}")
    return a.astype(np.float32)


def static_hw(shape: list[Any]) -> tuple[int, int] | None:
    """``(H, W)`` if the last two dims of an input shape are fixed ints, else None."""
    if len(shape) >= 2 and all(isinstance(v, int) and v > 0 for v in shape[-2:]):
        return int(shape[-2]), int(shape[-1])
    return None


class DA2Estimator:
    """Depth Anything V2 estimator (``da2-small`` / ``da2-base`` / ``da2-large``)."""

    def __init__(
        self,
        model_id: str = "da2-small",
        device: str | int = "auto",
        *,
        max_long: int = MAX_LONG,
        buckets: bool | None = None,
    ) -> None:
        self.id = model_id
        self.device = device
        self.max_long = max_long
        self.buckets = buckets  # None = automatic (on DirectML only)
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self, progress: ProgressReporter, cancel: CancelToken | None) -> Any:
        with self._lock:
            # reload when the cached session was closed (model deleted / re-downloaded or
            # sessions unloaded in Settings); load_model re-uses its own session cache
            if self._model is None or getattr(self._model, "closed", False) is True:
                from ..models.ort import load_model

                self._model = load_model(self.id, device=self.device, progress=progress, cancel=cancel)
            return self._model

    def _use_buckets(self, model: Any) -> bool:
        if self.buckets is not None:
            return bool(self.buckets)
        if os.environ.get("VRPHOTO_DA2_BUCKETS", "").strip() in ("1", "true", "yes"):
            return True
        return getattr(model, "provider", "") == "DmlExecutionProvider"

    def net_size(self, h: int, w: int, model: Any) -> tuple[int, int]:
        """Network input size for an image of ``h`` x ``w`` with ``model``."""
        fixed = static_hw(model.input_shapes()[0])
        if fixed is not None:
            return fixed
        return input_size(
            h, w, max_long=self.max_long, buckets=LONG_BUCKETS if self._use_buckets(model) else None
        )

    def estimate(
        self, img: RGB, *, progress: ProgressReporter | None = None, cancel: CancelToken | None = None
    ) -> DepthResult:
        """Relative disparity in [0, 1] (1 = near) at the input resolution."""
        rep = reporter(progress)
        rgb = as_rgb(img)
        h, w = rgb.shape[:2]
        model = self._load(rep.sub(0.0, 0.5), cancel)
        check(cancel)
        rep(0.5, "Estimating depth")
        th, tw = self.net_size(h, w, model)
        x = preprocess(rgb, (th, tw))
        name = model.input_names()[0]
        outs = model.run({name: x}, cancel=cancel)
        pred = outs.get("predicted_depth")
        if pred is None:
            pred = next(iter(outs.values()))
        rel = squeeze_map(pred)
        check(cancel)
        rep(0.85, "Refining depth")
        disp, raw = finalize_disparity(rel, rgb)
        rep(1.0, "Depth ready")
        return DepthResult(disparity=disp, model_id=self.id, raw=raw)


__all__ = ["DA2Estimator", "input_size", "preprocess", "LONG_BUCKETS", "IMAGENET_MEAN", "IMAGENET_STD"]
