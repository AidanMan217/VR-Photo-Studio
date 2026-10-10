"""Shared implementation of the single-image ONNX matting models (BiRefNet, ISNet, U2Net).

All of them take one ``[1,3,S,S]`` float tensor (RGB/255, per-channel normalised) and
return a ``[1,1,S,S]`` map that is turned into alpha either with a sigmoid (BiRefNet
logits) or with min-max normalisation (ISNet / U2Net).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import cv2
import numpy as np

from ..core.errors import InferenceError, InputError
from ..core.log import get_logger
from ..core.types import CancelToken, FloatMap, ProgressReporter
from . import _runtime as rt

log = get_logger(__name__)

Rect = tuple[int, int, int, int]

#: Context added around a user rectangle before running the model (fraction of its size).
RECT_CONTEXT = 0.15


def validate_rect(rect: Rect | None, w: int, h: int) -> Rect | None:
    """Clamp an ``(x, y, w, h)`` rectangle to the image; ``None`` stays ``None``.

    Raises :class:`InputError` if the rectangle does not overlap the image."""
    if rect is None:
        return None
    try:
        x, y, rw, rh = (int(round(float(v))) for v in rect)
    except (TypeError, ValueError) as e:
        raise InputError(f"Invalid selection rectangle {rect!r}; expected (x, y, width, height).") from e
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + rw), min(h, y + rh)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise InputError(
            "The selection rectangle is outside the image or too small.",
            hint="Drag a box around the subject.",
        )
    return x0, y0, x1 - x0, y1 - y0


def expand_rect(rect: Rect, frac: float, w: int, h: int) -> Rect:
    """Grow ``rect`` by ``frac`` of its size on every side, clamped to the image."""
    x, y, rw, rh = rect
    dx, dy = int(round(rw * frac)), int(round(rh * frac))
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(w, x + rw + dx), min(h, y + rh + dy)
    return x0, y0, x1 - x0, y1 - y0


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (float32)."""
    x = np.clip(x.astype(np.float32), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; a constant map becomes zeros."""
    x = x.astype(np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


class OnnxMattingSegmenter:
    """Base class: subclass and set the class attributes describing the model."""

    #: default model id (registry id) for the subclass
    default_model_id: ClassVar[str] = ""
    #: get_segmenter name per model id (``id`` attribute of instances)
    names: ClassVar[dict[str, str]] = {}
    default_size: ClassVar[int] = 1024
    mean: ClassVar[tuple[float, float, float]] = (0.485, 0.456, 0.406)
    std: ClassVar[tuple[float, float, float]] = (0.229, 0.224, 0.225)
    activation: ClassVar[Literal["sigmoid", "minmax"]] = "sigmoid"
    preferred_outputs: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        model_id: str | None = None,
        device: str | int = "auto",
        *,
        model: rt.SessionLike | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.id = self.names.get(self.model_id, self.model_id)
        self._lazy = rt.LazyModel(self.model_id, device=device, model=model, model_path=model_path)

    # ------------------------------------------------------------------ public API
    def load(self, *, progress: ProgressReporter | None = None, cancel: CancelToken | None = None) -> None:
        """Load (and download on first use) the model now instead of on first use."""
        self._lazy.get(progress=progress, cancel=cancel)

    def segment(
        self,
        img: np.ndarray,
        *,
        rect: Rect | None = None,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> FloatMap:
        """Foreground alpha ``float32 HxW`` in [0, 1] (1 = subject).

        ``rect`` = ``(x, y, width, height)`` in pixels: the subject lies inside it.  The
        model then runs on the rectangle (plus some context) at full model resolution
        and everything outside the rectangle is background."""
        rgb = rt.as_rgb(img)
        h, w = rgb.shape[:2]
        r = validate_rect(rect, w, h)
        rt.report(progress, 0.0, "Loading segmentation model")
        model = self._lazy.get(progress=rt.sub_progress(progress, 0.0, 0.4), cancel=cancel)
        rt.report(progress, 0.4, "Segmenting subject")
        if r is None:
            alpha = self._predict(model, rgb, cancel)
        else:
            cx, cy, cw, ch = expand_rect(r, RECT_CONTEXT, w, h)
            sub = self._predict(model, rgb[cy : cy + ch, cx : cx + cw], cancel)
            alpha = np.zeros((h, w), np.float32)
            x, y, rw, rh = r
            alpha[y : y + rh, x : x + rw] = sub[y - cy : y - cy + rh, x - cx : x - cx + rw]
        rt.report(progress, 1.0, "Segmentation done")
        return alpha

    # ------------------------------------------------------------------ internals
    def _input_hw(self, model: rt.SessionLike) -> tuple[int, int]:
        default = rt.spec_input_size(self.model_id) or (self.default_size, self.default_size)
        return rt.static_hw(model, 0, default)

    def preprocess(self, rgb: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        """RGB uint8 -> normalised ``[1,3,H,W]`` float32 tensor."""
        sh, sw = size_hw
        x = rt.resize_to(rgb, sw, sh, smooth=True).astype(np.float32) / 255.0
        x = (x - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)

    def postprocess(self, out: np.ndarray, w: int, h: int) -> FloatMap:
        """Model output -> alpha at ``(h, w)``."""
        a = np.asarray(out, dtype=np.float32)
        while a.ndim > 2:
            a = a[0]
        if a.ndim != 2:
            raise InferenceError(f"Unexpected segmentation output shape {np.shape(out)}.")
        if self.activation == "sigmoid":
            a = sigmoid(np.nan_to_num(a, nan=-60.0, posinf=60.0, neginf=-60.0))
        else:
            finite = a[np.isfinite(a)]
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
            a = minmax(np.nan_to_num(a, nan=lo, posinf=hi, neginf=lo))
        if a.shape != (h, w):
            shrink = w <= a.shape[1] and h <= a.shape[0]
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
        return np.clip(a, 0.0, 1.0).astype(np.float32)

    def _predict(self, model: rt.SessionLike, rgb: np.ndarray, cancel: CancelToken | None) -> FloatMap:
        h, w = rgb.shape[:2]
        x = self.preprocess(rgb, self._input_hw(model))
        name = model.input_names()[0]
        outs = rt.run_model(model, {name: x}, model_id=self.model_id, cancel=cancel)
        return self.postprocess(rt.pick_output(outs, self.preferred_outputs), w, h)
