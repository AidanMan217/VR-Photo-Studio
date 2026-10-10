"""Shared implementation of the fixed-size ONNX inpainting models (LaMa, MI-GAN).

Subclasses implement :meth:`OnnxInpainter.encode` (crop + bool mask -> feeds, the place
where each model's mask convention lives) and :meth:`OnnxInpainter.decode` (output ->
``HxWx3 uint8``).  Holes are processed with :func:`inpaint_by_crops`.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from ..core.errors import InferenceError
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, ProgressReporter
from ..segment import _runtime as rt
from .crops import inpaint_by_crops

log = get_logger(__name__)


def to_hwc(out: np.ndarray, size: int) -> np.ndarray:
    """``[1,3,H,W]`` / ``[3,H,W]`` / ``[1,H,W,3]`` / ``[H,W,3]`` -> ``HxWx3`` (dtype kept)."""
    a = np.asarray(out)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[2] not in (3, 4):
        a = a.transpose(1, 2, 0)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise InferenceError(f"Unexpected inpainting output shape {np.shape(out)}.")
    a = a[..., :3]
    if a.shape[:2] != (size, size):
        raise InferenceError(f"Inpainting output is {a.shape[1]}x{a.shape[0]}, expected {size}x{size}.")
    return a


class OnnxInpainter:
    """Base class for model-backed inpainters."""

    model_id: ClassVar[str] = ""
    default_size: ClassVar[int] = 512
    preferred_outputs: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        device: str | int = "auto",
        *,
        model: rt.SessionLike | None = None,
        model_path: str | Path | None = None,
        size: int | None = None,
        margin: float = 1.0,
        feather: float = 4,
    ) -> None:
        self.id = self.model_id
        self.size = size
        self.margin = margin
        self.feather = feather
        self._lazy = rt.LazyModel(self.model_id, device=device, model=model, model_path=model_path)

    def load(self, *, progress: ProgressReporter | None = None, cancel: CancelToken | None = None) -> None:
        """Load (and download on first use) the model now."""
        self._lazy.get(progress=progress, cancel=cancel)

    # ------------------------------------------------------------------ subclass hooks
    def encode(
        self, model: rt.SessionLike, crop: RGB, hole: np.ndarray
    ) -> dict[str, np.ndarray]:  # pragma: no cover - abstract
        raise NotImplementedError

    def decode(self, out: np.ndarray, crop: RGB, hole: np.ndarray) -> RGB:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ public API
    def model_size(self, model: rt.SessionLike) -> int:
        """Square crop size fed to the model (static model input size if declared)."""
        if self.size is not None:
            return int(self.size)
        img_name, _ = rt.image_mask_inputs(model)
        idx = list(model.input_names()).index(img_name)
        default = rt.spec_input_size(self.model_id) or (self.default_size, self.default_size)
        hh, ww = rt.static_hw(model, idx, default)
        if hh != ww:
            raise InferenceError(f"Unsupported non-square input size {ww}x{hh} for {self.model_id}.")
        return int(hh)

    def infer(
        self, model: rt.SessionLike, crop: RGB, hole: np.ndarray, cancel: CancelToken | None = None
    ) -> RGB:
        """Run the model once on a ``size x size`` crop (no cropping/blending)."""
        outs = rt.run_model(model, self.encode(model, crop, hole), model_id=self.model_id, cancel=cancel)
        res = self.decode(rt.pick_output(outs, self.preferred_outputs), crop, hole)
        return np.ascontiguousarray(res)

    def inpaint(
        self,
        img: RGB,
        mask: np.ndarray,
        *,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> RGB:
        """Fill ``mask`` (bool HxW, True = fill).  Unmasked pixels are returned unchanged.

        Downloads the model on first use (raises :class:`ModelUnavailable` if impossible)."""
        rgb = rt.as_rgb(img)
        m = rt.as_bool_mask(mask, rgb.shape[:2])
        if not m.any():
            return rgb.copy()
        rt.report(progress, 0.0, "Loading inpainting model")
        model = self._lazy.get(progress=rt.sub_progress(progress, 0.0, 0.3), cancel=cancel)
        size = self.model_size(model)

        def fn(crop: RGB, hole: np.ndarray) -> RGB:
            return self.infer(model, crop, hole, cancel)

        return inpaint_by_crops(
            rgb,
            m,
            fn,
            size=size,
            margin=self.margin,
            feather=self.feather,
            progress=rt.sub_progress(progress, 0.3, 1.0),
            cancel=cancel,
        )
