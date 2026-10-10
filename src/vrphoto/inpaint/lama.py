"""LaMa inpainting (Carve/LaMa-ONNX ``lama_fp32.onnx``).

Inputs: ``image`` float32 ``[1,3,512,512]`` RGB/255 and ``mask`` float32 ``[1,1,512,512]``
with **1 = fill, 0 = keep** (binarised).  Output ``[1,3,512,512]`` already in 0..255
(other LaMa exports return 0..1; the range is detected from the known pixels unless
``output_range`` is set).  The registry marks LaMa ``dml_ok=False`` (DirectML crash),
so it runs on the CPU there.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from ..core.types import RGB
from ..segment import _runtime as rt
from ._onnxinpaint import OnnxInpainter, to_hwc

__all__ = ["LamaInpainter", "output_scale"]

OutputRange = Literal["auto", "0-255", "0-1"]


def output_scale(out: np.ndarray, crop: RGB, hole: np.ndarray) -> float:
    """Factor that maps a float model output to 0..255: 1.0 if it already is 0..255,
    255.0 if it is 0..1.  Decided by which interpretation reproduces the known pixels."""
    known = ~hole
    if known.any():
        o = out[known].astype(np.float32)
        c = crop[known].astype(np.float32)
        err_255 = float(np.mean(np.abs(o - c)))
        err_1 = float(np.mean(np.abs(o * 255.0 - c)))
        return 1.0 if err_255 <= err_1 else 255.0
    return 1.0 if float(np.max(out)) > 1.5 else 255.0


class LamaInpainter(OnnxInpainter):
    """Large-mask inpainting with LaMa (best quality; ~200 MB download)."""

    model_id: ClassVar[str] = "lama"
    default_size: ClassVar[int] = 512
    preferred_outputs: ClassVar[tuple[str, ...]] = ("output",)

    def __init__(
        self,
        device: str | int = "auto",
        *,
        model: rt.SessionLike | None = None,
        model_path: str | Path | None = None,
        size: int | None = None,
        margin: float = 1.0,
        feather: float = 4,
        output_range: OutputRange = "auto",
    ) -> None:
        super().__init__(
            device, model=model, model_path=model_path, size=size, margin=margin, feather=feather
        )
        self.output_range: OutputRange = output_range

    def encode(self, model: rt.SessionLike, crop: RGB, hole: np.ndarray) -> dict[str, np.ndarray]:
        """LaMa convention: float image 0..1 (holes zeroed), float mask 1 = fill."""
        img_name, mask_name = rt.image_mask_inputs(model)
        m = hole.astype(np.float32)
        x = crop.astype(np.float32) / 255.0 * (1.0 - m)[..., None]
        return {
            img_name: np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32),
            mask_name: np.ascontiguousarray(m[None, None], dtype=np.float32),
        }

    def decode(self, out: np.ndarray, crop: RGB, hole: np.ndarray) -> RGB:
        a = to_hwc(out, crop.shape[0])
        if a.dtype == np.uint8:
            return a
        a = np.nan_to_num(a.astype(np.float32))
        if self.output_range == "0-255":
            scale = 1.0
        elif self.output_range == "0-1":
            scale = 255.0
        else:
            scale = output_scale(a, crop, hole)
        return np.clip(np.rint(a * scale), 0, 255).astype(np.uint8)
