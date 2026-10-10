"""MI-GAN inpainting (official ONNX pipeline ``migan_pipeline_v2.onnx``).

From Picsart-AI-Research/MI-GAN ``scripts/create_onnx_pipeline.py``: inputs ``image``
uint8 ``[1,3,H,W]`` RGB and ``mask`` uint8 ``[1,1,H,W]`` where **255 = KNOWN and 0 = HOLE**
(the opposite of LaMa); output ``result`` uint8 ``[1,3,H,W]``.  The pipeline crops around
the holes, resizes to 512, runs the generator and blends back itself; we still feed it
512x512 crops (``inpaint_by_crops``) so distant holes are processed separately at full
detail.  Input names are resolved from the session metadata (channel count, then name).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..core.types import RGB
from ..segment import _runtime as rt
from ._onnxinpaint import OnnxInpainter, to_hwc
from .lama import output_scale

__all__ = ["MiganInpainter", "KNOWN", "HOLE"]

KNOWN = 255
HOLE = 0


class MiganInpainter(OnnxInpainter):
    """Fast mobile inpainting with MI-GAN (~30 MB download)."""

    model_id: ClassVar[str] = "migan"
    default_size: ClassVar[int] = 512
    preferred_outputs: ClassVar[tuple[str, ...]] = ("result", "output")

    def encode(self, model: rt.SessionLike, crop: RGB, hole: np.ndarray) -> dict[str, np.ndarray]:
        """MI-GAN convention: uint8 image, uint8 mask with 255 = known, 0 = hole."""
        img_name, mask_name = rt.image_mask_inputs(model)
        mk = np.where(hole, HOLE, KNOWN).astype(np.uint8)
        return {
            img_name: np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.uint8),
            mask_name: np.ascontiguousarray(mk[None, None], dtype=np.uint8),
        }

    def decode(self, out: np.ndarray, crop: RGB, hole: np.ndarray) -> RGB:
        a = to_hwc(out, crop.shape[0])
        if a.dtype == np.uint8:
            return a
        a = np.nan_to_num(a.astype(np.float32))
        return np.clip(np.rint(a * output_scale(a, crop, hole)), 0, 255).astype(np.uint8)
