"""BiRefNet subject segmentation (ONNX).

Input ``input_image`` float32 ``[1,3,1024,1024]``: RGB/255 normalised with the ImageNet
mean/std.  Output ``output_image`` ``[1,1,1024,1024]`` LOGITS -> sigmoid -> bilinear
resize back to the photo size.

Model ids: ``birefnet-lite`` (general, default) and ``birefnet-portrait`` (people).
"""

from __future__ import annotations

from typing import ClassVar, Literal

from ._onnxseg import OnnxMattingSegmenter


class BiRefNetSegmenter(OnnxMattingSegmenter):
    """BiRefNet matting.  ``BiRefNetSegmenter("birefnet-portrait")`` for people."""

    default_model_id: ClassVar[str] = "birefnet-lite"
    names: ClassVar[dict[str, str]] = {
        "birefnet-lite": "birefnet-lite",
        "birefnet-portrait": "birefnet-portrait",
    }
    default_size: ClassVar[int] = 1024
    mean: ClassVar[tuple[float, float, float]] = (0.485, 0.456, 0.406)
    std: ClassVar[tuple[float, float, float]] = (0.229, 0.224, 0.225)
    activation: ClassVar[Literal["sigmoid", "minmax"]] = "sigmoid"
    preferred_outputs: ClassVar[tuple[str, ...]] = ("output_image",)


__all__ = ["BiRefNetSegmenter"]
