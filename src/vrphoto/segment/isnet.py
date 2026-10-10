"""ISNet (DIS general use) and U^2-Net salient object segmentation (ONNX).

* ``isnet-general``: 1024x1024, RGB/255 with mean 0.5 / std 1.0, output[0] min-max
  normalised.
* ``u2net``: 320x320, RGB/255 ImageNet-normalised, output[0] (the fused ``d0`` map)
  min-max normalised.

Unlike rembg we do NOT divide the input by the image maximum (that brightens dark photos
and changes the model's input statistics).
"""

from __future__ import annotations

from typing import ClassVar, Literal

from ._onnxseg import OnnxMattingSegmenter


class ISNetSegmenter(OnnxMattingSegmenter):
    """IS-Net general-use dichotomous image segmentation (``get_segmenter("isnet")``)."""

    default_model_id: ClassVar[str] = "isnet-general"
    names: ClassVar[dict[str, str]] = {"isnet-general": "isnet"}
    default_size: ClassVar[int] = 1024
    mean: ClassVar[tuple[float, float, float]] = (0.5, 0.5, 0.5)
    std: ClassVar[tuple[float, float, float]] = (1.0, 1.0, 1.0)
    activation: ClassVar[Literal["sigmoid", "minmax"]] = "minmax"
    preferred_outputs: ClassVar[tuple[str, ...]] = ()


class U2NetSegmenter(OnnxMattingSegmenter):
    """U^2-Net salient object detection (``get_segmenter("u2net")``)."""

    default_model_id: ClassVar[str] = "u2net"
    names: ClassVar[dict[str, str]] = {"u2net": "u2net"}
    default_size: ClassVar[int] = 320
    mean: ClassVar[tuple[float, float, float]] = (0.485, 0.456, 0.406)
    std: ClassVar[tuple[float, float, float]] = (0.229, 0.224, 0.225)
    activation: ClassVar[Literal["sigmoid", "minmax"]] = "minmax"
    preferred_outputs: ClassVar[tuple[str, ...]] = ()


__all__ = ["ISNetSegmenter", "U2NetSegmenter"]
