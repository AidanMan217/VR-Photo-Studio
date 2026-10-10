"""Inpainting ("look behind"): LaMa / MI-GAN (ONNX), OpenCV Telea / NS and push-pull.

Public API (see docs/CONTRACTS.md section 4)::

    from vrphoto.inpaint import get_inpainter, inpaint_by_crops
    out = get_inpainter("auto").inpaint(rgb, mask)      # mask: bool HxW, True = fill
"""

from .base import AutoInpainter, Inpainter, get_inpainter, list_inpainters
from .crops import inpaint_by_crops

__all__ = ["AutoInpainter", "Inpainter", "get_inpainter", "list_inpainters", "inpaint_by_crops"]
