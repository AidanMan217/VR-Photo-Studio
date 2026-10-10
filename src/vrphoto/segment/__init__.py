"""Subject segmentation (alpha mattes): BiRefNet / IS-Net / U2-Net (ONNX) and GrabCut.

Public API (see docs/CONTRACTS.md section 4)::

    from vrphoto.segment import get_segmenter, clean_matte
    alpha = get_segmenter("auto").segment(rgb)          # float32 HxW, 1 = subject
    alpha = clean_matte(alpha, feather_px=2)
"""

from .base import AutoSegmenter, Segmenter, get_segmenter, list_segmenters
from .matte import clean_matte, refine_matte

__all__ = ["AutoSegmenter", "Segmenter", "get_segmenter", "list_segmenters", "clean_matte", "refine_matte"]
