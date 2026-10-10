"""Helpers for the pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def decode_frames(path: Path) -> list[np.ndarray]:
    from vrphoto.video.decode import iter_frames

    return list(iter_frames(path))
