"""Real-model smoke tests (download weights).  Run with VRPHOTO_REAL_MODELS=1.

Weights are cached in ``$VRPHOTO_REAL_MODELS_HOME`` (default ``~/.cache/vrphoto-test-models``)
instead of the per-test isolated home so they download only once.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = [
    pytest.mark.models,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("VRPHOTO_REAL_MODELS") != "1", reason="set VRPHOTO_REAL_MODELS=1"),
]


@pytest.fixture
def real_home(monkeypatch):
    home = Path(os.environ.get("VRPHOTO_REAL_MODELS_HOME", Path.home() / ".cache" / "vrphoto-test-models"))
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VRPHOTO_HOME", str(home))
    from vrphoto.core import config

    config.set_settings(None)
    yield home
    config.set_settings(None)


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """A plain wall with a large, saturated, textured ball in front of it."""
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([180 + 0 * xx, 175 + yy * 20 // h, 165 + xx * 20 // w], -1).astype(np.uint8)
    gt = np.zeros((h, w), np.uint8)
    cv2.circle(gt, (320, 260), 140, 1, -1)
    shade = np.clip(255 - np.hypot(xx - 280, yy - 220) * 1.2, 60, 255)
    ball = np.stack([shade, shade * 0.25, shade * 0.15], -1).astype(np.uint8)
    img[gt > 0] = ball[gt > 0]
    return img, gt.astype(bool)


@pytest.mark.parametrize("name", ["birefnet-lite", "isnet", "u2net"])
def test_real_segmenter_finds_ball(real_home, name):
    from vrphoto.segment import get_segmenter

    img, gt = _scene()
    alpha = get_segmenter(name, device="cpu").segment(img)
    assert alpha.shape == gt.shape and alpha.dtype == np.float32
    pred = alpha > 0.5
    iou = (pred & gt).sum() / max(1, (pred | gt).sum())
    assert iou > 0.8, iou
