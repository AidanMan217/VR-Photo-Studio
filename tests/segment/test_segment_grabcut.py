"""GrabCut segmenter and saliency helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, ProgressReporter
from vrphoto.segment.grabcut import GrabCutSegmenter, rect_from_saliency, saliency_map


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


@pytest.fixture
def red_gt(photo_rgb) -> np.ndarray:
    gt = np.zeros(photo_rgb.shape[:2], bool)
    gt[120:220, 120:200] = True
    return gt


def test_finds_red_subject_without_rect(photo_rgb, red_gt):
    alpha = GrabCutSegmenter().segment(photo_rgb)
    assert alpha.dtype == np.float32 and alpha.shape == photo_rgb.shape[:2]
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert _iou(alpha > 0.5, red_gt) > 0.9


def test_finds_red_subject_with_rect(photo_rgb, red_gt):
    alpha = GrabCutSegmenter().segment(photo_rgb, rect=(100, 100, 120, 135))
    assert _iou(alpha > 0.5, red_gt) > 0.9
    assert alpha[:100].max() == 0.0  # outside the rectangle is background


def test_large_image_is_processed_downscaled():
    rng = np.random.default_rng(0)
    h, w = 900, 1200
    bg = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), 6)
    bg = (bg.astype(np.float32) * 0.5 + np.array([60, 120, 170]) * 0.5).astype(np.uint8)
    sub = np.zeros((h, w), np.uint8)
    cv2.ellipse(sub, (600, 520), (220, 330), 0, 0, 360, 255, -1)
    img = bg.copy()
    img[sub > 0] = (205, 150, 90)
    alpha = GrabCutSegmenter(max_side=400).segment(img)
    assert alpha.shape == (h, w)
    assert _iou(alpha > 0.5, sub > 0) > 0.9


def test_saliency_highlights_subject(photo_rgb, red_gt):
    sal = saliency_map(photo_rgb)
    assert sal.shape == red_gt.shape and sal.dtype == np.float32
    assert 0.0 <= sal.min() and sal.max() <= 1.0 + 1e-6
    assert sal[red_gt].mean() > 5 * sal[~red_gt].mean()
    x, y, w, h = rect_from_saliency(sal)
    assert x <= 120 and y <= 120 and x + w >= 200 and y + h >= 220


def test_rect_from_flat_saliency_falls_back_to_centre():
    x, y, w, h = rect_from_saliency(np.zeros((100, 200), np.float32))
    assert 0 < x < 50 and 0 < y < 20 and w > 100 and y + h == 100


def test_uniform_image_does_not_crash():
    alpha = GrabCutSegmenter().segment(np.full((60, 80, 3), 128, np.uint8))
    assert alpha.shape == (60, 80)
    assert np.isfinite(alpha).all()


def test_progress_and_cancel(photo_rgb):
    seen: list[float] = []
    GrabCutSegmenter(iterations=3).segment(photo_rgb, progress=ProgressReporter(lambda f, m: seen.append(f)))
    assert seen[-1] == pytest.approx(1.0)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        GrabCutSegmenter().segment(photo_rgb, cancel=tok)


def test_invalid_rect(photo_rgb):
    with pytest.raises(InputError):
        GrabCutSegmenter().segment(photo_rgb, rect=(1000, 1000, 5, 5))
    with pytest.raises(InputError):
        GrabCutSegmenter().segment(photo_rgb, rect=("a", 0, 1, 1))  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 2, 3), (3, 500, 3), (6, 6, 3)])
def test_degenerate_tiny_images_do_not_raise(shape):
    img = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    alpha = GrabCutSegmenter().segment(img)
    assert alpha.shape == shape[:2] and alpha.dtype == np.float32
    assert np.isfinite(alpha).all() and alpha.min() >= 0.0 and alpha.max() <= 1.0
