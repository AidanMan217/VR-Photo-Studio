"""Heuristic (no model) depth estimator."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, DepthResult, ProgressReporter
from vrphoto.depth.heuristic import HeuristicEstimator, spectral_residual_saliency


def test_output_range_shape_dtype(photo_rgb):
    r = HeuristicEstimator().estimate(photo_rgb)
    assert isinstance(r, DepthResult)
    assert r.model_id == "heuristic"
    assert r.disparity.shape == photo_rgb.shape[:2]
    assert r.disparity.dtype == np.float32
    assert r.disparity.min() >= 0.0 and r.disparity.max() <= 1.0
    assert r.disparity.max() - r.disparity.min() > 0.5
    assert r.metric_m is None and r.intrinsics is None
    assert r.raw is not None and r.raw.shape == photo_rgb.shape[:2]


def test_bottom_is_nearer_on_uniform_image():
    img = np.full((120, 160, 3), 128, np.uint8)
    d = HeuristicEstimator().estimate(img).disparity
    assert d[-5:].mean() > d[:5].mean() + 0.5
    rows = d.mean(axis=1)
    assert np.all(np.diff(rows) >= -1e-3)  # monotone ramp


def test_salient_subject_pops_out():
    rng = np.random.default_rng(0)
    img = (rng.normal(120, 4, (200, 300, 3))).clip(0, 255).astype(np.uint8)
    img[80:120, 130:170] = (250, 40, 40)
    d = HeuristicEstimator().estimate(img).disparity
    # subject is nearer than background at the same height
    assert d[100, 150] > d[100, 40] + 0.05


@pytest.mark.parametrize(
    "img",
    [
        np.zeros((50, 70), np.uint8),
        np.zeros((50, 70, 1), np.uint8),
        np.zeros((50, 70, 4), np.uint8),
        np.zeros((50, 70, 3), np.float32),
        np.zeros((3000, 40, 3), np.uint8),
        np.zeros((2, 2, 3), np.uint8),
    ],
    ids=["gray", "gray1", "rgba", "float", "tall", "tiny"],
)
def test_accepts_various_inputs(img):
    d = HeuristicEstimator().estimate(img).disparity
    assert d.shape == img.shape[:2]
    assert np.isfinite(d).all() and d.min() >= 0 and d.max() <= 1


def test_rejects_garbage():
    with pytest.raises(InputError):
        HeuristicEstimator().estimate(np.zeros((10,), np.uint8))
    with pytest.raises(InputError):
        HeuristicEstimator().estimate(np.zeros((10, 10, 5), np.uint8))
    with pytest.raises(InputError):
        HeuristicEstimator().estimate("not an image")  # type: ignore[arg-type]


def test_progress_and_cancel(photo_rgb):
    ev = []
    HeuristicEstimator().estimate(photo_rgb, progress=ProgressReporter(lambda f, m: ev.append(f)))
    assert ev[0] == 0.0 and ev[-1] == 1.0
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        HeuristicEstimator().estimate(photo_rgb, cancel=tok)


def test_spectral_residual_saliency():
    img = np.full((64, 64, 3), 100, np.uint8)
    assert not spectral_residual_saliency(img).any()
    img[20:30, 40:50] = 255
    s = spectral_residual_saliency(img)
    assert s.shape == (64, 64) and s.min() >= 0 and s.max() <= 1
    assert s[25, 45] > s[50, 10]
    wide = spectral_residual_saliency(np.zeros((100, 400, 3), np.uint8))
    assert wide.shape == (16, 64)


def test_large_image_is_fast():
    import time

    img = np.random.default_rng(1).integers(0, 255, (2000, 3000, 3), dtype=np.uint8)
    t = time.perf_counter()
    d = HeuristicEstimator().estimate(img).disparity
    assert d.shape == (2000, 3000)
    assert time.perf_counter() - t < 8.0
