"""MoGe-2 port: focal/shift recovery on synthetic point maps + end-to-end on the stub."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vrphoto.core.types import ProgressReporter
from vrphoto.depth.moge import (
    MoGeEstimator,
    focal_to_fov,
    focal_to_intrinsics,
    fov_to_focal,
    normalized_view_plane_uv,
    points_to_depth,
    recover_focal_shift,
    token_grid,
)


def synth_points(h=96, w=128, fov=60.0, shift=1.5, scale=1.0):
    """Affine-invariant point map of a bumpy surface: true z minus ``shift``, times ``scale``."""
    uv = normalized_view_plane_uv(w, h).astype(np.float64)
    focal = fov_to_focal(fov, w, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    z = 3.0 + 1.2 * np.sin(xx / 13.0) * np.cos(yy / 11.0) + 2.0 * yy / h
    xy = uv * z[..., None] / focal
    pts = np.concatenate([xy, z[..., None]], -1)
    pts[..., 2] -= shift
    return (pts * scale).astype(np.float32), focal, z


def test_uv_convention():
    uv = normalized_view_plane_uv(4, 2)
    diag = math.hypot(4, 2)
    # pixel i -> (i + 0.5 - W/2) / (diag/2)
    np.testing.assert_allclose(uv[0, :, 0], [(i + 0.5 - 2) / (diag / 2) for i in range(4)], atol=1e-6)
    np.testing.assert_allclose(uv[:, 0, 1], [(j + 0.5 - 1) / (diag / 2) for j in range(2)], atol=1e-6)


def test_focal_fov_roundtrip():
    for fov in (30.0, 60.0, 90.0, 120.0):
        f = fov_to_focal(fov, 640, 480)
        assert focal_to_fov(f, 640, 480) == pytest.approx(fov)
        K = focal_to_intrinsics(f, 640, 480)
        assert K.hfov_deg == pytest.approx(fov)
        assert K.fx == K.fy and K.cx == 320 and K.cy == 240


@pytest.mark.parametrize("shift", [0.0, 1.5, -2.0])
@pytest.mark.parametrize("fov", [45.0, 75.0])
def test_recover_focal_shift_synthetic(shift, fov):
    pts, focal, _ = synth_points(fov=fov, shift=shift)
    f, s = recover_focal_shift(pts)
    assert f == pytest.approx(focal, rel=0.02)
    assert s == pytest.approx(shift, abs=0.05)


def test_recover_is_scale_invariant():
    pts, focal, _ = synth_points(shift=1.0, scale=0.25)
    f, s = recover_focal_shift(pts)
    assert f == pytest.approx(focal, rel=0.02)
    assert s == pytest.approx(0.25, abs=0.02)


def test_recover_with_mask_ignores_garbage():
    pts, focal, _ = synth_points(shift=1.2)
    mask = np.ones(pts.shape[:2], bool)
    mask[:30] = False
    pts[:30] = np.array([5.0, -7.0, 0.01], np.float32)  # sky / garbage
    pts[40, 40] = np.nan
    f, s = recover_focal_shift(pts, mask)
    assert f == pytest.approx(focal, rel=0.03)
    assert s == pytest.approx(1.2, abs=0.08)
    # batch dim + float mask accepted by points_to_depth
    z, valid, f2, s2 = points_to_depth(pts[None], mask.astype(np.float32)[None])
    assert valid[:30].sum() == 0 and not valid[40, 40]
    assert s2 == pytest.approx(s, abs=1e-4)


def test_recover_with_known_focal():
    pts, focal, _ = synth_points(shift=0.8)
    f, s = recover_focal_shift(pts, focal=focal)
    assert f == focal
    assert s == pytest.approx(0.8, abs=0.03)


def test_recover_degenerate():
    pts = np.zeros((10, 10, 3), np.float32)
    assert recover_focal_shift(pts, np.zeros((10, 10), bool)) == (1.0, 0.0)
    with pytest.raises(ValueError):
        recover_focal_shift(np.zeros((10, 10, 2), np.float32))


def test_fronto_parallel_plane_keeps_zero_shift():
    h, w = 64, 64
    uv = normalized_view_plane_uv(w, h)
    focal = fov_to_focal(60.0, w, h)
    pts = np.concatenate([uv * 2.0 / focal, np.full((h, w, 1), 2.0, np.float32)], -1)
    f, s = recover_focal_shift(pts)
    assert s == pytest.approx(0.0, abs=1e-3)  # flat cost: the solver must not wander
    assert focal_to_fov(f, w, h) == pytest.approx(60.0, abs=0.5)


def test_points_to_depth_metric_and_fallback():
    pts, focal, z_true = synth_points(shift=1.0)
    z, valid, f, s = points_to_depth(pts, None, metric_scale=2.0)
    assert valid.all()
    np.testing.assert_allclose(z, z_true * 2.0, rtol=0.03)
    # known FOV
    z2, _, f2, _ = points_to_depth(pts, None, fov_x=60.0)
    assert f2 == pytest.approx(focal)
    # implausible geometry -> 60 degree fallback
    weird = pts.copy()
    weird[..., :2] *= 1e-4  # tiny xy -> focal huge (FOV ~0)
    _, _, f3, _ = points_to_depth(weird)
    assert focal_to_fov(f3, 128, 96) == pytest.approx(60.0, abs=0.01)


def test_token_grid():
    rows, cols = token_grid(480, 640, 1800)
    assert rows * cols == pytest.approx(1800, rel=0.05)
    assert cols / rows == pytest.approx(640 / 480, rel=0.05)
    assert token_grid(100, 100, 1800) == (42, 42)


def test_stub_end_to_end_plane_at_2m(use_stubs):
    """Stub: plane at z = 2 m seen by a 60 degree camera, metric_scale 1."""
    img = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    ev = []
    r = MoGeEstimator(device="cpu").estimate(img, progress=ProgressReporter(lambda f, m: ev.append(f)))
    assert r.model_id == "moge2-vitl"
    assert r.disparity.shape == (240, 320) and r.disparity.dtype == np.float32
    assert r.intrinsics is not None
    assert r.intrinsics.width == 320 and r.intrinsics.height == 240
    assert r.intrinsics.hfov_deg == pytest.approx(60.0, abs=1.0)
    assert r.intrinsics.vfov_deg == pytest.approx(60.0, abs=1.0)  # stub squashes to 256x256
    assert r.metric_m is not None and r.metric_m.shape == (240, 320)
    np.testing.assert_allclose(r.metric_m, 2.0, rtol=0.02)
    assert np.allclose(r.disparity, r.disparity.mean(), atol=1e-3)  # flat scene
    assert ev[-1] == 1.0


def test_feeds_and_net_size(use_stubs):
    est = MoGeEstimator(device="cpu", num_tokens=2000)
    model = est._load(ProgressReporter.null(), None)
    assert est.net_size(480, 640, model) == (256, 256)  # static stub
    feeds = est._feeds(model, np.zeros((1, 3, 256, 256), np.float32))
    assert feeds["num_tokens"].dtype == np.int64 and feeds["num_tokens"].shape == ()
    assert int(feeds["num_tokens"]) == 2000


class FakeMoGe:
    """Dynamic-shape fake returning a synthetic tilted scene with a sky mask."""

    def __init__(self):
        self.provider = "CPUExecutionProvider"
        self.seen = {}

    def input_names(self):
        return ["image", "num_tokens"]

    def input_shapes(self):
        return [["batch", 3, "h", "w"], []]

    def run(self, feeds, output_names=None, **kw):
        x = feeds["image"]
        self.seen = {k: (v.shape, v.dtype) for k, v in feeds.items()}
        _, _, h, w = x.shape
        pts, _, _ = synth_points(h, w, fov=70.0, shift=0.5)
        mask = np.ones((1, h, w), np.float32)
        mask[0, : h // 5] = 0.0
        return {"points": pts[None], "mask": mask, "metric_scale": np.array([3.0], np.float32)}


def test_dynamic_model_metric_depth_and_sky(monkeypatch):
    est = MoGeEstimator(device="cpu")
    fake = FakeMoGe()
    est._model = fake
    img = np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    r = est.estimate(img)
    shape, dtype = fake.seen["image"]
    rows, cols = token_grid(300, 400, 1800)
    assert shape == (1, 3, rows * 14, cols * 14) and dtype == np.float32
    assert r.intrinsics.hfov_deg == pytest.approx(70.0, abs=1.5)
    # sky (masked) is the farthest region; metric depth is finite and positive
    assert r.metric_m is not None and np.isfinite(r.metric_m).all() and r.metric_m.min() > 0
    assert r.metric_m[5].mean() > r.metric_m[-5].mean()
    assert r.disparity[5].mean() < r.disparity[-5].mean()


def test_feeds_with_token_input_first():
    class M:
        def input_names(self):
            return ["num_tokens", "pixel"]

        def input_shapes(self):
            return [[1], [1, 3, 224, 280]]

    est = MoGeEstimator(num_tokens=1500)
    assert est.net_size(480, 640, M()) == (224, 280)
    feeds = est._feeds(M(), np.zeros((1, 3, 224, 280), np.float32))
    assert feeds["pixel"].shape == (1, 3, 224, 280)
    assert feeds["num_tokens"].shape == (1,) and int(feeds["num_tokens"][0]) == 1500
