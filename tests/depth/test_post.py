"""Depth post-processing: normalize, guided filter, refine, temporal alignment, edges."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.depth.post import (
    depth_edges,
    finalize_disparity,
    gray_guide,
    guided_filter,
    normalize,
    refine,
    temporal_align,
    upsample,
)


def test_normalize_range_and_percentiles():
    rng = np.random.default_rng(0)
    raw = rng.normal(10, 3, (100, 120)).astype(np.float32)
    raw[0, 0] = 1e6  # outlier must not squash everything
    d = normalize(raw)
    assert d.dtype == np.float32 and d.shape == raw.shape
    assert d.min() == 0.0 and d.max() == 1.0
    assert 0.4 < float(np.median(d)) < 0.6
    assert (d == 1.0).mean() == pytest.approx(0.02, abs=0.01)
    # orientation preserved
    assert normalize(np.array([[1.0, 2.0, 3.0]] * 3, np.float32))[0, 2] == 1.0


def test_normalize_edge_cases():
    assert np.all(normalize(np.full((5, 5), 3.0, np.float32)) == 0.5)
    a = np.ones((4, 4), np.float32)
    a[0, 0] = np.nan
    a[1, 1] = np.inf
    a[2, 2] = 5.0
    d = normalize(a, 0, 100)
    assert np.isfinite(d).all() and d[0, 0] == 0.0 and d[2, 2] == 1.0
    assert np.all(normalize(np.full((3, 3), np.nan, np.float32)) == 0.0)
    # (1, H, W) inputs are squeezed
    assert normalize(np.arange(12, dtype=np.float32).reshape(1, 3, 4)).shape == (3, 4)
    with pytest.raises(ValueError):
        normalize(np.zeros((2, 3, 4), np.float32))
    # mostly constant map with a few distinct values falls back to min/max
    m = np.zeros((50, 50), np.float32)
    m[0, :3] = 1.0
    assert normalize(m).max() == 1.0


def _step_guide(h=64, w=64):
    g = np.zeros((h, w), np.float32)
    g[:, w // 2 :] = 1.0
    return g


def test_guided_filter_gray_preserves_edges():
    rng = np.random.default_rng(1)
    guide = _step_guide()
    src = guide + rng.normal(0, 0.1, guide.shape).astype(np.float32)
    out = guided_filter(src, guide, r=4, eps=1e-3)
    assert out.dtype == np.float32 and out.shape == src.shape
    # edge stays sharp: columns adjacent to the step keep their side's value
    assert out[:, 31].mean() < 0.15 and out[:, 32].mean() > 0.85
    # noise is reduced
    assert np.std(out[:, :28]) < 0.5 * np.std(src[:, :28])
    # a box blur (no guide) would smear the edge
    blur = guided_filter(src, np.zeros_like(guide), r=4, eps=1e-3)
    assert blur[:, 31].mean() > 0.3


def test_guided_filter_color_guide_preserves_colour_edges():
    rng = np.random.default_rng(2)
    h, w = 48, 64
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, : w // 2] = (200, 30, 30)
    rgb[:, w // 2 :] = (30, 30, 200)  # same luma-ish, different hue
    target = np.zeros((h, w), np.float32)
    target[:, w // 2 :] = 1.0
    src = target + rng.normal(0, 0.1, target.shape).astype(np.float32)
    out = guided_filter(src, rgb, r=5, eps=1e-3)
    assert out[:, w // 2 - 1].mean() < 0.15 and out[:, w // 2].mean() > 0.85
    # float guide in [0, 1] and RGBA guide give the same result as uint8
    out_f = guided_filter(src, rgb.astype(np.float32) / 255.0, r=5, eps=1e-3)
    np.testing.assert_allclose(out, out_f, atol=1e-4)
    rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
    np.testing.assert_allclose(guided_filter(src, rgba, r=5, eps=1e-3), out, atol=1e-5)


def test_guided_filter_constant_and_errors():
    src = np.full((20, 30), 0.3, np.float32)
    np.testing.assert_allclose(guided_filter(src, _step_guide(20, 30), r=3), 0.3, atol=1e-5)
    with pytest.raises(ValueError):
        guided_filter(src, np.zeros((10, 10), np.float32))
    with pytest.raises(ValueError):
        guided_filter(np.zeros((2, 3, 4), np.float32), np.zeros((2, 3), np.float32))
    # huge radius is clamped, tiny image ok
    assert guided_filter(np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32), r=50).shape == (3, 3)


def test_refine(photo_rgb):
    rng = np.random.default_rng(3)
    h, w = photo_rgb.shape[:2]
    d = np.full((h, w), 0.3, np.float32)
    d[120:220, 120:200] = 0.7  # the subject in photo_rgb
    noisy = (d + rng.normal(0, 0.08, d.shape)).astype(np.float32)
    out = refine(noisy, photo_rgb, 1.0)
    assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1
    rmse = lambda a: float(np.sqrt(np.mean((a - d) ** 2)))  # noqa: E731
    assert rmse(out) < 0.6 * rmse(noisy)
    # the subject boundary stays sharp
    assert out[170, 118] < 0.4 and out[170, 122] > 0.6
    np.testing.assert_array_equal(refine(noisy, photo_rgb, 0.0), noisy)
    half = refine(noisy, photo_rgb, 0.5)
    assert np.abs(half - noisy).mean() < np.abs(out - noisy).mean()
    # image of a different size is resized
    assert refine(noisy[::2, ::2].copy(), photo_rgb, 1.0).shape == (h // 2, w // 2)


def test_temporal_align_recovers_scale_shift():
    rng = np.random.default_rng(4)
    prev = rng.random((60, 80)).astype(np.float32) * 0.8 + 0.1
    cur = (prev - 0.1) / 2.0  # same scene, model drifted in scale + offset
    out = temporal_align(prev, cur, ema=0.0, range_pull=0.0)
    np.testing.assert_allclose(out, prev, atol=1e-3)
    # by default the range only moves a little toward the current frame's (no jump)
    soft = temporal_align(prev, cur, ema=0.0)
    assert np.abs(soft - prev).max() < 0.2 * np.abs(cur - prev).max()
    assert temporal_align(None, cur) is not cur
    np.testing.assert_array_equal(temporal_align(None, cur), cur)


def test_temporal_align_ema_and_motion():
    prev = np.full((40, 40), 0.5, np.float32)
    prev[:, :20] = 0.2
    cur = prev.copy()
    cur[10:20, 25:35] = 0.9  # an object moved in
    out = temporal_align(prev, cur, ema=0.6)
    # static pixels unchanged, moving object follows the current frame (no trail)
    np.testing.assert_allclose(out[30:, :], prev[30:, :], atol=0.02)
    assert out[15, 30] > 0.85
    # small flicker is smoothed by the EMA
    flick = prev + 0.02
    flick[:, :20] -= 0.04
    sm = temporal_align(prev, flick, ema=0.6)
    assert np.abs(sm - prev).mean() < np.abs(flick - prev).mean()


def test_temporal_align_shape_mismatch_and_degenerate():
    prev = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    out = temporal_align(prev, np.zeros((20, 20), np.float32))
    assert out.shape == (20, 20) and np.isfinite(out).all()
    inverted = temporal_align(prev, 1.0 - prev, ema=0.0)  # inverted fit is rejected
    assert inverted.shape == prev.shape and np.isfinite(inverted).all()
    np.testing.assert_array_equal(inverted, 1.0 - prev)  # treated as a new shot


def test_depth_edges():
    d = np.zeros((10, 10), np.float32)
    d[:, 5:] = 1.0
    e = depth_edges(d, 0.5)
    assert e.dtype == bool
    assert e[:, 4].all() and e[:, 5].all()
    assert e.sum() == 20
    assert not depth_edges(d, 1.5).any()
    v = np.zeros((10, 10), np.float32)
    v[5:, :] = 2.0
    assert depth_edges(v, 1.0)[4:6].all()


def test_gray_guide_and_upsample(photo_rgb):
    g = gray_guide(photo_rgb)
    assert g.shape == photo_rgb.shape[:2] and 0 <= g.min() and g.max() <= 1
    small = np.random.default_rng(5).random((12, 16)).astype(np.float32)
    up = upsample(small, (48, 64))
    assert up.shape == (48, 64) and up.dtype == np.float32
    assert upsample(up, (12, 16)).shape == (12, 16)


def test_finalize_disparity(photo_rgb):
    h, w = photo_rgb.shape[:2]
    rel = np.zeros((37, 49), np.float32)
    rel[18:, 20:30] = 5.0
    d, raw = finalize_disparity(rel, photo_rgb)
    assert d.shape == (h, w) and raw.shape == (h, w)
    assert d.dtype == np.float32 and d.min() >= 0 and d.max() <= 1
    assert d[200, 160] > 0.8 and d[20, 20] < 0.2
