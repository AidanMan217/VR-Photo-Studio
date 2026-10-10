"""temporal_align over long frame chains: the depth range must not fade (review F1)."""

from __future__ import annotations

import cv2
import numpy as np

from vrphoto.depth.post import scene_cut, temporal_align

H, W = 90, 160


def _scene_a() -> np.ndarray:
    yy = np.mgrid[0:H, 0:W][0]
    d = (yy / H).astype(np.float32)  # ground plane
    d[30:65, 60:110] = 0.9  # a near box
    return d


def _scene_b() -> np.ndarray:
    xx = np.mgrid[0:H, 0:W][1]
    d = (1.0 - xx / W).astype(np.float32) * 0.6
    d[15:40, 15:50] = 0.95
    return d


def _normalised(x: np.ndarray) -> np.ndarray:
    """What every estimator returns: a [0, 1] map stretched to its 2..98 % range."""
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def _noise(rng: np.random.Generator, sigma: float, smooth: bool) -> np.ndarray:
    if not smooth:
        return rng.normal(0, sigma, (H, W)).astype(np.float32)
    n = rng.normal(0, 1, (H // 10 + 2, W // 10 + 2)).astype(np.float32)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    return n / n.std() * sigma


def _range(x: np.ndarray) -> float:
    lo, hi = np.percentile(x, [2, 98])
    return float(hi - lo)


def _chain(frames, ema: float = 0.6) -> list[np.ndarray]:
    prev, outs = None, []
    for cur in frames:
        prev = temporal_align(prev, cur, ema=ema)
        outs.append(prev)
    return outs


def test_long_static_noisy_sequence_keeps_its_depth_range():
    # 900 frames = 30 s at 30 fps; the old OLS chain lost ~45-80% of the range here
    for sigma, smooth in ((0.01, False), (0.02, False), (0.02, True)):
        rng = np.random.default_rng(1)
        base = _scene_a()
        frames = (_normalised(base + _noise(rng, sigma, smooth)) for _ in range(900))
        outs = _chain(frames)
        for i in (299, 899):
            assert _range(outs[i]) > 0.9, (sigma, smooth, i, _range(outs[i]))
        # the near box keeps its depth difference to the background
        o = outs[-1]
        assert float(o[45:50, 80:90].mean() - o[45:50, 10:20].mean()) > 0.3


def test_global_scale_jitter_is_smoothed_without_shrinking():
    rng = np.random.default_rng(2)
    base = _scene_a()
    frames = []
    for _ in range(600):
        x = base + _noise(rng, 0.01, False)
        frames.append(_normalised(x * (1 + rng.normal(0, 0.02)) + rng.normal(0, 0.02)))
    outs = _chain(frames)
    assert _range(outs[-1]) > 0.9
    # frame-to-frame change of the output is smaller than that of the raw estimates
    d_out = np.mean([np.abs(outs[i] - outs[i - 1]).mean() for i in range(1, 600)])
    d_in = np.mean([np.abs(frames[i] - frames[i - 1]).mean() for i in range(1, 600)])
    assert d_out < 0.7 * d_in


def test_scene_cut_restarts_the_chain():
    rng = np.random.default_rng(3)
    a, b = _scene_a(), _scene_b()
    frames = [_normalised(a + _noise(rng, 0.02, False)) for _ in range(200)]
    frames += [_normalised(b + _noise(rng, 0.02, False)) for _ in range(10)]
    assert not scene_cut(frames[198], frames[199])
    assert scene_cut(frames[199], frames[200])
    assert not scene_cut(None, frames[0])
    outs = _chain(frames)
    np.testing.assert_array_equal(outs[200], frames[200])  # starts over at the cut
    for i in range(200, 210):  # full range right after the cut, and it stays
        assert _range(outs[i]) > 0.95
    # the new shot's depth is followed, not the old one's
    assert np.corrcoef(outs[-1].ravel(), b.ravel())[0, 1] > 0.95


def test_partially_correlated_cut_does_not_flatten_the_rest():
    # edited videos: a cut to a shot whose depth only partly resembles the previous one
    rng = np.random.default_rng(4)
    a, b = _scene_a(), _scene_b()
    mix = 0.55 * a + 0.45 * b
    frames = [_normalised(a + _noise(rng, 0.01, False)) for _ in range(60)]
    frames += [_normalised(mix + _noise(rng, 0.01, False)) for _ in range(60)]
    outs = _chain(frames)
    assert min(_range(o) for o in outs[60:]) > 0.85
    assert _range(outs[-1]) > 0.95


def test_range_pull_zero_keeps_the_pure_alignment():
    rng = np.random.default_rng(5)
    prev = rng.random((40, 60)).astype(np.float32) * 0.6 + 0.2
    cur = (prev - 0.2) / 0.6  # same depth, other normalisation
    out = temporal_align(prev, cur, ema=0.5, range_pull=0.0)
    np.testing.assert_allclose(out, prev, atol=2e-3)
    full = temporal_align(prev, cur, ema=0.5, range_pull=1.0)
    lo, hi = np.percentile(full, [2, 98])
    clo, chi = np.percentile(cur, [2, 98])
    assert abs(lo - clo) < 0.01 and abs(hi - chi) < 0.01
