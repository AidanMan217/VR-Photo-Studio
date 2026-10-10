"""Depth Anything V2 estimator: preprocessing, orientation, output size (stub model)."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled
from vrphoto.core.types import CancelToken, ProgressReporter
from vrphoto.depth import da2
from vrphoto.depth.da2 import IMAGENET_MEAN, IMAGENET_STD, LONG_BUCKETS, DA2Estimator, input_size, preprocess


@pytest.mark.parametrize(
    "h,w",
    [(480, 640), (640, 480), (3000, 4000), (518, 518), (100, 150), (1080, 1920), (1000, 333), (37, 2000)],
)
def test_input_size_short_side_518_multiple_of_14(h, w):
    th, tw = input_size(h, w)
    assert th % 14 == 0 and tw % 14 == 0
    if max(h, w) / min(h, w) * 518 <= da2.MAX_LONG:
        assert min(th, tw) == 518
        assert abs((max(th, tw) / min(th, tw)) - max(h, w) / min(h, w)) < 0.05
    else:  # extreme panoramas: long side capped
        assert max(th, tw) <= da2.MAX_LONG + 7
    assert (th >= tw) == (h >= w)


def test_input_size_known_values():
    assert input_size(480, 640) == (518, 686)  # 690.67 -> 686
    assert input_size(3000, 4000) == (518, 686)
    assert input_size(1080, 1920) == (518, 924)  # 920.9 -> 924
    with pytest.raises(ValueError):
        input_size(0, 10)


def test_input_size_buckets():
    th, tw = input_size(480, 640, buckets=LONG_BUCKETS)
    assert th == 518 and tw in LONG_BUCKETS
    th, tw = input_size(4000, 3000, buckets=LONG_BUCKETS)
    assert tw == 518 and th in LONG_BUCKETS
    # nearby aspect ratios share a shape
    assert input_size(480, 640, buckets=LONG_BUCKETS) == input_size(470, 640, buckets=LONG_BUCKETS)
    assert all(b % 14 == 0 for b in LONG_BUCKETS)


def test_preprocess_normalisation():
    img = np.zeros((10, 20, 3), np.uint8)
    img[..., 0] = 255
    img[..., 1] = 128
    x = preprocess(img, (14, 28))
    assert x.shape == (1, 3, 14, 28) and x.dtype == np.float32 and x.flags.c_contiguous
    expect = (np.array([1.0, 128 / 255, 0.0], np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(x[0, :, 7, 14], expect, atol=1e-5)
    # identity size: no resize
    y = preprocess(np.full((14, 14, 3), 124, np.uint8), (14, 14))
    np.testing.assert_allclose(y[0, 0], (124 / 255 - 0.485) / 0.229, atol=1e-6)


def _bright_square_img(h=240, w=320):
    img = np.full((h, w, 3), 30, np.uint8)
    img[60:140, 200:300] = 240
    return img


def test_orientation_bright_region_is_near(use_stubs):
    img = _bright_square_img()
    r = DA2Estimator("da2-small", device="cpu").estimate(img)
    assert r.model_id == "da2-small"
    assert r.disparity.shape == img.shape[:2] and r.disparity.dtype == np.float32
    assert r.disparity.min() >= 0 and r.disparity.max() <= 1
    assert r.disparity[100, 250] > 0.9  # stub output = mean RGB -> bright = near
    assert r.disparity[200, 50] < 0.1
    assert r.raw is not None and r.raw.shape == img.shape[:2]


def test_network_receives_expected_shape(use_stubs, monkeypatch):
    est = DA2Estimator("da2-small", device="cpu")
    seen = {}
    img = np.random.default_rng(0).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    model = est._load(ProgressReporter.null(), None)
    orig = model.run

    def spy(feeds, output_names=None, **kw):
        seen.update({k: v.shape for k, v in feeds.items()})
        return orig(feeds, output_names, **kw)

    monkeypatch.setattr(model, "run", spy)
    r = est.estimate(img)
    assert seen == {"pixel_values": (1, 3, 518, 868)}  # 863.3 -> 868
    assert r.disparity.shape == (300, 500)


def test_buckets_used_on_directml(use_stubs, monkeypatch):
    est = DA2Estimator("da2-small", device="cpu")
    model = est._load(ProgressReporter.null(), None)
    assert est.net_size(480, 640, model) == (518, 686)
    monkeypatch.setattr(model, "provider", "DmlExecutionProvider")
    th, tw = est.net_size(480, 640, model)
    assert th == 518 and tw in LONG_BUCKETS
    assert DA2Estimator("da2-small", buckets=False).net_size(480, 640, model) == (518, 686)
    monkeypatch.setenv("VRPHOTO_DA2_BUCKETS", "1")
    monkeypatch.setattr(model, "provider", "CPUExecutionProvider")
    assert est.net_size(480, 640, model)[1] in LONG_BUCKETS


def test_static_input_shape_is_respected(use_stubs, monkeypatch):
    est = DA2Estimator("da2-small", device="cpu")
    model = est._load(ProgressReporter.null(), None)
    monkeypatch.setattr(model, "input_shapes", lambda: [[1, 3, 392, 392]])
    assert est.net_size(480, 640, model) == (392, 392)


def test_various_inputs(use_stubs):
    est = DA2Estimator("da2-small", device="cpu")
    for img in (
        np.zeros((40, 30), np.uint8),
        np.zeros((40, 30, 4), np.uint8),
        np.ones((20, 400, 3), np.float32),
    ):
        d = est.estimate(img).disparity
        assert d.shape == img.shape[:2] and np.isfinite(d).all()


def test_progress_and_cancel(use_stubs):
    ev: list[tuple[float, str]] = []
    DA2Estimator(device="cpu").estimate(
        _bright_square_img(), progress=ProgressReporter(lambda f, m: ev.append((f, m)))
    )
    fr = [f for f, _ in ev]
    assert fr == sorted(fr) and fr[-1] == 1.0
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        DA2Estimator(device="cpu").estimate(_bright_square_img(), cancel=tok)


def test_model_loaded_lazily_and_once(use_stubs, monkeypatch):
    calls = []
    from vrphoto.models import ort

    real = ort.load_model

    def counting(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(ort, "load_model", counting)
    est = DA2Estimator(device="cpu")
    assert calls == []  # construction does not load
    est.estimate(_bright_square_img())
    est.estimate(_bright_square_img())
    assert len(calls) == 1
