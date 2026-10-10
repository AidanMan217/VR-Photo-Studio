"""Review follow-ups: session reload, auto-fallback recovery, cancel plumbing, edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.core.errors import DownloadError, ModelUnavailable
from vrphoto.core.types import CancelToken
from vrphoto.depth.base import AutoEstimator
from vrphoto.depth.da2 import DA2Estimator
from vrphoto.depth.da3 import MIN_SKY_PIXELS, SKY_THRESHOLD, DA3Estimator, sky_mask
from vrphoto.depth.moge import MoGeEstimator


def _img(h=120, w=160):
    img = np.full((h, w, 3), 30, np.uint8)
    img[:40, :40] = 250
    return img


def test_estimator_reloads_a_closed_session(use_stubs, monkeypatch):
    """Deleting / re-downloading a model in Settings closes cached sessions; estimators
    held by the run context must transparently load it again."""
    from vrphoto.models import ort

    calls = []
    real = ort.load_model

    def counting(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(ort, "load_model", counting)
    est = DA2Estimator(device="cpu")
    first = est.estimate(_img())
    ort.unload_model("da2-small")  # what delete_model() does
    assert est._model.closed
    second = est.estimate(_img())
    assert len(calls) == 2
    np.testing.assert_allclose(first.disparity, second.disparity, atol=1e-6)


@pytest.mark.parametrize(
    "cls,mid", [(DA2Estimator, "da2-small"), (MoGeEstimator, "moge2-vitl"), (DA3Estimator, "da3-mono-large")]
)
def test_estimators_pass_cancel_into_inference(use_stubs, monkeypatch, cls, mid):
    est = cls(mid, device="cpu")
    model = est._load(None, None)
    seen = {}
    orig = model.run

    def spy(feeds, output_names=None, **kw):
        seen.update(kw)
        return orig(feeds, output_names, **kw)

    monkeypatch.setattr(model, "run", spy)
    tok = CancelToken()
    est.estimate(_img(), cancel=tok)
    assert seen.get("cancel") is tok


def test_auto_retries_the_model_after_the_window(monkeypatch, onnx_stubs):
    monkeypatch.delenv("VRPHOTO_MODEL_DA2_SMALL", raising=False)

    def offline(model_id, **kw):
        raise DownloadError("You appear to be offline.")

    monkeypatch.setattr("vrphoto.models.download.ensure_model", offline)
    est = AutoEstimator(device="cpu", retry_after_s=0.0)
    assert est.estimate(_img()).model_id == "heuristic"
    assert est.id == "heuristic"
    monkeypatch.setattr(
        "vrphoto.models.download.ensure_model", lambda model_id, **kw: onnx_stubs["da2-small"]
    )
    r = est.estimate(_img())
    assert r.model_id == "da2-small" and est.id == "da2-small" and est.fallback_reason is None


def test_auto_recovers_as_soon_as_the_model_is_on_disk(monkeypatch, onnx_stubs):
    from vrphoto.models import download

    monkeypatch.delenv("VRPHOTO_MODEL_DA2_SMALL", raising=False)
    real_ensure = download.ensure_model
    calls = []

    def ensure(model_id, **kw):
        calls.append(model_id)
        if download.env_override(model_id) is None:
            raise DownloadError("You appear to be offline.")
        return real_ensure(model_id, **kw)

    monkeypatch.setattr(download, "ensure_model", ensure)
    est = AutoEstimator(device="cpu")  # default window: minutes
    assert est.estimate(_img()).model_id == "heuristic"
    assert est.estimate(_img()).model_id == "heuristic"
    assert len(calls) == 1  # sticky within the window
    # the user downloads the model from Settings > Models (here: the file appears)
    monkeypatch.setenv("VRPHOTO_MODEL_DA2_SMALL", str(onnx_stubs["da2-small"]))
    assert est.estimate(_img()).model_id == "da2-small"


def test_auto_does_not_retry_a_broken_local_model_per_image(monkeypatch, use_stubs):
    calls = []

    def broken(self, img, *, progress=None, cancel=None):
        calls.append(1)
        raise ModelUnavailable("Cannot load the Depth Anything V2 Small model: INVALID_PROTOBUF")

    monkeypatch.setattr(DA2Estimator, "estimate", broken)
    est = AutoEstimator(device="cpu")
    assert est.estimate(_img()).model_id == "heuristic"
    assert est.estimate(_img()).model_id == "heuristic"
    assert calls == [1]  # file is present but unusable: wait for the retry window


class _BlankMoGe:
    provider = "CPUExecutionProvider"

    def input_names(self):
        return ["image", "num_tokens"]

    def input_shapes(self):
        return [[1, 3, "h", "w"], []]

    def output_names(self):
        return ["points", "mask", "metric_scale"]

    def run(self, feeds, output_names=None, **kw):
        _, _, h, w = feeds["image"].shape
        v, u = np.mgrid[0:h, 0:w].astype(np.float32)
        pts = np.stack([(u - w / 2) / w, (v - h / 2) / w, np.ones_like(u)], -1)
        return {
            "points": pts[None],
            "mask": np.zeros((1, h, w), np.float32),
            "metric_scale": np.array([2.0], np.float32),
        }


def test_moge_without_valid_geometry_reports_no_camera():
    est = MoGeEstimator(device="cpu")
    est._model = _BlankMoGe()
    r = est.estimate(np.full((90, 120, 3), 200, np.uint8))
    assert r.intrinsics is None and r.metric_m is None  # EXIF / default FOV take over
    assert r.disparity.shape == (90, 120) and np.isfinite(r.disparity).all()


def test_da3_sky_mask_matches_upstream_threshold():
    s = np.zeros((40, 40), np.float32)
    s[:10] = SKY_THRESHOLD  # >= 0.3 is sky upstream
    s[10:12] = SKY_THRESHOLD - 0.01
    m = sky_mask(s)
    assert m is not None and m[:10].all() and not m[10:].any()
    tiny = np.zeros((40, 40), np.float32)
    tiny.flat[:MIN_SKY_PIXELS] = 1.0
    assert sky_mask(tiny) is None  # too few sky pixels: ignored like upstream
    assert sky_mask(np.ones((40, 40), np.float32)) is None  # all sky: ignored too
    assert sky_mask(np.full((4, 4), np.nan, np.float32)) is None


# --------------------------------------------------------------------------- fast guided filter
def _edge_scene(h=240, w=320):
    import cv2

    rng = np.random.default_rng(0)
    img = cv2.resize(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    d = cv2.resize(rng.random((12, 16), dtype=np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return img, np.clip(d, 0, 1).astype(np.float32)


def test_auto_subsample_thresholds():
    from vrphoto.depth.post import auto_subsample

    assert auto_subsample(1920 * 1080, 12) == 1  # normal photos: exact filter
    assert auto_subsample(4096 * 3072, 25) == 3
    assert auto_subsample(8000 * 6000, 48) == 5
    assert auto_subsample(8000 * 6000, 2) == 2  # never coarser than the radius


def test_fast_guided_filter_matches_exact():
    from vrphoto.depth.post import guided_filter

    img, d = _edge_scene(480, 640)
    for guide in (img, img[..., 1]):
        exact = guided_filter(d, guide, r=16, eps=1e-3)
        fast = guided_filter(d, guide, r=16, eps=1e-3, subsample=4)
        assert fast.shape == d.shape and fast.dtype == np.float32
        assert float(np.abs(fast - exact).mean()) < 0.01
    # edge preservation: a step in src aligned with a step in the guide survives
    step_src = np.zeros((100, 200), np.float32)
    step_src[:, 100:] = 1.0
    step_guide = (step_src * 255).astype(np.uint8)
    out = guided_filter(step_src, step_guide, r=12, eps=1e-4, subsample=3)
    assert out[:, :90].max() < 0.1 and out[:, 110:].min() > 0.9


def test_refine_and_finalize_switch_to_fast_filter_for_big_images(monkeypatch):
    from vrphoto.depth import post

    img, d = _edge_scene()
    exact = post.refine(d, img)
    seen = []
    real = post.guided_filter

    def spy(*a, **k):
        seen.append(k.get("subsample", 1))
        return real(*a, **k)

    monkeypatch.setattr(post, "guided_filter", spy)
    monkeypatch.setattr(post, "FAST_GF_MIN_PIXELS", 10_000)
    monkeypatch.setattr(post, "FAST_GF_TARGET_PIXELS", 5_000)
    fast = post.refine(d, img)
    post.finalize_disparity(d[::4, ::4], img)
    assert seen[0] > 1 and seen[1] > 1
    assert float(np.abs(fast - exact).mean()) < 0.02
