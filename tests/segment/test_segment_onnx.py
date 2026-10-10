"""BiRefNet / ISNet / U2Net wrappers: stub ONNX models and fake sessions."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InferenceError, InputError, ModelUnavailable
from vrphoto.core.types import CancelToken, ProgressReporter
from vrphoto.segment import _runtime as rt
from vrphoto.segment.birefnet import BiRefNetSegmenter
from vrphoto.segment.isnet import ISNetSegmenter, U2NetSegmenter


# ----------------------------------------------------------------------------- stub ONNX
def test_birefnet_stub_bright_subject(stub_models, subject_img):
    seg = BiRefNetSegmenter(model_path=stub_models["birefnet-lite"], device="cpu")
    alpha = seg.segment(subject_img)
    assert alpha.dtype == np.float32
    assert alpha.shape == subject_img.shape[:2]
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert alpha[45:115, 75:125].min() > 0.5  # bright subject
    assert alpha[:30].max() < 0.5  # dark background
    assert alpha[:, :60].max() < 0.5
    assert seg.id == "birefnet-lite"


def test_birefnet_via_env_override_and_factory(stub_models, stub_builder, subject_img, monkeypatch):
    pytest.importorskip("vrphoto.models.ort")
    monkeypatch.setenv(stub_builder.env_var_for("birefnet-lite"), str(stub_models["birefnet-lite"]))
    from vrphoto.segment import get_segmenter

    seg = get_segmenter("birefnet-lite", device="cpu")
    alpha = seg.segment(subject_img)
    assert alpha[80, 100] > 0.5 and alpha[5, 5] < 0.5
    auto = get_segmenter("auto", device="cpu")
    a2 = auto.segment(subject_img)
    assert auto.id == "birefnet-lite"
    np.testing.assert_allclose(a2, alpha, atol=1e-6)


def test_birefnet_rect_limits_subject(stub_models, subject_img):
    seg = BiRefNetSegmenter(model_path=stub_models["birefnet-lite"], device="cpu")
    alpha = seg.segment(subject_img, rect=(60, 30, 50, 100))
    assert alpha[80, 100] > 0.5  # inside rect and subject
    assert alpha[80, 120] == 0.0  # subject, but outside the rect
    assert alpha[:30].max() == 0.0


def test_progress_and_cancel(stub_models, subject_img):
    seg = BiRefNetSegmenter(model_path=stub_models["birefnet-lite"], device="cpu")
    seen: list[float] = []
    seg.segment(subject_img, progress=ProgressReporter(lambda f, m: seen.append(f)))
    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        seg.segment(subject_img, cancel=tok)


def test_missing_model_file_raises_model_unavailable(tmp_path, subject_img):
    seg = BiRefNetSegmenter(model_path=tmp_path / "nope.onnx")
    with pytest.raises(ModelUnavailable):
        seg.segment(subject_img)


# ----------------------------------------------------------------------------- fake sessions
def _mean_logit_session(fake_session_cls, size, k=8.0, name="input_image"):
    def fn(feeds):
        x = feeds[name]
        return (k * x.mean(axis=1, keepdims=True)).astype(np.float32)

    return fake_session_cls([name], [[1, 3, size, size]], fn, out_name="output_image")


def test_birefnet_preprocessing_matches_spec(fake_session_cls, subject_img):
    sess = _mean_logit_session(fake_session_cls, 1024)
    BiRefNetSegmenter(model=sess).segment(subject_img)
    x = sess.calls[0]["input_image"]
    assert x.shape == (1, 3, 1024, 1024) and x.dtype == np.float32
    # bright subject pixel (240,235,230) normalised with ImageNet mean/std
    exp = (np.array([240, 235, 230]) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np.testing.assert_allclose(x[0, :, 512, 512], exp, atol=1e-3)


def test_birefnet_applies_sigmoid(fake_session_cls):
    img = np.zeros((64, 64, 3), np.uint8)

    def fn(feeds):
        out = np.full((1, 1, 1024, 1024), -4.0, np.float32)
        out[..., 512:] = 4.0
        return out

    sess = fake_session_cls(["input_image"], [[1, 3, 1024, 1024]], fn)
    a = BiRefNetSegmenter(model=sess).segment(img)
    assert a[10, 5] == pytest.approx(1 / (1 + np.exp(4.0)), abs=1e-3)
    assert a[10, 60] == pytest.approx(1 / (1 + np.exp(-4.0)), abs=1e-3)


def test_dynamic_input_shape_uses_default_size(fake_session_cls, subject_img):
    sess = fake_session_cls(
        ["input_image"],
        [["batch", 3, "height", "width"]],
        lambda f: np.zeros((1, 1, 1024, 1024), np.float32),
    )
    BiRefNetSegmenter(model=sess).segment(subject_img)
    assert sess.calls[0]["input_image"].shape == (1, 3, 1024, 1024)


def test_isnet_normalisation_and_minmax(fake_session_cls, subject_img):
    def fn(feeds):
        x = next(iter(feeds.values()))
        return x.mean(axis=1, keepdims=True) * 3.0 + 7.0  # arbitrary affine: min-max undoes it

    sess = fake_session_cls(["input_image"], [[1, 3, 1024, 1024]], fn, out_name="output")
    seg = ISNetSegmenter(model=sess)
    a = seg.segment(subject_img)
    assert seg.id == "isnet" and seg.model_id == "isnet-general"
    x = sess.calls[0]["input_image"]
    np.testing.assert_allclose(x[0, :, 512, 512], np.array([240, 235, 230]) / 255.0 - 0.5, atol=2e-3)
    assert a.max() == pytest.approx(1.0, abs=1e-3)
    assert a.min() == pytest.approx(0.0, abs=1e-3)
    assert a[80, 100] > 0.9 and a[5, 5] < 0.1


def test_u2net_size_320_and_first_output(fake_session_cls, subject_img):
    def fn(feeds):
        x = feeds["input.1"]
        return x.mean(axis=1, keepdims=True)

    sess = fake_session_cls(["input.1"], [[1, 3, 320, 320]], fn, out_name="1959")
    seg = U2NetSegmenter(model=sess)
    a = seg.segment(subject_img)
    x = sess.calls[0]["input.1"]
    assert x.shape == (1, 3, 320, 320)
    exp = (np.array([240, 235, 230]) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np.testing.assert_allclose(x[0, :, 160, 160], exp, atol=1e-3)
    assert seg.id == "u2net"
    assert a[80, 100] > 0.9 and a[5, 5] < 0.1


def test_no_divide_by_image_max(fake_session_cls):
    """Unlike rembg, a dark image must not be brightened before normalisation."""
    img = np.full((50, 50, 3), 51, np.uint8)  # max = 51 -> rembg would feed 1.0
    sess = fake_session_cls(
        ["input.1"], [[1, 3, 320, 320]], lambda f: np.zeros((1, 1, 320, 320), np.float32), out_name="d0"
    )
    U2NetSegmenter(model=sess).segment(img)
    x = sess.calls[0]["input.1"]
    np.testing.assert_allclose(x[0, 0, 0, 0], (0.2 - 0.485) / 0.229, atol=1e-3)


def test_constant_minmax_output_gives_zeros(fake_session_cls, subject_img):
    sess = fake_session_cls(
        ["x"], [[1, 3, 1024, 1024]], lambda f: np.full((1, 1, 1024, 1024), 3.0, np.float32)
    )
    a = ISNetSegmenter(model=sess).segment(subject_img)
    assert a.max() == 0.0


def test_model_exception_becomes_inference_error(fake_session_cls, subject_img):
    def boom(feeds):
        raise RuntimeError("bad things")

    sess = fake_session_cls(["input_image"], [[1, 3, 1024, 1024]], boom)
    with pytest.raises(InferenceError, match="bad things"):
        BiRefNetSegmenter(model=sess).segment(subject_img)


def test_bad_output_shape(fake_session_cls, subject_img):
    sess = fake_session_cls(["input_image"], [[1, 3, 1024, 1024]], lambda f: np.zeros((5,), np.float32))
    with pytest.raises(InferenceError):
        BiRefNetSegmenter(model=sess).segment(subject_img)


def test_input_validation(fake_session_cls):
    sess = _mean_logit_session(fake_session_cls, 1024)
    seg = BiRefNetSegmenter(model=sess)
    with pytest.raises(InputError):
        seg.segment(np.zeros((10, 10, 3), np.float32))
    with pytest.raises(InputError):
        seg.segment(np.zeros((10, 10, 2), np.uint8))
    with pytest.raises(InputError):
        seg.segment(np.zeros((40, 40, 3), np.uint8), rect=(100, 100, 10, 10))
    # RGBA and grey are accepted
    assert seg.segment(np.zeros((20, 30, 4), np.uint8)).shape == (20, 30)
    assert seg.segment(np.zeros((20, 30), np.uint8)).shape == (20, 30)


def test_registry_model_fetched_from_models_layer_each_time(monkeypatch, fake_session_cls, subject_img):
    """Registry sessions are cached by the models layer, not by the wrapper, so a model
    deleted / re-downloaded in Settings never leaves a stale (closed) session behind."""
    calls = []
    sess = _mean_logit_session(fake_session_cls, 1024)

    def fake_open(model_id, **kw):
        calls.append((model_id, kw.get("path")))
        return sess

    monkeypatch.setattr(rt, "open_model", fake_open)
    seg = BiRefNetSegmenter("birefnet-portrait")
    seg.segment(subject_img)
    seg.segment(subject_img)
    assert calls == [("birefnet-portrait", None), ("birefnet-portrait", None)]
    assert seg.id == "birefnet-portrait"


def test_explicit_path_model_opened_once(monkeypatch, fake_session_cls, subject_img, tmp_path):
    calls = []
    sess = _mean_logit_session(fake_session_cls, 1024)

    def fake_open(model_id, **kw):
        calls.append(kw.get("path"))
        return sess

    monkeypatch.setattr(rt, "open_model", fake_open)
    seg = BiRefNetSegmenter(model_path=tmp_path / "m.onnx")
    seg.segment(subject_img)
    seg.segment(subject_img)
    assert calls == [tmp_path / "m.onnx"]


def test_concurrent_segment_calls(fake_session_cls, subject_img):
    from concurrent.futures import ThreadPoolExecutor

    sess = _mean_logit_session(fake_session_cls, 1024)
    seg = BiRefNetSegmenter(model=sess)
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: seg.segment(subject_img), range(4)))
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])
