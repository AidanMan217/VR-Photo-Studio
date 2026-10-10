"""get_segmenter factory and the auto fallback chain."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InferenceError, InputError, ModelUnavailable
from vrphoto.core.types import CancelToken
from vrphoto.segment import AutoSegmenter, Segmenter, get_segmenter, list_segmenters
from vrphoto.segment import _runtime as rt
from vrphoto.segment.birefnet import BiRefNetSegmenter
from vrphoto.segment.grabcut import GrabCutSegmenter
from vrphoto.segment.isnet import ISNetSegmenter, U2NetSegmenter


@pytest.mark.parametrize(
    ("name", "cls", "seg_id"),
    [
        ("birefnet-lite", BiRefNetSegmenter, "birefnet-lite"),
        ("birefnet-portrait", BiRefNetSegmenter, "birefnet-portrait"),
        ("isnet", ISNetSegmenter, "isnet"),
        ("isnet-general", ISNetSegmenter, "isnet"),
        ("u2net", U2NetSegmenter, "u2net"),
        ("grabcut", GrabCutSegmenter, "grabcut"),
        ("GrabCut ", GrabCutSegmenter, "grabcut"),
    ],
)
def test_factory_names(name, cls, seg_id):
    seg = get_segmenter(name)
    assert isinstance(seg, cls)
    assert seg.id == seg_id
    assert isinstance(seg, Segmenter)


def test_factory_does_not_load_models(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("must not load at construction")

    monkeypatch.setattr(rt, "open_model", fail)
    for name, _title in list_segmenters():
        get_segmenter(name)


def test_unknown_name():
    with pytest.raises(InputError):
        get_segmenter("magic")


def test_list_segmenters():
    names = [n for n, _ in list_segmenters()]
    assert names[0] == "auto"
    assert {"birefnet-lite", "birefnet-portrait", "isnet", "u2net", "grabcut"} <= set(names)


def test_auto_falls_back_to_grabcut(monkeypatch, photo_rgb, caplog):
    def unavailable(model_id, **kw):
        raise ModelUnavailable("offline")

    monkeypatch.setattr(rt, "open_model", unavailable)
    seg = get_segmenter("auto")
    assert isinstance(seg, AutoSegmenter)
    with caplog.at_level(logging.WARNING):
        alpha = seg.segment(photo_rgb)
    assert seg.id == "grabcut" and seg.resolved == "grabcut"
    assert alpha[170, 160] > 0.5 and alpha[10, 10] < 0.5
    assert any("falling back" in r.message for r in caplog.records)
    # a failed model is not retried on the next call
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: pytest.fail("retried"))
    seg.segment(photo_rgb)


def test_auto_falls_back_on_inference_error(monkeypatch, photo_rgb, fake_session_cls):
    def boom(feeds):
        raise RuntimeError("GPU exploded")

    sess = fake_session_cls(["input_image"], [[1, 3, 1024, 1024]], boom)
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: sess)
    seg = AutoSegmenter()
    alpha = seg.segment(photo_rgb)
    assert seg.resolved == "grabcut"
    assert alpha.shape == photo_rgb.shape[:2]


def test_auto_uses_model_when_available(monkeypatch, fake_session_cls, subject_img):
    sess = fake_session_cls(
        ["input_image"],
        [[1, 3, 1024, 1024]],
        lambda f: 8.0 * f["input_image"].mean(axis=1, keepdims=True),
    )
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: sess)
    seg = AutoSegmenter()
    assert seg.id == "birefnet-lite"
    alpha = seg.segment(subject_img)
    assert seg.resolved == "birefnet-lite"
    assert alpha[80, 100] > 0.5
    assert len(sess.calls) == 1


def test_auto_propagates_cancel(monkeypatch, photo_rgb):
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: pytest.fail("should not load"))
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        AutoSegmenter().segment(photo_rgb, cancel=tok)


def test_auto_propagates_input_error(monkeypatch):
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: pytest.fail("should not load"))
    with pytest.raises(InputError):
        AutoSegmenter().segment(np.zeros((5, 5, 3), np.float64))


def test_auto_skips_unregistered_models(monkeypatch, photo_rgb):
    monkeypatch.setattr(rt, "is_registered", lambda mid: False)
    monkeypatch.setattr(rt, "is_available", lambda mid: False)
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: pytest.fail("should not load"))
    seg = AutoSegmenter()
    assert seg.preferred == "grabcut"
    seg.segment(photo_rgb)
    assert seg.resolved == "grabcut"


def test_needs_download(monkeypatch):
    monkeypatch.setattr(rt, "is_available", lambda mid: False)
    monkeypatch.setattr(rt, "is_registered", lambda mid: True)
    assert AutoSegmenter().needs_download()
    monkeypatch.setattr(rt, "is_available", lambda mid: True)
    assert not AutoSegmenter().needs_download()


def test_runtime_open_model_wraps_errors(monkeypatch, tmp_path):
    with pytest.raises(ModelUnavailable):
        rt.open_model("birefnet-lite", path=tmp_path / "missing.onnx")
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx file")
    with pytest.raises(ModelUnavailable):
        rt.open_model("birefnet-lite", path=bad, device="cpu")


def test_run_model_wraps_exceptions(fake_session_cls):
    def boom(feeds):
        raise ValueError("nope")

    with pytest.raises(InferenceError):
        rt.run_model(fake_session_cls(["a"], [[1]], boom), {"a": np.zeros(1)}, model_id="u2net")
    with pytest.raises(InferenceError):
        rt.run_model(
            type("S", (), {"run": lambda self, f: {}})(),  # type: ignore[arg-type]
            {"a": np.zeros(1)},
            model_id="u2net",
        )


def test_env_override_helpers(monkeypatch, tmp_path):
    f = tmp_path / "m.onnx"
    f.write_bytes(b"x")
    assert rt.env_var_for("birefnet-lite") == "VRPHOTO_MODEL_BIREFNET_LITE"
    monkeypatch.setenv("VRPHOTO_MODEL_BIREFNET_LITE", str(f))
    assert rt.env_override("birefnet-lite") == f
    assert rt.is_available("birefnet-lite")
    monkeypatch.setenv("VRPHOTO_MODEL_BIREFNET_LITE", str(tmp_path / "missing"))
    assert rt.env_override("birefnet-lite") is None
