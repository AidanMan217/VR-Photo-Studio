"""get_inpainter factory and the auto fallback chain (lama -> migan -> opencv)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, DownloadError, InputError, ModelUnavailable
from vrphoto.core.types import CancelToken
from vrphoto.inpaint import AutoInpainter, Inpainter, get_inpainter, list_inpainters
from vrphoto.inpaint.lama import LamaInpainter
from vrphoto.inpaint.migan import MiganInpainter
from vrphoto.inpaint.opencv import OpenCVInpainter
from vrphoto.segment import _runtime as rt


@pytest.fixture
def holes(textured_img):
    m = np.zeros(textured_img.shape[:2], bool)
    m[60:100, 80:140] = True
    return m


@pytest.mark.parametrize(
    ("name", "cls", "inp_id"),
    [
        ("lama", LamaInpainter, "lama"),
        ("migan", MiganInpainter, "migan"),
        ("opencv", OpenCVInpainter, "opencv"),
        ("telea", OpenCVInpainter, "opencv"),
        ("opencv-ns", OpenCVInpainter, "opencv-ns"),
        ("ns", OpenCVInpainter, "opencv-ns"),
        ("pushpull", OpenCVInpainter, "pushpull"),
        (" LaMa", LamaInpainter, "lama"),
    ],
)
def test_factory_names(name, cls, inp_id):
    inp = get_inpainter(name)
    assert isinstance(inp, cls)
    assert inp.id == inp_id
    assert isinstance(inp, Inpainter)


def test_factory_never_loads(monkeypatch):
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: pytest.fail("loaded at construction"))
    for name, _ in list_inpainters():
        get_inpainter(name)


def test_unknown_name_raises_but_auto_never_does():
    with pytest.raises(InputError):
        get_inpainter("photoshop")
    assert isinstance(get_inpainter("auto"), AutoInpainter)
    assert isinstance(get_inpainter(""), AutoInpainter)


def test_list_inpainters():
    names = [n for n, _ in list_inpainters()]
    assert names[0] == "auto"
    assert {"lama", "migan", "opencv", "pushpull"} <= set(names)


def test_auto_falls_back_lama_migan_opencv(monkeypatch, textured_img, holes, caplog):
    tried: list[str] = []

    def unavailable(model_id, **kw):
        tried.append(model_id)
        raise DownloadError(f"{model_id}: no network")

    monkeypatch.setattr(rt, "open_model", unavailable)
    inp = get_inpainter("auto")
    with caplog.at_level(logging.WARNING):
        out = inp.inpaint(textured_img, holes)
    assert tried == ["lama", "migan"]
    assert inp.resolved == "opencv" and inp.id == "opencv"
    assert np.array_equal(out[~holes], textured_img[~holes])
    assert sum("falling back" in r.message for r in caplog.records) == 2
    # failures are remembered: no new download attempts
    tried.clear()
    inp.inpaint(textured_img, holes)
    assert tried == []


def test_auto_uses_migan_when_lama_fails(monkeypatch, fake_session_cls, textured_img, holes):
    def fn(feeds):
        img, mk = feeds["image"], feeds["mask"]
        out = img.copy()
        out[np.broadcast_to(mk == 0, img.shape)] = 77
        return out

    migan_sess = fake_session_cls(["image", "mask"], [[1, 3, 512, 512], [1, 1, 512, 512]], fn, "result")

    def opener(model_id, **kw):
        if model_id == "lama":
            raise ModelUnavailable("corrupt")
        return migan_sess

    monkeypatch.setattr(rt, "open_model", opener)
    inp = AutoInpainter()
    out = inp.inpaint(textured_img, holes)
    assert inp.resolved == "migan"
    assert abs(int(out[80, 110, 0]) - 77) <= 3


def test_auto_inference_error_falls_back(monkeypatch, fake_session_cls, textured_img, holes):
    def boom(feeds):
        raise RuntimeError("DirectML device removed")

    sess = fake_session_cls(["image", "mask"], [[1, 3, 512, 512], [1, 1, 512, 512]], boom)
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: sess)
    inp = AutoInpainter()
    out = inp.inpaint(textured_img, holes)
    assert inp.resolved == "opencv"
    assert out.shape == textured_img.shape


def test_auto_prefers_lama(monkeypatch, fake_session_cls, textured_img, holes):
    def fn(feeds):
        x, m = feeds["image"], feeds["mask"]
        return (x * (1 - m) + 0.5 * m) * 255.0

    sess = fake_session_cls(["image", "mask"], [[1, 3, 512, 512], [1, 1, 512, 512]], fn)
    opened: list[str] = []

    def opener(model_id, **kw):
        opened.append(model_id)
        return sess

    monkeypatch.setattr(rt, "open_model", opener)
    inp = AutoInpainter()
    assert inp.preferred == "lama"
    out = inp.inpaint(textured_img, holes)
    assert opened == ["lama"] and inp.resolved == "lama"
    assert abs(int(out[80, 110, 0]) - 128) <= 3


def test_auto_skips_unknown_models(monkeypatch, textured_img, holes):
    monkeypatch.setattr(rt, "is_registered", lambda mid: False)
    monkeypatch.setattr(rt, "is_available", lambda mid: False)
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: pytest.fail("should not load"))
    inp = AutoInpainter()
    assert inp.preferred == "opencv"
    assert inp.needs_download() is False
    inp.inpaint(textured_img, holes)
    assert inp.resolved == "opencv"


def test_auto_custom_candidates_always_end_model_free(monkeypatch, textured_img, holes):
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: (_ for _ in ()).throw(ModelUnavailable("x")))
    inp = AutoInpainter(["migan"])
    assert inp.candidates == ["migan", "opencv"]
    inp.inpaint(textured_img, holes)
    assert inp.resolved == "opencv"
    assert AutoInpainter(["pushpull"]).candidates == ["pushpull"]


def test_auto_propagates_cancel_and_input_errors(monkeypatch, textured_img, holes):
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: pytest.fail("should not load"))
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        AutoInpainter().inpaint(textured_img, holes, cancel=tok)
    with pytest.raises(InputError):
        AutoInpainter().inpaint(textured_img, holes[:10])


def test_auto_empty_mask_no_model(monkeypatch, textured_img):
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: pytest.fail("should not load"))
    out = AutoInpainter().inpaint(textured_img, np.zeros(textured_img.shape[:2], bool))
    assert np.array_equal(out, textured_img)


def test_auto_thread_safe(monkeypatch, textured_img, holes):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(rt, "open_model", lambda *a, **k: (_ for _ in ()).throw(ModelUnavailable("x")))
    inp = AutoInpainter()
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: inp.inpaint(textured_img, holes), range(6)))
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])
    assert inp.resolved == "opencv"
