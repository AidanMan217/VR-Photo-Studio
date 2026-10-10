"""Review fixes: auto-chain priority/failure memory, download-failure memo, large-hole
OpenCV path, masked downscale helper."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, DownloadError, ModelUnavailable
from vrphoto.core.types import CancelToken
from vrphoto.inpaint import AutoInpainter
from vrphoto.inpaint import opencv as cvmod
from vrphoto.inpaint.crops import masked_downscale
from vrphoto.inpaint.opencv import OpenCVInpainter
from vrphoto.segment import _runtime as rt


@pytest.fixture
def holes(textured_img):
    m = np.zeros(textured_img.shape[:2], bool)
    m[60:100, 80:140] = True
    return m


def _lama_sess(fake_session_cls):
    def fn(feeds):
        x, m = feeds["image"], feeds["mask"]
        return (x * (1 - m) + 0.5 * m) * 255.0

    return fake_session_cls(["image", "mask"], [[1, 3, 512, 512], [1, 1, 512, 512]], fn)


def _gradient(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([60 + xx * 120 // w, 100 + yy * 80 // h, 150 + 0 * xx], -1).astype(np.uint8)


# ----------------------------------------------------------------------------- auto chain
def test_auto_returns_to_lama_once_downloaded(monkeypatch, fake_session_cls, textured_img, holes):
    """After an offline fallback, a LaMa download made later (Settings) is used again."""
    avail = {"lama": False, "migan": False}
    monkeypatch.setattr(rt, "is_available", lambda mid: avail.get(mid, False))
    monkeypatch.setattr(rt, "is_registered", lambda mid: True)
    sess = _lama_sess(fake_session_cls)
    opened: list[str] = []

    def opener(model_id, **kw):
        opened.append(model_id)
        if not avail[model_id]:
            raise DownloadError("offline")
        return sess

    monkeypatch.setattr(rt, "open_model", opener)
    inp = AutoInpainter()
    inp.inpaint(textured_img, holes)
    assert inp.resolved == "opencv" and opened == ["lama", "migan"]
    inp.inpaint(textured_img, holes)
    assert opened == ["lama", "migan"]  # failures remembered while nothing changed
    avail["lama"] = True
    assert inp.preferred == "lama"
    out = inp.inpaint(textured_img, holes)
    assert inp.resolved == "lama" and opened[-1] == "lama"
    assert abs(int(out[80, 110, 0]) - 128) <= 3


def test_auto_failure_expires_after_ttl(monkeypatch, textured_img, holes):
    clock = [0.0]
    monkeypatch.setattr(rt, "_clock", lambda: clock[0])
    monkeypatch.setattr(rt, "is_available", lambda mid: True)  # present but corrupt
    opened: list[str] = []

    def corrupt(model_id, **kw):
        opened.append(model_id)
        raise ModelUnavailable("Cannot load model: INVALID_PROTOBUF")

    monkeypatch.setattr(rt, "open_model", corrupt)
    inp = AutoInpainter()
    inp.inpaint(textured_img, holes)
    assert opened == ["lama", "migan"]
    clock[0] += rt.FAILURE_TTL_S / 2
    inp.inpaint(textured_img, holes)
    assert opened == ["lama", "migan"]
    clock[0] += rt.FAILURE_TTL_S
    inp.inpaint(textured_img, holes)
    assert opened == ["lama", "migan", "lama", "migan"]


def test_new_auto_instance_skips_recent_download_failure(monkeypatch, textured_img, holes):
    monkeypatch.setattr(rt, "is_available", lambda mid: False)
    monkeypatch.setattr(rt, "is_registered", lambda mid: True)
    opened: list[str] = []

    def offline(model_id, **kw):
        opened.append(model_id)
        rt.note_download_failure(model_id)  # what the real open_model does
        raise DownloadError("offline")

    monkeypatch.setattr(rt, "open_model", offline)
    AutoInpainter().inpaint(textured_img, holes)
    assert opened == ["lama", "migan"]
    second = AutoInpainter()
    assert second.preferred == "opencv" and not second.needs_download()
    out = second.inpaint(textured_img, holes)
    assert opened == ["lama", "migan"]
    assert np.array_equal(out[~holes], textured_img[~holes])


def test_auto_last_resort_is_reported(monkeypatch, textured_img, holes):
    monkeypatch.setattr(rt, "open_model", lambda *a, **k: (_ for _ in ()).throw(ModelUnavailable("x")))
    inp = AutoInpainter(["lama"])
    monkeypatch.setattr(
        cvmod.OpenCVInpainter,
        "_cv_inpaint",
        lambda self, *a: (_ for _ in ()).throw(cvmod.InferenceError("opencv broke")),
    )
    out = inp.inpaint(textured_img, holes)
    assert inp.resolved == "pushpull"
    assert np.array_equal(out[~holes], textured_img[~holes])


# ----------------------------------------------------------------------------- opencv
@pytest.mark.parametrize("method", ["telea", "ns"])
def test_large_hole_filled_at_reduced_resolution(monkeypatch, method):
    h, w = 400, 600
    img = _gradient(h, w)
    m = np.zeros((h, w), bool)
    m[100:300, 150:450] = True  # 60 000 px > max_hole_px
    bad = img.copy()
    bad[m] = (255, 0, 255)
    calls: list[tuple[int, int]] = []
    real = cvmod.cv2.inpaint

    def spy(src, mask, radius, flags):
        calls.append(src.shape[:2])
        return real(src, mask, radius, flags)

    monkeypatch.setattr(cvmod.cv2, "inpaint", spy)
    out = OpenCVInpainter(method, max_hole_px=15_000).inpaint(bad, m)
    assert len(calls) == 1 and calls[0][0] < h and calls[0][1] < w
    assert np.array_equal(out[~m], img[~m])
    assert not (out[m] == (255, 0, 255)).all(axis=-1).any()  # no hole content leaks back
    err = np.abs(out[m].astype(int) - img[m].astype(int)).mean()
    assert err < 25, err
    # the border of the fill meets its surroundings (seam blending)
    ring = np.abs(out[100, 150:450].astype(int) - img[99, 150:450].astype(int)).mean()
    assert ring < 12, ring


def test_small_hole_stays_full_resolution(monkeypatch):
    img = _gradient(200, 300)
    m = np.zeros((200, 300), bool)
    m[90:110, 100:140] = True
    calls: list[tuple[int, int]] = []
    real = cvmod.cv2.inpaint
    monkeypatch.setattr(
        cvmod.cv2, "inpaint", lambda s, mk, r, f: (calls.append(s.shape[:2]), real(s, mk, r, f))[1]
    )
    OpenCVInpainter("telea").inpaint(img, m)
    assert calls == [(200, 300)]


def test_large_hole_cancel_checked(monkeypatch):
    img = _gradient(300, 400)
    m = np.zeros((300, 400), bool)
    m[50:250, 50:350] = True
    tok = CancelToken()
    real = cvmod.cv2.inpaint

    def cancel_during(src, mask, radius, flags):
        tok.cancel()
        return real(src, mask, radius, flags)

    monkeypatch.setattr(cvmod.cv2, "inpaint", cancel_during)
    with pytest.raises(Cancelled):
        OpenCVInpainter("telea", max_hole_px=10_000).inpaint(img, m, cancel=tok)


def test_max_hole_px_has_a_floor():
    assert OpenCVInpainter("telea", max_hole_px=1).max_hole_px == 1024


# ----------------------------------------------------------------------------- masked downscale
def test_masked_downscale_never_leaks_hole_content():
    img = np.full((300, 400, 3), (40, 120, 200), np.uint8)
    hole = np.zeros((300, 400), bool)
    hole[100:200, 100:300] = True
    hole[0:3, 0:3] = True  # tiny hole: must still be a hole after shrinking
    img[hole] = (255, 0, 255)
    small, small_hole = masked_downscale(img, hole, 100, 75)
    assert small.shape == (75, 100, 3) and small.dtype == np.uint8
    assert small_hole.shape == (75, 100) and small_hole.dtype == bool
    assert small_hole[0, 0] and small_hole[37, 50]
    np.testing.assert_allclose(
        small.reshape(-1, 3).astype(float), [[40, 120, 200]] * small[..., 0].size, atol=1
    )


def test_masked_downscale_all_hole_is_grey():
    img = np.zeros((40, 40, 3), np.uint8)
    small, small_hole = masked_downscale(img, np.ones((40, 40), bool), 10, 10)
    assert small_hole.all() and (small == 128).all()
