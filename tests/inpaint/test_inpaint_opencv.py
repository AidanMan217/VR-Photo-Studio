"""OpenCV Telea / NS and push-pull inpainters."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken
from vrphoto.inpaint.opencv import OpenCVInpainter


@pytest.fixture
def two_tone():
    img = np.zeros((120, 160, 3), np.uint8)
    img[:, :80] = (200, 40, 40)
    img[:, 80:] = (40, 40, 200)
    m = np.zeros((120, 160), bool)
    m[40:80, 20:50] = True  # inside the red half
    m[40:80, 110:140] = True  # inside the blue half
    img[m] = (0, 255, 0)  # garbage to be replaced
    return img, m


@pytest.mark.parametrize("method", ["telea", "ns", "pushpull"])
def test_fills_from_surroundings(two_tone, method):
    img, m = two_tone
    out = OpenCVInpainter(method).inpaint(img, m)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out[~m], img[~m])
    red = out[45:75, 25:45].reshape(-1, 3).mean(0)
    blue = out[45:75, 115:135].reshape(-1, 3).mean(0)
    np.testing.assert_allclose(red, (200, 40, 40), atol=25)
    np.testing.assert_allclose(blue, (40, 40, 200), atol=25)


def test_ids():
    assert OpenCVInpainter("telea").id == "opencv"
    assert OpenCVInpainter("ns").id == "opencv-ns"
    assert OpenCVInpainter("pushpull").id == "pushpull"
    with pytest.raises(InputError):
        OpenCVInpainter("magic")  # type: ignore[arg-type]


def test_pushpull_smooth_gradient():
    h, w = 64, 64
    xx = np.tile(np.linspace(0, 255, w), (h, 1))
    img = np.stack([xx, xx, xx], -1).astype(np.uint8)
    m = np.zeros((h, w), bool)
    m[20:44, 20:44] = True
    out = OpenCVInpainter("pushpull").inpaint(img, m)
    # the fill varies smoothly from the left to the right edge of the hole
    row = out[32, 20:44, 0].astype(int)
    assert row[0] < row[-1]
    assert np.abs(np.diff(row)).max() < 25


def test_empty_and_full_masks(two_tone):
    img, _ = two_tone
    inp = OpenCVInpainter()
    out = inp.inpaint(img, np.zeros(img.shape[:2], bool))
    assert np.array_equal(out, img) and out is not img
    full = inp.inpaint(img, np.ones(img.shape[:2], bool))
    assert np.all(full == 128)


def test_radius_auto_and_explicit():
    assert OpenCVInpainter()._radius(480, 640) == 3
    assert OpenCVInpainter()._radius(3000, 4000) == 9
    assert OpenCVInpainter(radius=7)._radius(10, 10) == 7


def test_validation_and_cancel(two_tone):
    img, m = two_tone
    with pytest.raises(InputError):
        OpenCVInpainter().inpaint(img, m[:5])
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        OpenCVInpainter().inpaint(img, m, cancel=tok)
