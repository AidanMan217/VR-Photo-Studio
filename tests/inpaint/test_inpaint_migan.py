"""MI-GAN wrapper: the inverted mask convention (255 = known, 0 = hole)."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.inpaint.lama import LamaInpainter
from vrphoto.inpaint.migan import HOLE, KNOWN, MiganInpainter

MIGAN_FILL = (200, 30, 30)  # must match conftest.make_migan_stub


@pytest.fixture
def holes(textured_img):
    m = np.zeros(textured_img.shape[:2], bool)
    m[60:100, 80:140] = True
    m[220:240, 290:320] = True
    return m


def _migan_fake(fake_session_cls):
    """Behaves like the real pipeline: fills where mask == 0 (hole), keeps mask == 255."""

    def fn(feeds):
        img, mk = feeds["image"], feeds["mask"]
        assert img.dtype == np.uint8 and mk.dtype == np.uint8
        out = img.copy()
        hole = np.broadcast_to(mk == HOLE, img.shape)
        out[hole] = np.broadcast_to(np.array(MIGAN_FILL, np.uint8).reshape(1, 3, 1, 1), img.shape)[hole]
        return out

    return fake_session_cls(
        ["image", "mask"], [["b", 3, "h", "w"], ["b", 1, "h", "w"]], fn, out_name="result"
    )


def test_mask_is_inverted_uint8(fake_session_cls, textured_img, holes):
    sess = _migan_fake(fake_session_cls)
    out = MiganInpainter(model=sess).inpaint(textured_img, holes)
    for call in sess.calls:
        img, mk = call["image"], call["mask"]
        assert img.shape == (1, 3, 512, 512) and img.dtype == np.uint8
        assert mk.shape == (1, 1, 512, 512) and mk.dtype == np.uint8
        assert set(np.unique(mk)) <= {HOLE, KNOWN}
        assert (mk == KNOWN).mean() > 0.5  # most of the crop is known context
    assert np.array_equal(out[~holes], textured_img[~holes])
    np.testing.assert_allclose(out[70:90, 95:125].reshape(-1, 3).mean(0), MIGAN_FILL, atol=8)
    np.testing.assert_allclose(out[226:236, 296:316].reshape(-1, 3).mean(0), MIGAN_FILL, atol=10)


def test_convention_mismatch_is_detectable(fake_session_cls, textured_img, holes):
    """Feeding MI-GAN a LaMa-style mask would paint the KNOWN area: make sure we don't."""
    sess = _migan_fake(fake_session_cls)
    MiganInpainter(model=sess).inpaint(textured_img, holes)
    mk = sess.calls[0]["mask"][0, 0]
    # LaMa convention for the same crop is the exact inverse
    lama_like = LamaInpainter().encode(
        type(
            "M",
            (),
            {
                "input_names": lambda s: ["image", "mask"],
                "input_shapes": lambda s: [[1, 3, 1, 1], [1, 1, 1, 1]],
            },
        )(),
        sess.calls[0]["image"][0].transpose(1, 2, 0),
        mk == HOLE,
    )["mask"][0, 0]
    assert np.array_equal(lama_like == 1.0, mk == HOLE)
    assert np.array_equal(lama_like == 0.0, mk == KNOWN)


def test_real_onnx_stub_through_models_layer(stub_models, textured_img, holes):
    pytest.importorskip("vrphoto.models.ort")
    mig = MiganInpainter(model_path=stub_models["migan"], device="cpu")
    out = mig.inpaint(textured_img, holes)
    assert mig.id == "migan"
    assert np.array_equal(out[~holes], textured_img[~holes])
    np.testing.assert_allclose(out[70:90, 95:125].reshape(-1, 3).mean(0), MIGAN_FILL, atol=8)


def test_float_output_is_converted(fake_session_cls, textured_img, holes):
    def fn(feeds):
        img, mk = feeds["image"], feeds["mask"]
        out = img.astype(np.float32) / 255.0
        out[np.broadcast_to(mk == HOLE, img.shape)] = 0.25
        return out

    sess = fake_session_cls(["image", "mask"], [[1, 3, 512, 512], [1, 1, 512, 512]], fn, out_name="result")
    out = MiganInpainter(model=sess).inpaint(textured_img, holes)
    assert abs(int(out[80, 110, 0]) - 64) <= 3
