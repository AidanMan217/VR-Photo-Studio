"""LaMa wrapper: stub ONNX model (via the models layer) and fake sessions."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InferenceError, ModelUnavailable
from vrphoto.core.types import CancelToken, ProgressReporter
from vrphoto.inpaint.lama import LamaInpainter, output_scale


@pytest.fixture
def holes(textured_img):
    m = np.zeros(textured_img.shape[:2], bool)
    m[60:100, 80:140] = True
    m[0:20, 290:320] = True  # at the image border
    m[200:240, 0:30] = True
    return m


def test_stub_fills_holes_with_grey(stub_models, textured_img, holes):
    lama = LamaInpainter(model_path=stub_models["lama"], device="cpu")
    out = lama.inpaint(textured_img, holes)
    assert out.shape == textured_img.shape and out.dtype == np.uint8
    assert np.array_equal(out[~holes], textured_img[~holes])  # bit-identical outside
    for ys, xs in (
        (slice(70, 90), slice(95, 125)),
        (slice(2, 15), slice(295, 315)),
        (slice(210, 235), slice(5, 25)),
    ):
        np.testing.assert_allclose(out[ys, xs].reshape(-1, 3).mean(0), 128, atol=6)
    assert lama.id == "lama"


def test_stub_via_env_override(stub_models, stub_builder, textured_img, holes, monkeypatch):
    pytest.importorskip("vrphoto.models.ort")
    monkeypatch.setenv(stub_builder.env_var_for("lama"), str(stub_models["lama"]))
    from vrphoto.inpaint import get_inpainter

    out = get_inpainter("lama", device="cpu").inpaint(textured_img, holes)
    assert abs(int(out[80, 110, 0]) - 128) <= 3
    auto = get_inpainter("auto", device="cpu")
    out2 = auto.inpaint(textured_img, holes)
    assert auto.id == "lama"
    assert np.array_equal(out, out2)


def _lama_fake(fake_session_cls, scale=255.0, names=("image", "mask")):
    def fn(feeds):
        x, m = feeds[names[0]], feeds[names[1]]
        return ((x * (1 - m) + 0.5 * m) * scale).astype(np.float32)

    return fake_session_cls(list(names), [[1, 3, 512, 512], [1, 1, 512, 512]], fn)


def test_mask_convention_and_input_encoding(fake_session_cls, textured_img, holes):
    sess = _lama_fake(fake_session_cls)
    LamaInpainter(model=sess).inpaint(textured_img, holes)
    assert sess.calls
    for call in sess.calls:
        x, m = call["image"], call["mask"]
        assert x.shape == (1, 3, 512, 512) and x.dtype == np.float32
        assert m.shape == (1, 1, 512, 512) and m.dtype == np.float32
        assert set(np.unique(m)) <= {0.0, 1.0}  # binarised, 1 = fill
        assert 0.0 < m.mean() < 1.0
        assert 0.0 <= x.min() and x.max() <= 1.0  # RGB / 255
        assert np.all(x[:, :, m[0, 0] > 0] == 0.0)  # holes zeroed like LaMa training


def test_output_range_detection(fake_session_cls, textured_img, holes):
    out255 = LamaInpainter(model=_lama_fake(fake_session_cls, 255.0)).inpaint(textured_img, holes)
    out1 = LamaInpainter(model=_lama_fake(fake_session_cls, 1.0)).inpaint(textured_img, holes)
    assert np.array_equal(out255, out1)
    assert abs(int(out1[80, 110, 1]) - 128) <= 3
    forced = LamaInpainter(model=_lama_fake(fake_session_cls, 1.0), output_range="0-1").inpaint(
        textured_img, holes
    )
    assert np.array_equal(forced, out1)
    wrong = LamaInpainter(model=_lama_fake(fake_session_cls, 1.0), output_range="0-255").inpaint(
        textured_img, holes
    )
    assert wrong[80, 110, 1] <= 2  # 0.5 treated as 0..255 -> black


def test_output_scale_helper():
    crop = np.full((4, 4, 3), 200, np.uint8)
    hole = np.zeros((4, 4), bool)
    hole[1:3, 1:3] = True
    assert output_scale(crop.astype(np.float32), crop, hole) == 1.0
    assert output_scale(crop.astype(np.float32) / 255.0, crop, hole) == 255.0
    allhole = np.ones((4, 4), bool)
    assert output_scale(np.full((4, 4, 3), 0.7, np.float32), crop, allhole) == 255.0
    assert output_scale(np.full((4, 4, 3), 170.0, np.float32), crop, allhole) == 1.0


def test_input_names_resolved_by_shape(fake_session_cls, textured_img, holes):
    """Inputs named arbitrarily and in swapped order are found via their channel count."""

    def fn(feeds):
        m, x = feeds["in_b"], feeds["in_a"]
        return (x * (1 - m) + 0.5 * m) * 255.0

    sess = fake_session_cls(["in_b", "in_a"], [[1, 1, 512, 512], [1, 3, 512, 512]], fn)
    out = LamaInpainter(model=sess).inpaint(textured_img, holes)
    assert abs(int(out[80, 110, 0]) - 128) <= 3


def test_nhwc_output_accepted(fake_session_cls, textured_img, holes):
    def fn(feeds):
        x, m = feeds["image"], feeds["mask"]
        return ((x * (1 - m) + 0.5 * m) * 255.0).transpose(0, 2, 3, 1)

    sess = fake_session_cls(["image", "mask"], [[1, 3, 512, 512], [1, 1, 512, 512]], fn)
    out = LamaInpainter(model=sess).inpaint(textured_img, holes)
    assert abs(int(out[80, 110, 2]) - 128) <= 3


def test_bad_output_raises(fake_session_cls, textured_img, holes):
    sess = fake_session_cls(
        ["image", "mask"],
        [[1, 3, 512, 512], [1, 1, 512, 512]],
        lambda f: np.zeros((1, 3, 256, 256), np.float32),
    )
    with pytest.raises(InferenceError):
        LamaInpainter(model=sess).inpaint(textured_img, holes)
    one_input = fake_session_cls(["image"], [[1, 3, 512, 512]], lambda f: f["image"])
    with pytest.raises(InferenceError):
        LamaInpainter(model=one_input).inpaint(textured_img, holes)


def test_empty_mask_skips_model(fake_session_cls, textured_img):
    sess = _lama_fake(fake_session_cls)
    out = LamaInpainter(model=sess).inpaint(textured_img, np.zeros(textured_img.shape[:2], bool))
    assert np.array_equal(out, textured_img)
    assert not sess.calls


def test_missing_model_raises(tmp_path, textured_img, holes):
    with pytest.raises(ModelUnavailable):
        LamaInpainter(model_path=tmp_path / "missing.onnx").inpaint(textured_img, holes)


def test_progress_and_cancel(fake_session_cls, textured_img, holes):
    seen: list[float] = []
    LamaInpainter(model=_lama_fake(fake_session_cls)).inpaint(
        textured_img, holes, progress=ProgressReporter(lambda f, m: seen.append(f))
    )
    assert seen[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        LamaInpainter(model=_lama_fake(fake_session_cls)).inpaint(textured_img, holes, cancel=tok)


def test_lama_never_on_directml():
    from vrphoto.segment import _runtime as rt

    assert rt.spec_dml_ok("lama") is False
