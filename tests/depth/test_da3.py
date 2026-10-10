"""Depth Anything 3 (generic community export) on stub models."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.core.types import ProgressReporter
from vrphoto.depth.da3 import DA3Estimator, da3_input_size, depth_to_disparity
from vrphoto.models.ort import OrtModel


def _img():
    img = np.full((210, 300, 3), 20, np.uint8)
    img[50:120, 180:280] = 245
    return img


def test_input_size():
    assert da3_input_size(480, 640) == (378, 504)
    th, tw = da3_input_size(1000, 333)
    assert th == 504 and tw % 14 == 0


def test_depth_to_disparity():
    d = np.array([[1.0, 2.0], [4.0, -1.0]], np.float32)
    inv = depth_to_disparity(d)
    assert inv[0, 0] > inv[0, 1] > inv[1, 0] > inv[1, 1]  # invalid -> farthest
    sky = np.array([[False, True], [False, False]])
    inv2 = depth_to_disparity(d, sky)
    assert inv2[0, 1] < inv2[1, 0]
    assert not depth_to_disparity(np.zeros((2, 2), np.float32)).any()


def test_static_depth_stub_orientation(use_stubs):
    r = DA3Estimator(device="cpu").estimate(_img())
    assert r.model_id == "da3-mono-large"
    assert r.disparity.shape == (210, 300) and r.disparity.dtype == np.float32
    # stub outputs DEPTH that is smaller for bright pixels -> bright square must be near
    assert r.disparity[85, 230] > 0.9
    assert r.disparity[180, 30] < 0.1


def test_rank5_input(use_stubs):
    est = DA3Estimator(device="cpu")
    est._model = OrtModel(use_stubs["da3-rank5"], "cpu")
    r = est.estimate(_img())
    assert r.disparity[85, 230] > 0.9


def test_disparity_named_output(use_stubs):
    est = DA3Estimator(device="cpu")
    est._model = OrtModel(use_stubs["da3-disp"], "cpu")
    assert est.kind_for("disparity") == "disparity"
    r = est.estimate(_img())
    # output named "disparity" is used as-is: stub value is larger for DARK pixels here
    assert r.disparity[85, 230] < 0.1


def test_output_kind_override_and_pick():
    assert DA3Estimator.pick_output(["feat", "depth", "sky"]) == "depth"
    assert DA3Estimator.pick_output(["pred_disp", "depth"]) == "pred_disp"
    assert DA3Estimator.pick_output(["out0"]) == "out0"
    assert DA3Estimator(output_kind="disparity").kind_for("depth") == "disparity"
    assert DA3Estimator().kind_for("depth") == "depth"


class FakeDA3:
    provider = "CPUExecutionProvider"

    def __init__(self):
        self.seen = None

    def input_names(self):
        return ["x"]

    def input_shapes(self):
        return [[1, 3, "h", "w"]]

    def output_names(self):
        return ["depth", "sky"]

    def run(self, feeds, output_names=None, **kw):
        x = feeds["x"]
        self.seen = (x.shape, output_names)
        h, w = x.shape[2:]
        depth = np.tile(np.linspace(10, 1, h, dtype=np.float32)[:, None], (1, w))[None]
        sky = np.zeros((1, h, w), np.float32)
        sky[0, : h // 4] = 1.0
        return {"depth": depth, "sky": sky}


def test_dynamic_model_with_sky_output():
    est = DA3Estimator(device="cpu")
    fake = FakeDA3()
    est._model = fake
    r = est.estimate(np.zeros((480, 640, 3), np.uint8), progress=ProgressReporter.null())
    assert fake.seen == ((1, 3, 378, 504), ["depth", "sky"])
    rows = r.disparity.mean(axis=1)
    assert rows[-1] > rows[300] > rows[200]  # depth decreasing downward -> nearer at bottom
    assert rows[10] <= rows[200] + 1e-6  # sky region is far


@pytest.mark.parametrize("name", ["da3", "da3-mono-large"])
def test_factory(name):
    from vrphoto.depth import get_estimator

    assert isinstance(get_estimator(name), DA3Estimator)
