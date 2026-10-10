"""Fixtures for inpainting tests: stub ONNX models, a MI-GAN-like uint8 stub, fakes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

#: colour the MI-GAN stub paints into holes
MIGAN_FILL = (200, 30, 30)


def _load_stub_builder():
    spec = importlib.util.spec_from_file_location("make_test_onnx", ROOT / "scripts" / "make_test_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("make_test_onnx", mod)
    spec.loader.exec_module(mod)
    return mod


def make_migan_stub(path: Path) -> Path:
    """MI-GAN pipeline stand-in: uint8 ``image`` [1,3,H,W] + uint8 ``mask`` [1,1,H,W]
    (255 = known, 0 = hole) -> uint8 ``result`` = image where known, MIGAN_FILL in holes."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    img = helper.make_tensor_value_info("image", TensorProto.UINT8, ["batch", 3, "height", "width"])
    mask = helper.make_tensor_value_info("mask", TensorProto.UINT8, ["batch", 1, "height", "width"])
    out = helper.make_tensor_value_info("result", TensorProto.UINT8, ["batch", 3, "height", "width"])
    inits = [
        numpy_helper.from_array(np.array(MIGAN_FILL, np.uint8).reshape(1, 3, 1, 1), "fill"),
        numpy_helper.from_array(np.array(128.0, np.float32), "half"),
    ]
    nodes = [
        helper.make_node("Cast", ["mask"], ["maskf"], to=TensorProto.FLOAT),
        helper.make_node("Less", ["maskf", "half"], ["hole"]),
        helper.make_node("Where", ["hole", "fill", "image"], ["result"]),
    ]
    g = helper.make_graph(nodes, "migan_stub", [img, mask], [out], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(path))
    return path


@pytest.fixture(autouse=True)
def _reset_download_failures():
    """The "recent download failure" memo is process-wide: isolate every test."""
    from vrphoto.segment import _runtime as rt

    rt.reset_download_failures()
    yield
    rt.reset_download_failures()


@pytest.fixture(scope="session")
def stub_builder():
    pytest.importorskip("onnx")
    return _load_stub_builder()


@pytest.fixture(scope="session")
def stub_models(tmp_path_factory, stub_builder) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("stub_onnx")
    paths = dict(stub_builder.build_all(d))
    paths["migan"] = make_migan_stub(d / "migan.onnx")
    return paths


class FakeSession:
    """Session-like object: records feeds and returns ``fn(feeds)`` as output."""

    def __init__(self, names, shapes, fn, out_name="output"):
        self._names = list(names)
        self._shapes = [list(s) for s in shapes]
        self._fn = fn
        self.out_name = out_name
        self.calls: list[dict[str, np.ndarray]] = []

    def input_names(self):
        return list(self._names)

    def input_shapes(self):
        return [list(s) for s in self._shapes]

    def run(self, feeds, output_names=None):
        self.calls.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        return {self.out_name: self._fn(feeds)}


@pytest.fixture
def fake_session_cls():
    return FakeSession


@pytest.fixture
def textured_img() -> np.ndarray:
    """Deterministic 240x320 smooth-ish colour image."""
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    r = 128 + 60 * np.sin(xx / 17.0)
    g = 128 + 60 * np.cos(yy / 13.0)
    b = (xx + yy) * 255 // (h + w)
    return np.stack([r, g, b], -1).astype(np.uint8)
