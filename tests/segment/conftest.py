"""Fixtures for segmentation tests: stub ONNX models and a fake session class."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_stub_builder():
    spec = importlib.util.spec_from_file_location("make_test_onnx", ROOT / "scripts" / "make_test_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("make_test_onnx", mod)
    spec.loader.exec_module(mod)
    return mod


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
    """Stub ONNX models (real input/output names) built once per test session."""
    return stub_builder.build_all(tmp_path_factory.mktemp("stub_onnx"))


class FakeSession:
    """Session-like object: records feeds and returns ``fn(feeds)`` as the output."""

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
def subject_img() -> np.ndarray:
    """Dark 200x160 image with a bright 60x80 'subject' (what the BiRefNet stub detects)."""
    img = np.full((160, 200, 3), 25, np.uint8)
    img[40:120, 70:130] = (240, 235, 230)
    return img
