"""Fixtures for the models package tests: stub ONNX files and clean ORT state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    mod_name = f"_vrphoto_script_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def make_onnx():
    """The scripts/make_test_onnx.py module."""
    return _load_script("make_test_onnx")


@pytest.fixture(scope="session")
def pin_models_mod():
    """The scripts/pin_models.py module."""
    return _load_script("pin_models")


@pytest.fixture(scope="session")
def stub_models(tmp_path_factory, make_onnx) -> dict[str, Path]:
    """Stub ONNX models (real I/O names) + extra variants (fp16, nan, rank-5 da3)."""
    d = tmp_path_factory.mktemp("stub_onnx")
    paths = dict(make_onnx.build_all(d))
    paths["da2-fp16"] = make_onnx.make_da2_fp16(d / "da2-fp16.onnx")
    paths["nan"] = make_onnx.make_nan(d / "nan.onnx")
    paths["slow"] = make_onnx.make_slow(d / "slow.onnx")
    return paths


@pytest.fixture(autouse=True)
def _clean_ort_state(monkeypatch):
    """Fresh session cache / sticky fallbacks and instant download retries per test."""
    from vrphoto.models import download, ort

    monkeypatch.setattr(download, "BACKOFF_BASE_S", 0.0)
    monkeypatch.setattr(download, "CHUNK_SIZE", 64 * 1024)
    monkeypatch.setattr(download, "PROGRESS_INTERVAL_S", 0.0)
    ort.unload_model()
    ort.reset_cpu_fallbacks()
    yield
    ort.unload_model()
    ort.reset_cpu_fallbacks()
