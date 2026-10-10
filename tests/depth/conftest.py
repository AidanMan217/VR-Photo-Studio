"""Fixtures for depth estimator tests: stub ONNX models wired through env overrides."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def onnx_stubs(tmp_path_factory) -> dict[str, Path]:
    """Stub ONNX models from scripts/make_test_onnx.py (plus a rank-5 DA3 variant)."""
    name = "_vrphoto_script_make_test_onnx"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "make_test_onnx.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    d = tmp_path_factory.mktemp("depth_stubs")
    paths = dict(mod.build_all(d))
    paths["da3-rank5"] = mod.make_da3(d / "da3-rank5.onnx", rank5=True)
    paths["da3-disp"] = mod.make_da3(d / "da3-disp.onnx", h=196, w=252, output="disparity")
    paths["nan"] = mod.make_nan(d / "nan.onnx")
    return paths


@pytest.fixture
def use_stubs(onnx_stubs, monkeypatch) -> dict[str, Path]:
    """Point VRPHOTO_MODEL_<ID> at the stubs so no download is attempted."""
    from vrphoto.models.registry import env_var_for

    for mid in ("da2-small", "moge2-vitl", "da3-mono-large", "lama", "birefnet-lite"):
        monkeypatch.setenv(env_var_for(mid), str(onnx_stubs[mid]))
    yield onnx_stubs
    ort = sys.modules.get("vrphoto.models.ort")
    if ort is not None:
        ort.unload_model()


@pytest.fixture
def offline(monkeypatch):
    """Make every model download fail as if there were no network."""
    from vrphoto.core.errors import DownloadError

    def fail(model_id, **kw):
        raise DownloadError(f"Cannot reach huggingface.co to download {model_id}. You appear to be offline.")

    monkeypatch.setattr("vrphoto.models.download.ensure_model", fail)
    return fail
