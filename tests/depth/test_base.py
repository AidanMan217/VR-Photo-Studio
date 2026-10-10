"""Estimator factory, auto fallback, input validation, lazy imports."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InferenceError, InputError
from vrphoto.core.types import CancelToken, DepthResult, ProgressReporter
from vrphoto.depth import DepthEstimator, get_estimator, list_estimators
from vrphoto.depth.base import AutoEstimator, as_rgb
from vrphoto.depth.da2 import DA2Estimator
from vrphoto.depth.heuristic import HeuristicEstimator
from vrphoto.depth.moge import MoGeEstimator


def test_list_estimators_matches_contract():
    ids = [i for i, _ in list_estimators()]
    assert ids == ["auto", "da2-small", "da2-base", "da2-large", "moge2", "da3", "heuristic"]
    assert all(title for _, title in list_estimators())


@pytest.mark.parametrize(
    "name,cls,mid",
    [
        ("auto", AutoEstimator, "da2-small"),
        ("heuristic", HeuristicEstimator, "heuristic"),
        ("da2-small", DA2Estimator, "da2-small"),
        ("da2-base", DA2Estimator, "da2-base"),
        ("da2-large", DA2Estimator, "da2-large"),
        ("moge2", MoGeEstimator, "moge2-vitl"),
        ("MoGe2", MoGeEstimator, "moge2-vitl"),
        ("", AutoEstimator, "da2-small"),
    ],
)
def test_get_estimator(name, cls, mid):
    est = get_estimator(name, device="cpu")
    assert isinstance(est, cls)
    assert isinstance(est, DepthEstimator)
    assert est.id == mid


def test_get_estimator_unknown():
    with pytest.raises(InputError) as ei:
        get_estimator("midas")
    assert "da2-small" in (ei.value.hint or "")


def test_construction_does_not_download(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("downloaded at construction")

    monkeypatch.setattr("vrphoto.models.download.ensure_model", boom)
    for name, _ in list_estimators():
        get_estimator(name)


def test_auto_uses_da2_when_available(use_stubs):
    est = get_estimator("auto", device="cpu")
    img = np.full((120, 160, 3), 30, np.uint8)
    img[:40, :40] = 250
    r = est.estimate(img)
    assert r.model_id == "da2-small" and est.id == "da2-small"
    assert r.disparity[20, 20] > 0.9


def test_auto_falls_back_to_heuristic_when_offline(monkeypatch, offline, caplog, photo_rgb):
    monkeypatch.delenv("VRPHOTO_MODEL_DA2_SMALL", raising=False)
    est = get_estimator("auto", device="cpu")
    with caplog.at_level("WARNING"):
        r = est.estimate(photo_rgb)
    assert r.model_id == "heuristic"
    assert est.id == "heuristic"
    assert "offline" in (est.fallback_reason or "")
    assert any("heuristic" in rec.message for rec in caplog.records)
    assert r.disparity.shape == photo_rgb.shape[:2]
    # sticky: no second download attempt
    calls = []
    monkeypatch.setattr("vrphoto.models.download.ensure_model", lambda *a, **k: calls.append(1))
    est.estimate(photo_rgb)
    assert calls == []


def test_auto_download_progress_flows_through_estimate(monkeypatch, use_stubs, photo_rgb):
    """The first estimate() triggers ensure_model with a progress reporter."""
    monkeypatch.delenv("VRPHOTO_MODEL_DA2_SMALL", raising=False)
    seen = {}

    def fake_ensure(model_id, *, progress=None, cancel=None, token=None, variant="default"):
        seen["progress"] = progress
        seen["cancel"] = cancel
        progress(0.5, "Downloading Depth Anything V2 Small: 50 / 99 MB")
        progress(1.0, "Downloaded")
        return use_stubs["da2-small"]

    monkeypatch.setattr("vrphoto.models.download.ensure_model", fake_ensure)
    events: list[tuple[float, str]] = []
    tok = CancelToken()
    r = get_estimator("auto", device="cpu").estimate(
        photo_rgb, progress=ProgressReporter(lambda f, m: events.append((f, m))), cancel=tok
    )
    assert r.model_id == "da2-small"
    assert seen["cancel"] is tok
    assert any("Downloading" in m and f == pytest.approx(0.25) for f, m in events)  # sub-range 0..0.5
    assert events[-1][0] == 1.0


def test_auto_propagates_cancel(monkeypatch, photo_rgb):
    monkeypatch.delenv("VRPHOTO_MODEL_DA2_SMALL", raising=False)

    def cancelled(*a, **k):
        raise Cancelled("Operation cancelled")

    monkeypatch.setattr("vrphoto.models.download.ensure_model", cancelled)
    est = get_estimator("auto")
    with pytest.raises(Cancelled):
        est.estimate(photo_rgb)
    assert est.id == "da2-small"  # cancellation is not a reason to fall back


def test_auto_inference_error_falls_back_for_that_image(monkeypatch, use_stubs, photo_rgb):
    est = get_estimator("auto", device="cpu")

    def broken(self, img, *, progress=None, cancel=None):
        raise InferenceError("model failed")

    monkeypatch.setattr(DA2Estimator, "estimate", broken)
    r = est.estimate(photo_rgb)
    assert r.model_id == "heuristic"
    assert est.id == "da2-small"  # not sticky


def test_explicit_model_offline_raises(monkeypatch, offline, photo_rgb):
    from vrphoto.core.errors import DownloadError

    monkeypatch.delenv("VRPHOTO_MODEL_MOGE2_VITL", raising=False)
    with pytest.raises(DownloadError):
        get_estimator("moge2").estimate(photo_rgb)


def test_all_estimators_contract(use_stubs, photo_rgb):
    """Every estimator returns DepthResult at input resolution with disparity in [0,1]."""
    img = photo_rgb[:, :300]  # odd aspect ratio
    for name, _ in list_estimators():
        if name in ("da2-base", "da2-large"):
            continue  # same code path as da2-small; no stub registered under those ids
        r = get_estimator(name, device="cpu").estimate(img)
        assert isinstance(r, DepthResult), name
        assert r.disparity.shape == img.shape[:2], name
        assert r.disparity.dtype == np.float32, name
        assert float(r.disparity.min()) >= 0.0 and float(r.disparity.max()) <= 1.0, name
        assert np.isfinite(r.disparity).all(), name


def test_as_rgb_conversions():
    assert as_rgb(np.zeros((4, 5), np.uint8)).shape == (4, 5, 3)
    assert as_rgb(np.zeros((4, 5, 4), np.uint8)).shape == (4, 5, 3)
    f = as_rgb(np.full((4, 5, 3), 0.5, np.float32))
    assert f.dtype == np.uint8 and f[0, 0, 0] == 128
    assert as_rgb(np.full((4, 5, 3), 65535, np.uint16))[0, 0, 0] == 255
    assert as_rgb(np.full((4, 5, 3), 300, np.int32))[0, 0, 0] == 255
    for bad in (np.zeros((1, 5, 3), np.uint8), np.zeros((4, 5, 2), np.uint8), np.zeros(5, np.uint8)):
        with pytest.raises(InputError):
            as_rgb(bad)


def test_importing_depth_is_lightweight():
    code = (
        "import sys, vrphoto.depth, vrphoto.depth.da2, vrphoto.depth.moge, vrphoto.depth.da3, "
        "vrphoto.depth.heuristic, vrphoto.models.ort, vrphoto.models.download, vrphoto.models.gpu; "
        "assert 'onnxruntime' not in sys.modules; assert 'PySide6' not in sys.modules; print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
