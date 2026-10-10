"""Real downloaded models (CI 'models' job only: VRPHOTO_REAL_MODELS=1, network required)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.models,
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("VRPHOTO_REAL_MODELS") != "1", reason="set VRPHOTO_REAL_MODELS=1"),
]


def _scene() -> np.ndarray:
    """Floor/wall corridor-like synthetic image: bottom rows should come out nearer."""
    h, w = 384, 512
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (120 + 60 * np.sin(xx / 9.0) * (yy / h)).astype(np.uint8)
    img[..., 1] = (90 + 80 * (yy / h)).astype(np.uint8)
    img[..., 2] = 140
    img[h // 2 :] = (img[h // 2 :] * 0.6).astype(np.uint8)
    return img


@pytest.mark.parametrize("name", ["da2-small", "moge2"])
def test_real_estimator(name, monkeypatch):
    for var in ("VRPHOTO_MODEL_DA2_SMALL", "VRPHOTO_MODEL_MOGE2_VITL"):
        monkeypatch.delenv(var, raising=False)
    cache = os.environ.get("VRPHOTO_REAL_MODELS_HOME") or os.environ.get("VRPHOTO_TEST_MODELS_DIR")
    if cache:
        monkeypatch.setenv("VRPHOTO_HOME", cache)
    from vrphoto.depth import get_estimator

    r = get_estimator(name, device="auto").estimate(_scene())
    d = r.disparity
    assert d.shape == (384, 512) and np.isfinite(d).all()
    assert d.std() > 0.05
    if name == "moge2":
        assert r.intrinsics is not None and 20 < r.intrinsics.hfov_deg < 140
        assert r.metric_m is not None and np.isfinite(r.metric_m).all() and r.metric_m.min() > 0
