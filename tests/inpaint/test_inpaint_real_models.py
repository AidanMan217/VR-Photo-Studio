"""Real-model smoke tests (download weights).  Run with VRPHOTO_REAL_MODELS=1.

Weights are cached in ``$VRPHOTO_REAL_MODELS_HOME`` (default ``~/.cache/vrphoto-test-models``)
instead of the per-test isolated home so they download only once.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.models,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("VRPHOTO_REAL_MODELS") != "1", reason="set VRPHOTO_REAL_MODELS=1"),
]


@pytest.fixture
def real_home(monkeypatch):
    home = Path(os.environ.get("VRPHOTO_REAL_MODELS_HOME", Path.home() / ".cache" / "vrphoto-test-models"))
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VRPHOTO_HOME", str(home))
    from vrphoto.core import config

    config.set_settings(None)
    yield home
    config.set_settings(None)


@pytest.mark.parametrize("name", ["lama", "migan"])
def test_real_inpainter_fills_from_context(real_home, name):
    from vrphoto.inpaint import get_inpainter

    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([90 + xx * 60 // w, 140 + 0 * xx, 200 - yy * 40 // h], -1).astype(np.uint8)
    mask = np.zeros((h, w), bool)
    mask[180:300, 250:400] = True
    garbage = img.copy()
    garbage[mask] = (255, 0, 255)
    out = get_inpainter(name, device="cpu").inpaint(garbage, mask)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out[~mask], img[~mask])
    err = np.abs(out[mask].astype(int) - img[mask].astype(int)).mean()
    assert err < 20, err  # a smooth gradient must be continued, not magenta
