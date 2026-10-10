"""Error hints name GUI places that exist (E2E-3 / R9).

Add-ons and AI models are separate windows (toolbar / Tools menu); the Settings dialog has
no Add-ons section and cannot download or delete models.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import vrphoto
from vrphoto.core import errors

PKG = Path(vrphoto.__file__).resolve().parent
# the areas owned by this fix group; the rest of the package is covered by its own tests
AREAS = ["core", "models", "segment", "inpaint", "addons", "depth/base.py"]
STALE = re.compile(r"Settings > (Add-ons|Models)\b|Help > Open log folder|Video to 3D")


def _files():
    for a in AREAS:
        p = PKG / a
        if p.is_file():
            yield p
        else:
            yield from (f for f in p.rglob("*") if f.suffix in (".py", ".toml"))


def test_no_hint_points_to_missing_menus():
    bad = []
    for f in _files():
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if STALE.search(line):
                bad.append(f"{f.relative_to(PKG)}:{n}: {line.strip()}")
    assert not bad, "\n".join(bad)


def test_ui_constants_match_lead_decision():
    assert errors.UI_ADDONS == "Add-ons window (toolbar > Add-ons)"
    assert errors.UI_MODELS == "AI models window (toolbar > AI models)"
    assert errors.UI_LOG == "Help > Log"
    assert errors.MODE_VIDEO == "3D Video"


def test_runner_hints_use_addons_window():
    from vrphoto.addons.runner import error_hint

    hint = error_hint("ModuleNotFoundError: No module named 'torch'")
    assert hint and errors.UI_ADDONS in hint
    gated = error_hint("huggingface_hub.errors.GatedRepoError: 401 Client Error")
    assert gated and errors.UI_SETTINGS_MODELS in gated


def test_not_installed_hint(tmp_path):
    from vrphoto.addons import manager

    with pytest.raises(errors.AddonNotInstalled) as ei:
        manager.run("trellis", "image_to_3d", {}, {}, tmp_path)
    assert errors.UI_ADDONS in (ei.value.hint or "")


def test_ort_missing_model_hint(tmp_path):
    from vrphoto.models.ort import OrtModel

    with pytest.raises(errors.ModelUnavailable) as ei:
        OrtModel(tmp_path / "missing.onnx")
    assert errors.UI_MODELS in (ei.value.hint or "")
