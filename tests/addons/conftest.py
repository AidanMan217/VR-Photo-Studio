"""Fixtures for the add-on tests: test manifests, raw worker scripts, fake tools."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from _addon_helpers import dummy_manifest, write_manifest

from vrphoto.addons import manager


@pytest.fixture(autouse=True)
def _fresh_manifests(monkeypatch):
    """No cached manifests / extra manifest dirs leak between tests."""
    monkeypatch.delenv(manager.MANIFESTS_ENV, raising=False)
    manager.reload_manifests()
    yield
    manager.reload_manifests()


@pytest.fixture
def manifest_dir(tmp_path, monkeypatch) -> Path:
    """An extra manifest folder registered through VRPHOTO_ADDON_MANIFESTS."""
    d = tmp_path / "manifests"
    d.mkdir()
    monkeypatch.setenv(manager.MANIFESTS_ENV, str(d))
    manager.reload_manifests()
    return d


@pytest.fixture
def addon_python(monkeypatch) -> str:
    """Use this interpreter for add-on venvs (no managed-Python download)."""
    monkeypatch.setenv("VRPHOTO_ADDON_PYTHON", sys.executable)
    return sys.executable


@pytest.fixture
def uv_exe() -> str:
    try:
        return manager.find_uv()
    except Exception:
        pytest.skip("uv is not available")


@pytest.fixture
def installed_dummy(manifest_dir, addon_python, uv_exe) -> str:
    """A dummy add-on installed offline (venv + prefetch + check, no pip step)."""
    write_manifest(manifest_dir, "dummy", dummy_manifest("dummy"))
    manager.reload_manifests()
    manager.install("dummy")
    return "dummy"
