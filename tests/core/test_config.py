"""core.config.default_library_dir follows the Windows Pictures known folder (E2E-9)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from vrphoto.core import config


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(config.Path, "home", classmethod(lambda cls: home))
    for k in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        monkeypatch.delenv(k, raising=False)
    return home


def test_default_library_uses_local_pictures(fake_home, monkeypatch):
    monkeypatch.setattr(config, "_known_pictures_dir", lambda: None)
    (fake_home / "Pictures").mkdir()
    assert config.default_library_dir() == fake_home / "Pictures" / "VR Photo Studio"


def test_default_library_falls_back_to_home(fake_home, monkeypatch):
    monkeypatch.setattr(config, "_known_pictures_dir", lambda: None)
    assert config.default_library_dir() == fake_home / "VR Photo Studio"


def test_default_library_follows_moved_pictures_folder(fake_home, tmp_path, monkeypatch):
    moved = tmp_path / "D" / "MyPics"
    moved.mkdir(parents=True)
    (fake_home / "Pictures").mkdir()
    monkeypatch.setattr(config, "_known_pictures_dir", lambda: moved)
    assert config.default_library_dir() == moved / "VR Photo Studio"


def test_default_library_avoids_onedrive_pictures(fake_home, tmp_path, monkeypatch):
    od = tmp_path / "home" / "OneDrive - Contoso"
    (od / "Pictures").mkdir(parents=True)
    (fake_home / "Pictures").mkdir()
    monkeypatch.setenv("OneDriveCommercial", str(od))
    monkeypatch.setattr(config, "_known_pictures_dir", lambda: od / "Pictures")
    assert config.default_library_dir() == fake_home / "Pictures" / "VR Photo Studio"


def test_is_cloud_synced(tmp_path, monkeypatch):
    for k in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "Cloud"
    monkeypatch.setenv("OneDrive", str(root))
    assert config._is_cloud_synced(root / "Pictures")
    assert not config._is_cloud_synced(tmp_path / "Pictures")
    assert config._is_cloud_synced(tmp_path / "OneDrive" / "Pictures")  # env var missing


def test_known_pictures_dir_none_off_windows(monkeypatch):
    if sys.platform == "win32":
        p = config._known_pictures_dir()
        assert p is None or isinstance(p, Path)
    else:
        assert config._known_pictures_dir() is None


def test_settings_save_waits_out_a_brief_windows_lock(tmp_path, monkeypatch):
    """WIN-2: settings.json is replaced through the shared retrying helper (unique temp file)."""
    import os

    from vrphoto.core import fsutil

    path = tmp_path / "settings.json"
    s = config.Settings()
    s.save(path)
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(32, "The process cannot access the file", str(dst))
        return real(src, dst)

    monkeypatch.setattr(fsutil.os, "replace", flaky)
    monkeypatch.setattr(fsutil.sys, "platform", "win32")
    monkeypatch.setattr(fsutil.time, "sleep", lambda _s: None)
    s.output_dir = str(tmp_path / "out")
    s.save(path)
    assert calls["n"] == 3
    assert config.Settings.load(path).output_dir == str(tmp_path / "out")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["settings.json"]
