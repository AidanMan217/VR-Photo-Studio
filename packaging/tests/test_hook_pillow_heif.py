"""Tests for the custom PyInstaller hook packaging/hooks/hook-pillow_heif.py."""

from __future__ import annotations

import runpy
from importlib import metadata
from pathlib import Path

import pytest

pytest.importorskip("PyInstaller")
pytest.importorskip("pillow_heif")

HOOK = Path(__file__).resolve().parents[1] / "hooks" / "hook-pillow_heif.py"


def _run_hook() -> dict:
    return runpy.run_path(str(HOOK))


def test_hook_hidden_imports():
    ns = _run_hook()
    assert "_pillow_heif" in ns["hiddenimports"]
    assert "pillow_heif.HeifImagePlugin" in ns["hiddenimports"]


def test_hook_collects_installed_libheif():
    ns = _run_hook()
    names = [Path(src).name.lower() for src, _dest in ns["binaries"]]
    assert any("heif" in n for n in names), names
    for src, dest in ns["binaries"]:
        assert Path(src).is_file()
        assert ".." not in Path(dest).parts


class _FakeFile:
    def __init__(self, rel: str, base: Path):
        self._rel = rel
        self._base = base

    def __str__(self) -> str:
        return self._rel

    def locate(self) -> Path:
        return self._base / self._rel


class _FakeDist:
    def __init__(self, files):
        self.files = files


@pytest.fixture
def windows_style_dist(tmp_path, monkeypatch):
    """Simulate the delvewheel 'platlib' layout of the Windows wheel."""
    site = tmp_path / "site-packages"
    rels = [
        "_pillow_heif.cp312-win_amd64.pyd",
        "libheif-30fc1b3c.dll",
        "libde265-0-863aced2.dll",
        "libx265-217-8a7f7f4e.dll",
        "pillow_heif/__init__.py",
        "pillow_heif.libs/libextra-1.so.2",
        "../../Scripts/evil.dll",
        "pillow_heif-1.8.0.dist-info/RECORD",
    ]
    for rel in rels:
        p = site / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    files = [_FakeFile(r, site) for r in rels] + [_FakeFile("libmissing.dll", site)]
    real = metadata.distribution

    def fake_distribution(name):
        if name == "pillow-heif":
            return _FakeDist(files)
        return real(name)

    monkeypatch.setattr(metadata, "distribution", fake_distribution)
    return site


def test_hook_record_layout_windows(windows_style_dist):
    ns = _run_hook()
    got = {(Path(src).name, dest) for src, dest in ns["binaries"] if str(windows_style_dist) in src}
    assert ("libheif-30fc1b3c.dll", ".") in got
    assert ("libde265-0-863aced2.dll", ".") in got
    assert ("libx265-217-8a7f7f4e.dll", ".") in got
    assert ("libextra-1.so.2", "pillow_heif.libs") in got
    names = {n for n, _ in got}
    assert "_pillow_heif.cp312-win_amd64.pyd" not in names  # collected as a module instead
    assert "evil.dll" not in names  # outside site-packages
    assert "libmissing.dll" not in names  # listed but not on disk


def test_hook_without_distribution(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    ns = _run_hook()
    assert isinstance(ns["binaries"], list)
