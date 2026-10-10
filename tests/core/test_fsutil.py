"""core.fsutil: retrying replace and atomic writes (WIN-2 shared helper, R4 short writes)."""

from __future__ import annotations

import os

import pytest

from vrphoto.core import fsutil


def test_replace_with_retry_plain(tmp_path):
    src, dst = tmp_path / "a.tmp", tmp_path / "a.bin"
    src.write_bytes(b"new")
    dst.write_bytes(b"old")
    fsutil.replace_with_retry(src, dst)
    assert dst.read_bytes() == b"new" and not src.exists()


def test_replace_with_retry_retries_permission_error_on_windows(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.tmp", tmp_path / "a.bin"
    src.write_bytes(b"x")
    real = os.replace
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 4:
            raise PermissionError(13, "sharing violation")
        real(a, b)

    sleeps: list[float] = []
    monkeypatch.setattr(fsutil.sys, "platform", "win32")
    monkeypatch.setattr(fsutil.os, "replace", flaky)
    monkeypatch.setattr(fsutil.time, "sleep", sleeps.append)
    fsutil.replace_with_retry(src, dst)
    assert dst.read_bytes() == b"x"
    assert calls["n"] == 4
    assert sleeps == pytest.approx([0.05, 0.10, 0.15])
    assert all(0.05 <= s <= 0.25 for s in sleeps)


def test_replace_with_retry_gives_up_and_caps_backoff(monkeypatch, tmp_path):
    def always(a, b):
        raise PermissionError(13, "locked")

    sleeps: list[float] = []
    monkeypatch.setattr(fsutil.sys, "platform", "win32")
    monkeypatch.setattr(fsutil.os, "replace", always)
    monkeypatch.setattr(fsutil.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        fsutil.replace_with_retry(tmp_path / "a", tmp_path / "b", attempts=8)
    assert len(sleeps) == 7 and max(sleeps) == pytest.approx(0.25)


def test_replace_with_retry_no_retry_off_windows_or_other_errors(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fail(a, b):
        calls["n"] += 1
        raise PermissionError(13, "locked")

    monkeypatch.setattr(fsutil.sys, "platform", "linux")
    monkeypatch.setattr(fsutil.os, "replace", fail)
    with pytest.raises(PermissionError):
        fsutil.replace_with_retry(tmp_path / "a", tmp_path / "b")
    assert calls["n"] == 1
    # FileNotFoundError is never retried, even on Windows
    monkeypatch.undo()
    monkeypatch.setattr(fsutil.sys, "platform", "win32")
    with pytest.raises(FileNotFoundError):
        fsutil.replace_with_retry(tmp_path / "missing", tmp_path / "b")


def test_atomic_write_bytes_roundtrip_and_no_leftovers(tmp_path):
    p = tmp_path / "out.jpg"
    p.write_bytes(b"old")
    fsutil.atomic_write_bytes(p, b"hello")
    assert p.read_bytes() == b"hello"
    fsutil.atomic_write_bytes(p, memoryview(b"world"), fsync=False)
    assert p.read_bytes() == b"world"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["out.jpg"]


def test_atomic_write_bytes_failure_keeps_old_file_and_cleans_temp(tmp_path, monkeypatch):
    p = tmp_path / "out.bin"
    p.write_bytes(b"old")

    def boom(a, b, **kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fsutil, "replace_with_retry", boom)
    with pytest.raises(OSError):
        fsutil.atomic_write_bytes(p, b"new")
    assert p.read_bytes() == b"old"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["out.bin"]


def test_atomic_write_bytes_short_write_raises(tmp_path, monkeypatch):
    """A failing write (disk full) raises instead of leaving a truncated file (R4)."""
    import builtins

    real_open = builtins.open

    class FullDisk:
        def __init__(self, fh):
            self.fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.fh.close()

        def write(self, data):
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

        def fileno(self):
            return self.fh.fileno()

    def fake_open(path, mode="r", *a, **kw):
        fh = real_open(path, mode, *a, **kw)
        return FullDisk(fh) if "x" in mode else fh

    monkeypatch.setattr(builtins, "open", fake_open)
    p = tmp_path / "x.jpg"
    with pytest.raises(OSError):
        fsutil.atomic_write_bytes(p, b"data")
    monkeypatch.undo()
    assert not p.exists()
    assert list(tmp_path.iterdir()) == []


def test_temp_sibling_same_folder_unique(tmp_path):
    a = fsutil.temp_sibling(tmp_path / "f.png")
    b = fsutil.temp_sibling(tmp_path / "f.png")
    assert a.parent == tmp_path and a != b and a.name.startswith(".f.png.")
