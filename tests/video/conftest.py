"""Fixtures for the video tests (session-cached synthetic clips)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from video_helpers import encode_clip, synth_frames


@pytest.fixture(scope="session")
def clip_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("clips")


@pytest.fixture(scope="session")
def h264_clip(clip_dir) -> Path:
    """64 frames 256x128 @ 30 fps, libx264, faststart (session-cached; copy before mutating)."""
    return encode_clip(clip_dir / "base_h264.mp4", synth_frames(), encoder="libx264")


@pytest.fixture(scope="session")
def hevc_clip(clip_dir) -> Path:
    """64 frames 256x128 @ 30 fps, libx265 / hvc1 (session-cached; copy before mutating)."""
    return encode_clip(clip_dir / "base_hevc.mp4", synth_frames(), preset="quest3_hevc", encoder="libx265")


@pytest.fixture
def h264_copy(h264_clip, tmp_path) -> Path:
    dst = tmp_path / "clip.mp4"
    shutil.copyfile(h264_clip, dst)
    return dst


@pytest.fixture
def hevc_copy(hevc_clip, tmp_path) -> Path:
    dst = tmp_path / "clip_hevc.mp4"
    shutil.copyfile(hevc_clip, dst)
    return dst
