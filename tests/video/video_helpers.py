"""Helpers for the video tests (synthetic clips, ffprobe/ffmpeg inspection)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

W, H, N = 256, 128, 64


def synth_frame(i: int, w: int = W, h: int = H) -> np.ndarray:
    """Smooth, codec-friendly frame whose red channel encodes the frame index."""
    f = np.empty((h, w, 3), np.uint8)
    f[..., 0] = (i * 4) % 256
    f[..., 1] = (np.arange(w) * 255 // max(w - 1, 1)).astype(np.uint8)[None, :]
    f[..., 2] = (np.arange(h) * 255 // max(h - 1, 1)).astype(np.uint8)[:, None]
    return f


def synth_frames(n: int = N, w: int = W, h: int = H) -> list[np.ndarray]:
    return [synth_frame(i, w, h) for i in range(n)]


def ffmpeg() -> str:
    from vrphoto.video.encode import ffmpeg_exe

    return ffmpeg_exe()


def ffmpeg_banner(path: Path) -> str:
    """stderr of ``ffmpeg -i path`` (stream description incl. side data)."""
    res = subprocess.run([ffmpeg(), "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True)
    return res.stderr.decode("utf-8", errors="replace")


def ffprobe_side_data(path: Path) -> list[dict] | None:
    """Side data of the first video stream via ffprobe JSON, or None without ffprobe."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    res = subprocess.run(
        [exe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_entries",
         "stream_side_data", "-of", "json", str(path)],
        capture_output=True, check=True,
    )  # fmt: skip
    streams = json.loads(res.stdout.decode("utf-8")).get("streams", [])
    return streams[0].get("side_data_list", []) if streams else []


def top_level_boxes(path: Path) -> list[bytes]:
    from vrphoto.video.spatial import _top_level_order

    return _top_level_order(path)


def make_tone(path: Path, seconds: float = 3.0) -> Path:
    """Write an AAC test tone (440 Hz sine via lavfi) to ``path``."""
    subprocess.run(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         f"sine=frequency=440:sample_rate=48000:duration={seconds}", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )  # fmt: skip
    return path


def encode_clip(path: Path, frames: list[np.ndarray], fps: float = 30.0, preset: str = "h264_compat",
                **kw) -> Path:  # fmt: skip
    from vrphoto.video.encode import VideoWriter

    h, w = frames[0].shape[:2]
    with VideoWriter(path, w, h, fps, preset, **kw) as vw:
        for f in frames:
            vw.write(f)
    return path
