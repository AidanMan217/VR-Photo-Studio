"""End to end: layered photo -> camera path -> frames -> encode -> spatial metadata -> decode."""

from __future__ import annotations

import numpy as np
from video_helpers import ffmpeg_banner

from vrphoto.core.types import Layer, LayeredImage, StereoParams
from vrphoto.video import (
    VideoWriter,
    iter_frames,
    make_path,
    probe,
    render_path_frames,
    render_vr180_frames,
    spatial,
)


def _li(scene) -> LayeredImage:
    h, w = scene.inv_z.shape
    return LayeredImage([Layer(scene.img, scene.inv_z, np.ones((h, w), np.float32))], scene.K)


def test_parallax_sbs_video(tmp_path, two_plane_scene):
    li = _li(two_plane_scene)
    path = make_path("swing", n_frames=12, fps=24.0)
    out = tmp_path / "photo_LR.mp4"
    h, w = two_plane_scene.inv_z.shape
    with VideoWriter(out, 2 * w, h, path.fps, "h264_compat", encoder="libx264") as vw:
        for frame in render_path_frames(li, path, stereo=StereoParams(), layout="sbs"):
            vw.write(frame)
    info = probe(out)
    assert (info.width, info.height, info.n_frames) == (2 * w, h, 12)
    assert info.fps == 24.0


def test_vr180_video_with_metadata(tmp_path, two_plane_scene):
    li = _li(two_plane_scene)
    path = make_path("circle", n_frames=10, amplitude_m=0.03)
    eye = 128
    frames = list(render_vr180_frames(li, path, eye_px=eye))
    out = tmp_path / "photo_180_LR.mp4"
    with VideoWriter(out, 2 * eye, eye, path.fps, "quest3_hevc", encoder="libx265") as vw:
        for f in frames:
            vw.write(f)
    spatial.inject(out, coverage=180, stereo="lr")
    meta = spatial.read(out)
    assert meta["spherical"] and meta["coverage"] == 180 and meta["stereo"] == "lr"
    banner = ffmpeg_banner(out)
    assert "hvc1" in banner and "side by side" in banner and "tiled equirectangular" in banner
    dec = list(iter_frames(out))
    assert len(dec) == 10
    assert np.abs(dec[3].astype(int) - frames[3].astype(int)).mean() < 6
