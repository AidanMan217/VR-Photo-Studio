"""Regression tests for the second review round (pipeline): video depth range and cuts,
capture scale, scene movement range, NTSC rates, stereo screen size, flat-video format,
360 resolution, unwritable / full output folders, retrying renames, hints, add-on options."""

from __future__ import annotations

import errno
import json
import os
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from vrphoto.core import errors
from vrphoto.core.errors import InputError, VRPhotoError
from vrphoto.core.types import CancelToken, DepthResult, SplatCloud
from vrphoto.pipeline import context
from vrphoto.pipeline import run as R
from vrphoto.pipeline.run import run_job


@pytest.fixture
def fake_estimator(monkeypatch):
    def install(est):
        monkeypatch.setattr(context.RunContext, "estimator", lambda self, *a, **k: est)
        return est

    return install


def _normalised(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> Path:
    from vrphoto.video.encode import VideoWriter

    h, w = frames[0].shape[:2]
    with VideoWriter(path, w, h, fps, "h264_compat") as vw:
        for f in frames:
            vw.write(f)
    return path


def _photo(path: Path, w: int, h: int) -> Path:
    from PIL import Image

    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) * 255 // (w + h)], -1).astype(np.uint8)
    img[h // 2 : h * 9 // 10, w * 2 // 5 : w * 3 // 5] = (200, 60, 60)
    Image.fromarray(img).save(path, quality=92)
    return path


def _box_glb(path: Path) -> Path:
    """A small textured box as GLB (what an image-to-3D add-on returns)."""
    from vrphoto.core.types import Mesh
    from vrphoto.geometry.gltf import write_glb

    v = np.array([[x, y, z] for x in (-0.2, 0.4) for y in (0.3, 0.8) for z in (-0.1, 0.1)], np.float32)
    f = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
         [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        np.uint32,
    )  # fmt: skip
    return write_glb([Mesh(v, f, np.zeros((8, 2), np.float32), np.full((4, 4, 3), 128, np.uint8))], path)


# --------------------------------------------------------------------------- F1 video depth
class NoisyStaticEstimator:
    """A relative depth model on a static scene: same depth + noise, normalised per frame."""

    id = "da2-small"

    def __init__(self, sigma: float = 0.03):
        self.rng = np.random.default_rng(0)
        self.sigma = sigma

    def estimate(self, img, *, progress=None, cancel=None):
        h, w = img.shape[:2]
        base = np.repeat(np.linspace(0, 1, h, dtype=np.float32)[:, None], w, 1)
        base[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 0.9
        return DepthResult(_normalised(base + self.rng.normal(0, self.sigma, (h, w))), "da2-small")


def test_video_depth_range_does_not_fade(make_job, tmp_path, fake_estimator, monkeypatch):
    from vrphoto.depth import post

    frames = [np.full((48, 64, 3), 120, np.uint8) for _ in range(120)]
    src = _write_video(tmp_path / "static.mp4", frames, 10.0)
    fake_estimator(NoisyStaticEstimator(0.03))
    ranges: list[float] = []
    real = post.temporal_align

    def spy(prev, cur, ema=0.6, **kw):
        out = real(prev, cur, ema=ema, **kw)
        lo, hi = np.percentile(out, [2, 98])
        ranges.append(float(hi - lo))
        return out

    monkeypatch.setattr(post, "temporal_align", spy)
    run_job(
        make_job(
            "video",
            [src],
            encoder_preset="h264_compat",
            keep_audio=False,
            common_extra={"export_scene": False, "depth_refine": 0.0},
        )
    )
    assert len(ranges) == 120
    # the old chained least-squares fit had lost ~40% of the range after 120 frames
    assert min(ranges[60:]) > 0.9, ranges[-5:]


class TwoShotEstimator:
    """Depth A for dark frames, depth B (unrelated) for bright frames."""

    id = "da2-small"

    def estimate(self, img, *, progress=None, cancel=None):
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        if float(img.mean()) < 128:
            d = yy / h
        else:
            d = 0.3 + 0.7 * (xx / w)
            d[: h // 3, : w // 3] = 1.0
        return DepthResult(_normalised(d.astype(np.float32)), "da2-small")


def test_video_scene_cut_restarts_depth_and_screen_smoothing(make_job, tmp_path, fake_estimator, monkeypatch):
    from vrphoto.depth import post
    from vrphoto.geometry import stereo

    frames = [np.full((48, 64, 3), 40, np.uint8)] * 8 + [np.full((48, 64, 3), 220, np.uint8)] * 6
    src = _write_video(tmp_path / "cut.mp4", frames, 10.0)
    fake_estimator(TwoShotEstimator())
    prevs: list[bool] = []
    real_align = post.temporal_align

    def spy_align(prev, cur, ema=0.6, **kw):
        prevs.append(prev is not None)
        return real_align(prev, cur, ema=ema, **kw)

    zcs: list[float] = []
    real_conv = stereo.auto_convergence

    def spy_conv(inv, *a, **k):
        zc = float(real_conv(inv, *a, **k))
        zcs.append(zc)
        return zc

    conv: list[float | None] = []
    real_pair = stereo.stereo_pair

    def spy_pair(li, params, *a, **k):
        conv.append(params.convergence_m)
        return real_pair(li, params, *a, **k)

    monkeypatch.setattr(post, "temporal_align", spy_align)
    monkeypatch.setattr(stereo, "auto_convergence", spy_conv)
    monkeypatch.setattr(stereo, "stereo_pair", spy_pair)
    run_job(
        make_job(
            "video",
            [src],
            encoder_preset="h264_compat",
            keep_audio=False,
            common_extra={"export_scene": False, "depth_refine": 0.0},
        )
    )
    assert len(prevs) == 14
    assert prevs[0] is False and all(prevs[1:8])
    assert prevs[8] is False, "the cut must restart the depth smoothing"
    assert all(prevs[9:])
    # the screen plane restarts at the new shot's own convergence distance
    assert abs(zcs[8] - zcs[7]) > 0.05 * zcs[7]  # (the two shots need different screen planes)
    assert conv[8] == pytest.approx(zcs[8], rel=1e-6)  # not blended with the previous shot's


# --------------------------------------------------------------------------- F4 NTSC rates
def test_ntsc_rate_is_decoded_and_encoded_exactly(make_job, video_path, monkeypatch):
    from vrphoto.video import decode, encode

    real_probe = decode.probe

    def probe_2398(path):
        info = real_probe(path)
        info.fps = 23.98  # what the ffmpeg banner prints for 24000/1001
        return info

    seen: dict[str, float] = {}
    real_iter = decode.iter_frames

    def spy_iter(path, **kw):
        seen["decode"] = kw.get("fps")
        return real_iter(path, **kw)

    real_writer = encode.VideoWriter

    class SpyWriter(real_writer):  # type: ignore[misc, valid-type]
        def __init__(self, path, w, h, fps, *a, **k):
            seen["encode"] = fps
            super().__init__(path, w, h, fps, *a, **k)

    monkeypatch.setattr(decode, "probe", probe_2398)
    monkeypatch.setattr(decode, "iter_frames", spy_iter)
    monkeypatch.setattr(encode, "VideoWriter", SpyWriter)
    run_job(
        make_job("video", [video_path], encoder_preset="h264_compat", common_extra={"export_scene": False})
    )
    exact = 24000 / 1001
    assert seen["decode"] == exact and seen["encode"] == exact
    assert Fraction(encode.fps_arg(seen["decode"])) == Fraction(24000, 1001)


# --------------------------------------------------------------------------- F5 / F7 screen
def test_screen_width_follows_the_picture_shape():
    assert R._screen_width_m(16 / 9) == 3.0
    assert R._screen_width_m(4 / 3) == 2.9333
    assert R._screen_width_m(0.75) == 1.65  # portrait: 2.2 m tall
    assert R._screen_width_m(9 / 16) == pytest.approx(1.2375)
    assert R._screen_width_m(float("nan")) == 3.0


@pytest.mark.parametrize("size", [(300, 400), (400, 300)])
def test_stereo_photo_screen_and_divergence_limit_agree(make_job, tmp_path, check_scene, monkeypatch, size):
    from vrphoto.geometry import stereo

    w, h = size
    src = _photo(tmp_path / "pic.jpg", w, h)
    seen = []
    real = stereo.stereo_pair

    def spy(li, params, *a, **k):
        seen.append(params)
        return real(li, params, *a, **k)

    monkeypatch.setattr(stereo, "stereo_pair", spy)
    m = check_scene(run_job(make_job("stereo", [src], look_behind="none")), "stereo")
    block = m["stereo"]
    expected = min(3.0, 2.2 * w / h)
    assert block["screen_width_m"] == pytest.approx(expected, abs=1e-3)
    assert block["screen_width_m"] / block["aspect"] <= 2.2 + 1e-6  # never taller than 2.2 m
    assert seen and seen[0].mode == "flat"
    assert seen[0].screen_width_m == block["screen_width_m"]


def test_video_and_parallax_blocks_carry_the_screen_width(make_job, gradient_photo, video_path, check_scene):
    m = check_scene(run_job(make_job("video", [video_path], encoder_preset="h264_compat")), "stereo")
    assert m["stereo"]["screen_width_m"] == pytest.approx(min(3.0, 2.2 * 128 / 96), abs=1e-3)
    for output, layout in (("sbs", "sbs"), ("flat", "mono")):
        res = run_job(
            make_job(
                "parallax", [gradient_photo], output=output, duration_s=0.2, fps=10,
                encoder_preset="h264_compat", look_behind="none",
            )
        )  # fmt: skip
        b = check_scene(res, "stereo")["stereo"]
        assert b["layout"] == layout
        assert b["screen_width_m"] == pytest.approx(min(3.0, 2.2 * b["aspect"]), abs=1e-3)


# --------------------------------------------------------------------------- E2E-1 / R3
def test_auto_video_format_depends_on_the_output(settings, gradient_photo, make_job):
    run = R._Run(make_job("parallax", [gradient_photo]), context.RunContext(settings), CancelToken())
    assert R._video_preset(run, "auto", output="flat") == "h264_compat"
    assert R._video_preset(run, "auto", output="sbs") == "quest3_hevc"
    assert R._video_preset(run, "auto", output="vr180") == "quest3_hevc"
    assert R._video_preset(run, "h264_compat", output="sbs") == "h264_compat"


def test_flat_living_photo_is_h264_and_at_most_1920_px(make_job, tmp_path):
    from vrphoto.video.decode import parse_ffmpeg_info, probe
    from vrphoto.video.encode import ffmpeg_exe

    src = _photo(tmp_path / "big.jpg", 2400, 1600)
    res = run_job(
        make_job(
            "parallax",
            [src],
            duration_s=0.2,
            fps=10,
            look_behind="none",
            common_extra={"export_scene": False},
        )
    )
    v = res.outputs["Living photo video"]
    info = probe(v)
    assert (info.width, info.height) == (1920, 1280)
    import subprocess

    text = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", v], capture_output=True, text=True, errors="replace"
    ).stderr
    assert parse_ffmpeg_info(text)["codec"] == "h264"


# --------------------------------------------------------------------------- R2 360 size
def test_360_photo_keeps_its_resolution_above_max_process_px(make_job, tmp_path):
    from PIL import Image

    h, w = 600, 1200
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx // 40 % 2) * 200 + 20, xx * 255 // w, yy * 255 // h], -1).astype(np.uint8)
    src = tmp_path / "pano.jpg"
    Image.fromarray(img).save(src, quality=92)
    res = run_job(
        make_job("pano", [src], stereo=False, common_extra={"max_process_px": 512, "export_scene": False})
    )
    with Image.open(res.outputs["360 photo"]) as im:
        assert im.size == (1200, 600)  # not shrunk to 'Max processing size' (512)


def test_normal_photo_in_pano_mode_still_uses_max_process_px(make_job, tmp_path, monkeypatch):
    from vrphoto.geometry import projection

    src = _photo(tmp_path / "pic.jpg", 1600, 1200)
    seen = []
    real = projection.pano_from_photo

    def spy(rgb, K, *a, **k):
        seen.append(rgb.shape)
        return real(rgb, K, *a, **k)

    monkeypatch.setattr(projection, "pano_from_photo", spy)
    run_job(make_job("pano", [src], width=1024, common_extra={"max_process_px": 512, "export_scene": False}))
    assert seen and max(seen[0][:2]) == 512


# --------------------------------------------------------------------------- F3 scene range
class ConstantEstimator:
    id = "da2-small"

    def __init__(self, d: float):
        self.d = d

    def estimate(self, img, *, progress=None, cancel=None):
        return DepthResult(np.full(img.shape[:2], self.d, np.float32), "da2-small")


@pytest.mark.parametrize("d", [0.05, 0.9])
def test_scene_look_behind_is_sized_for_real_head_movement(
    make_job, gradient_photo, fake_estimator, monkeypatch, d
):
    from vrphoto.geometry import layers

    fake_estimator(ConstantEstimator(d))
    seen = []
    real = layers.build_layers

    def spy(img, inv_z, K, *a, max_baseline_m, **k):
        seen.append(max_baseline_m)
        return real(img, inv_z, K, *a, max_baseline_m=max_baseline_m, **k)

    monkeypatch.setattr(layers, "build_layers", spy)
    run_job(
        make_job(
            "scene", [gradient_photo], target_faces=4000, anchor_distance_m=2.0, max_baseline_m=0.4,
            common_extra={"export_scene": False, "near_m": 1.2, "far_m": 15.0},
        )
    )  # fmt: skip
    z = 1.0 / (1 / 15.0 + d * (1 / 1.2 - 1 / 15.0))  # centre depth in photo units
    s = 2.0 / z  # the anchor scale layered_meshes applies
    assert seen[0] == pytest.approx(0.4 / s, rel=1e-3)


def test_scene_triangles_follow_the_detail_setting(make_job, gradient_photo):
    from vrphoto.geometry.gltf import read_glb_meshes

    res = run_job(
        make_job("scene", [gradient_photo], target_faces=20000, common_extra={"export_scene": False})
    )
    meshes = read_glb_meshes(res.outputs["3D scene (GLB)"])
    total = sum(len(m.faces) for m in meshes)
    assert total <= 1.15 * 20000, total


# --------------------------------------------------------------------------- F2 capture scale
def _fake_capture(calls: list, scale_info: dict | None):
    from vrphoto.geometry.splat import write_splats

    def fake(inputs, out_dir_, *, name=None, progress=None, cancel=None, subject_height_m=0.0, **kw):
        calls.append({"subject_height_m": subject_height_m, **kw})
        n = 300
        rng = np.random.default_rng(1)
        sc = SplatCloud(
            rng.normal(0, 0.5, (n, 3)).astype(np.float32) + np.array([0, 0.9, 0], np.float32),
            np.full((n, 3), 0.02, np.float32),
            np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
            rng.random((n, 3)).astype(np.float32),
            np.ones(n, np.float32),
        )
        p = write_splats(sc, Path(out_dir_) / f"{name}.splat")
        write_splats(sc, Path(out_dir_) / f"{name}.ply")
        if scale_info is not None:
            (Path(out_dir_) / f"{name}.capture.json").write_text(json.dumps(scale_info), encoding="utf-8")
        return p

    return fake


def _capture_folder(tmp_path: Path, gradient_photo: Path) -> Path:
    folder = tmp_path / "walk"
    folder.mkdir()
    for i in range(3):
        (folder / f"img{i}.jpg").write_bytes(gradient_photo.read_bytes())
    return folder


def test_capture_object_is_shown_at_its_measured_real_size(
    make_job, gradient_photo, tmp_path, monkeypatch, check_scene
):
    from vrphoto.addons import capture

    calls: list = []
    info = {"scale": 0.48, "height_m": 2.61, "method": "subject", "subject_height_m": 1.8}
    monkeypatch.setattr(capture, "capture_to_splat", _fake_capture(calls, info))
    res = run_job(make_job("capture", [_capture_folder(tmp_path, gradient_photo)], subject_height_m=1.8))
    assert calls[0]["subject_height_m"] == 1.8
    block = check_scene(res, "object")["object"]
    assert block["scale"] == 1.0 and block["height_m"] == 1.8
    assert not any("real-world size" in w for w in res.warnings)


def test_capture_camera_scale_uses_the_splat_height(
    make_job, gradient_photo, tmp_path, monkeypatch, check_scene
):
    from vrphoto.addons import capture

    calls: list = []
    monkeypatch.setattr(
        capture, "capture_to_splat", _fake_capture(calls, {"scale": 0.5, "height_m": 2.2, "method": "camera"})
    )
    res = run_job(make_job("capture", [_capture_folder(tmp_path, gradient_photo)]))
    assert calls[0]["subject_height_m"] == 0.0  # automatic
    block = check_scene(res, "object")["object"]
    assert block["scale"] == 1.0 and block["height_m"] == 2.2


@pytest.mark.parametrize("info", [None, {"scale": 1.0, "height_m": 3.0, "method": "none"}])
def test_capture_without_a_known_scale_warns(
    make_job, gradient_photo, tmp_path, monkeypatch, check_scene, info
):
    from vrphoto.addons import capture

    monkeypatch.setattr(capture, "capture_to_splat", _fake_capture([], info))
    res = run_job(make_job("capture", [_capture_folder(tmp_path, gradient_photo)]))
    block = check_scene(res, "object")["object"]
    assert "scale" not in block and "height_m" not in block
    (warning,) = [w for w in res.warnings if "real-world size" in w]
    assert "grip" in warning and "Subject height" in warning


def test_capture_scene_without_a_known_scale_warns_without_resize_advice(
    make_job, gradient_photo, tmp_path, monkeypatch, check_scene
):
    from vrphoto.addons import capture

    monkeypatch.setattr(capture, "capture_to_splat", _fake_capture([], None))
    folder = _capture_folder(tmp_path, gradient_photo)
    res = run_job(make_job("capture", [folder], as_object=False, subject_height_m=1.7))
    assert check_scene(res, "splat")["splat"]["scale"] == 1.0
    (warning,) = [w for w in res.warnings if "real-world size" in w]
    # the walk-in splat scene cannot be resized in the viewer; the height was already given
    assert "grip" not in warning and "Subject height" not in warning


def test_capture_subject_height_is_validated(make_job, gradient_photo, tmp_path):
    with pytest.raises(InputError, match="Subject height"):
        run_job(make_job("capture", [_capture_folder(tmp_path, gradient_photo)], subject_height_m=-1.0))


def _quat(Rm: np.ndarray) -> list[float]:
    """Rotation matrix -> COLMAP (w, x, y, z) quaternion (Shepperd's method)."""
    t = float(np.trace(Rm))
    if t > 0:
        w = np.sqrt(1 + t) / 2
        q = [
            w,
            (Rm[2, 1] - Rm[1, 2]) / (4 * w),
            (Rm[0, 2] - Rm[2, 0]) / (4 * w),
            (Rm[1, 0] - Rm[0, 1]) / (4 * w),
        ]
    else:
        i = int(np.argmax(np.diag(Rm)))
        j, k = (i + 1) % 3, (i + 2) % 3
        r = np.sqrt(1 + Rm[i, i] - Rm[j, j] - Rm[k, k])
        v = np.zeros(3)
        v[i], v[j], v[k] = r / 2, (Rm[j, i] + Rm[i, j]) / (2 * r), (Rm[k, i] + Rm[i, k]) / (2 * r)
        q = [(Rm[k, j] - Rm[j, k]) / (2 * r), *v]
    return [float(x) for x in q]


def _synthetic_walkaround(tmp_path: Path, monkeypatch, units_per_m: float = 2.2) -> None:
    """Fake COLMAP + Brush output of a 1.7 m person on a floor, photographed by 16 phones
    orbiting at 2 m and 1.2 / 1.7 m height, in COLMAP's arbitrary scale (``units_per_m``)."""
    fakes = Path(__file__).resolve().parents[1] / "addons" / "fakes"
    rng = np.random.default_rng(3)
    a, r = rng.uniform(0, 2 * np.pi, 4000), 2.5 * np.sqrt(rng.uniform(0, 1, 4000))
    floor = np.stack([r * np.cos(a), np.zeros(4000), r * np.sin(a)], 1)
    a2 = rng.uniform(0, 2 * np.pi, 3000)
    person = np.stack([0.2 * np.cos(a2), rng.uniform(0, 1.7, 3000), 0.2 * np.sin(a2)], 1)
    flip = np.diag([1.0, -1.0, -1.0])  # COLMAP: y down
    pts = (units_per_m * np.vstack([floor, person]) @ flip.T).astype(np.float32)
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]  # fmt: skip
    rec = np.zeros(len(pts), dtype=[(p, "<f4") for p in props])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["opacity"], rec["rot_0"] = 3.0, 1.0
    for k in ("scale_0", "scale_1", "scale_2"):
        rec[k] = np.log(0.01 * units_per_m)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
    header += "".join(f"property float {p}\n" for p in props) + "end_header\n"
    ply = tmp_path / "trained.ply"
    ply.write_bytes(header.encode("ascii") + rec.tobytes())
    poses = []
    for i in range(16):
        th = 2 * np.pi * i / 16
        c = np.array([2.0 * np.cos(th), 1.2 if i % 2 else 1.7, 2.0 * np.sin(th)])
        f, d = np.array([-np.cos(th), 0.0, -np.sin(th)]), np.array([0.0, -1.0, 0.0])
        R_cw = np.stack([np.cross(d, f), d, f]) @ flip.T  # camera rows: right, down, forward
        poses.append([*_quat(R_cw), *(-R_cw @ (units_per_m * flip @ c))])
    (tmp_path / "cams.json").write_text(json.dumps(poses), encoding="utf-8")
    monkeypatch.setenv("VRPHOTO_COLMAP", str(fakes / "fake_colmap.py"))
    monkeypatch.setenv("VRPHOTO_BRUSH", str(fakes / "fake_brush.py"))
    monkeypatch.setenv("FAKE_CAMS", str(tmp_path / "cams.json"))
    monkeypatch.setenv("FAKE_BRUSH_PLY", str(ply))
    for k in ("FAKE_REG", "FAKE_GLOBAL_REG", "FAKE_NO_GLOBAL", "FAKE_GLOBAL_FAIL", "FAKE_GPU_FAIL",
              "FAKE_LEGACY", "FAKE_BRUSH_SLEEP", "FAKE_BRUSH_FAIL", "FAKE_TOOL_LOG"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("subject_h", [0.0, 1.8])
def test_capture_scale_reaches_the_viewer_end_to_end(make_job, tmp_path, monkeypatch, check_scene, subject_h):
    """Real capture_to_splat (fake COLMAP / Brush executables) -> <stem>.capture.json -> scene."""
    pytest.importorskip("vrphoto.geometry.splat")
    from vrphoto.addons import capture
    from vrphoto.geometry.splat import read_splats

    monkeypatch.setattr(capture, "_has_nvidia_gpu", lambda: True)
    capture._info_cache.clear()
    _synthetic_walkaround(tmp_path, monkeypatch)
    folder = tmp_path / "walk"
    folder.mkdir()
    for i in range(8):
        _photo(folder / f"IMG_{i:03d}.jpg", 128, 96)
    res = run_job(make_job("capture", [folder], steps=200, subject_height_m=subject_h))
    assert not any("real-world size" in w for w in res.warnings), res.warnings
    block = check_scene(res, "object")["object"]
    assert block["scale"] == 1.0  # the splat is already metric: shown 1:1
    m = np.asarray(read_splats(res.outputs["Captured 3D scene (Gaussian splat)"]).means, dtype=np.float64)
    near = np.linalg.norm(m[:, [0, 2]], axis=1) < 0.3
    person_h = float(np.percentile(m[near, 1], 99.5) - m[:, 1].min())
    if subject_h:
        assert block["height_m"] == pytest.approx(subject_h)  # the label names the subject
        assert person_h == pytest.approx(subject_h, rel=0.05)
    else:
        assert person_h == pytest.approx(1.7, rel=0.05)  # from the 1.45 m median camera height
        assert block["height_m"] == pytest.approx(m[:, 1].max() - m[:, 1].min(), abs=0.01)


# --------------------------------------------------------------------------- R5 / R4 output folder
def lock_folder(monkeypatch, folder: Path) -> list[Path]:
    """Creating files in ``folder`` fails like in a read-only / 'Controlled folder access'
    folder (``PermissionError``, as on Windows also for a folder that looks writable).
    Returns the list of attempted paths."""
    attempts: list[Path] = []
    real = os.open

    def fake_open(path, flags, *a, **k):
        if Path(path).parent == folder:
            attempts.append(Path(path))
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real(path, flags, *a, **k)

    monkeypatch.setattr(R.os, "open", fake_open)
    return attempts


def test_unwritable_output_folder_fails_before_any_work(
    make_job, gradient_photo, out_dir, fake_estimator, monkeypatch
):
    est = fake_estimator(ConstantEstimator(0.5))
    calls: list[int] = []
    monkeypatch.setattr(est, "estimate", lambda *a, **k: calls.append(1))
    lock_folder(monkeypatch, out_dir)
    with pytest.raises(InputError) as ei:
        run_job(make_job("stereo", [gradient_photo]))
    assert str(out_dir) in str(ei.value)
    assert "Controlled folder access" in (ei.value.hint or "")
    assert not calls  # no depth estimation (or any other slow step) ran
    assert not list(out_dir.iterdir())


def test_write_check_tries_once_and_never_uses_tempfile(out_dir, monkeypatch):
    # tempfile on Windows takes "access denied" for a name clash and retries while the folder
    # looks writable (up to os.TMP_MAX = 2**31 times on Python 3.11 / older 3.12: a hang)
    import tempfile

    def no_tempfile(*a, **k):
        raise AssertionError("tempfile must not be used for the output folder check")

    for name in ("TemporaryFile", "NamedTemporaryFile", "mkstemp"):
        monkeypatch.setattr(tempfile, name, no_tempfile)
    attempts = lock_folder(monkeypatch, out_dir)
    with pytest.raises(InputError, match="Cannot save files into"):
        R._check_writable(out_dir)
    assert len(attempts) == 1


def test_write_test_file_is_removed(make_job, gradient_photo, out_dir):
    R._check_writable(out_dir)
    assert not list(out_dir.iterdir())
    run_job(make_job("depth", [gradient_photo], save_color=False))
    assert not [p for p in out_dir.iterdir() if p.name.startswith(R.WRITE_TEST_PREFIX)]


@pytest.mark.parametrize(
    "exc, words",
    [
        (OSError(errno.ENOSPC, "No space left on device", "x_LR.jpg"), "disk is full"),
        (PermissionError(errno.EACCES, "Permission denied", "x_LR.jpg"), "permission denied"),
    ],
)
def test_disk_full_and_permission_errors_are_friendly(make_job, gradient_photo, monkeypatch, exc, words):
    def fail(path, img, **k):
        raise exc

    monkeypatch.setattr(R, "_save_jpeg_or_png", fail)
    with pytest.raises(VRPhotoError) as ei:
        run_job(make_job("stereo", [gradient_photo], look_behind="none"))
    assert words in str(ei.value) and "x_LR.jpg" in str(ei.value)
    assert ei.value.hint


def test_queue_message_for_raw_os_errors():
    from vrphoto.pipeline.queue import error_message

    text = error_message(OSError(errno.ENOSPC, "No space left on device"))
    assert "disk is full" in text and "Unexpected" not in text
    assert errors.UI_LOG in error_message(ValueError("x"))


def test_failed_image_and_gif_writes_leave_no_file(tmp_path, monkeypatch):
    from vrphoto.core import imaging

    def full(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    img = np.zeros((40, 60, 3), np.uint8)
    monkeypatch.setattr(imaging, "atomic_write_bytes", full)
    with pytest.raises(OSError):
        R._save_jpeg_or_png(tmp_path / "a.jpg", img)
    monkeypatch.setattr(R, "atomic_write_bytes", full)
    with pytest.raises(OSError):
        R._save_gif(tmp_path / "w.gif", [img, img])
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- WIN-2 renames
def test_glb_rescale_uses_the_retrying_atomic_write(tmp_path, monkeypatch):
    src = _box_glb(tmp_path / "in.glb")
    calls = []
    real = R.atomic_write_bytes

    def spy(path, data, **k):
        calls.append(Path(path))
        return real(path, data, **k)

    monkeypatch.setattr(R, "atomic_write_bytes", spy)
    assert R.fit_glb_height(src, tmp_path / "out.glb", 1.7)
    assert calls == [tmp_path / "out.glb"]
    monkeypatch.setattr(
        R, "atomic_write_bytes", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full"))
    )
    with pytest.raises(OSError):  # a write failure is not silently "unfittable"
        R.fit_glb_height(src, tmp_path / "out2.glb", 1.7)


def test_move_file_renames_or_copies(tmp_path, monkeypatch):
    a = tmp_path / "a.splat"
    a.write_bytes(b"abc")
    R._move_file(a, tmp_path / "b.splat")
    assert (tmp_path / "b.splat").read_bytes() == b"abc" and not a.exists()

    # another drive: the rename fails with EXDEV -> copied through a temp file
    c = tmp_path / "c.splat"
    c.write_bytes(b"xyz")
    real = R.replace_with_retry

    def cross_device(src, dst, **k):
        if Path(src) == c:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real(src, dst, **k)

    monkeypatch.setattr(R, "replace_with_retry", cross_device)
    R._move_file(c, tmp_path / "d.splat")
    assert (tmp_path / "d.splat").read_bytes() == b"xyz" and not c.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b.splat", "d.splat"]


# --------------------------------------------------------------------------- E2E-5 library
@pytest.mark.skipif(not hasattr(os, "link"), reason="no hard links")
def test_library_shares_the_output_file(make_job, gradient_photo):
    res = run_job(make_job("stereo", [gradient_photo], look_behind="none"))
    out = Path(res.outputs["3D photo (side-by-side)"])
    scene = Path(res.outputs["VR viewer scene"])
    media = json.loads((scene / "scene.json").read_text(encoding="utf-8"))["stereo"]["media"]
    assert os.path.samefile(out, scene / media)  # one file on disk, not a second full copy


# --------------------------------------------------------------------------- E2E-3 / R9 hints
def test_pipeline_hints_name_existing_gui_places():
    import vrphoto

    pkg = Path(vrphoto.__file__).resolve().parent
    stale = re.compile(r"Settings > (Add-ons|Models)\b|Help > Open log folder|Video to 3D")
    files = [*(pkg / "pipeline").glob("*.py"), pkg / "cli.py", pkg / "selftest.py"]
    bad = [
        f"{f.name}:{n}"
        for f in files
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
        if stale.search(line)
    ]
    assert not bad


def test_video_given_to_a_photo_mode_names_the_video_mode(make_job, video_path):
    with pytest.raises(InputError) as ei:
        run_job(make_job("stereo", [video_path]))
    assert errors.MODE_VIDEO in (ei.value.hint or "")


def test_missing_addon_hint_names_the_addons_window(make_job, gradient_photo, monkeypatch):
    from vrphoto.addons import manager

    monkeypatch.setattr(manager, "status", lambda addon_id: {"installed": False})
    with pytest.raises(errors.AddonNotInstalled) as ei:
        run_job(make_job("object", [gradient_photo], method="trellis"))
    assert errors.UI_ADDONS in (ei.value.hint or "")


# --------------------------------------------------------------------------- add-on options
class _Manager:
    def __init__(self, glb: Path):
        self.calls: list[dict] = []
        self.glb = glb

    def status(self, addon_id):
        return {"installed": True}

    def run(self, addon_id, task, inputs, params, out_dir, *, progress=None, cancel=None, timeout=None):
        self.calls.append(dict(params))
        if addon_id == "sharp":
            from vrphoto.geometry.splat import write_splats

            n = 2000
            rng = np.random.default_rng(0)
            sc = SplatCloud(
                rng.normal(0, 0.3, (n, 3)).astype(np.float32) + np.array([0, 0, 2], np.float32),
                np.full((n, 3), 0.01, np.float32),
                np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
                rng.random((n, 3)).astype(np.float32),
                np.ones(n, np.float32),
            )
            return {"ply": str(write_splats(sc, Path(out_dir) / "model.ply"))}
        return {"glb": str(self.glb)}


@pytest.fixture
def addon_manager(monkeypatch, tmp_path):
    from vrphoto.addons import manager

    fm = _Manager(_box_glb(tmp_path / "model.glb"))
    monkeypatch.setattr(manager, "status", fm.status)
    monkeypatch.setattr(manager, "run", fm.run)
    return fm


@pytest.mark.parametrize("texture", [False, True])
def test_hunyuan_texture_option_reaches_the_worker(make_job, gradient_photo, addon_manager, texture):
    run_job(make_job("object", [gradient_photo], method="hunyuan", hunyuan_texture=texture))
    assert addon_manager.calls[-1]["texture"] is texture


def test_texture_flag_is_only_sent_to_hunyuan(make_job, gradient_photo, addon_manager):
    run_job(make_job("object", [gradient_photo], method="trellis", hunyuan_texture=True))
    assert "texture" not in addon_manager.calls[-1]


def test_sharp_says_that_variation_has_no_effect(make_job, gradient_photo, addon_manager):
    res = run_job(make_job("object", [gradient_photo], method="sharp", seed=5))
    assert "seed" not in addon_manager.calls[-1]
    assert any("Variation" in w for w in res.warnings)
    res = run_job(make_job("object", [gradient_photo], method="sharp"))
    assert not any("Variation" in w for w in res.warnings)
