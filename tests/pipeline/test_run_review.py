"""Regression tests for the pipeline review fixes (object cut-outs, SHARP subject filter,
warnings from video writers / add-ons, constant-rate video decoding, capture logs, stale work
folders, early validation)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from vrphoto.core.errors import AddonError, InputError, VRPhotoError
from vrphoto.core.types import SplatCloud
from vrphoto.pipeline import context
from vrphoto.pipeline import run as R
from vrphoto.pipeline.run import run_job


class FakeManager:
    def __init__(self, installed, outputs):
        self.installed = set(installed)
        self.outputs = outputs
        self.calls: list[dict] = []

    def status(self, addon_id):
        return {"installed": addon_id in self.installed}

    def run(self, addon_id, task, inputs, params, out_dir, *, progress=None, cancel=None, timeout=None):
        self.calls.append(
            {"addon_id": addon_id, "task": task, "inputs": dict(inputs), "params": dict(params)}
        )
        return self.outputs(addon_id, task, inputs, params, Path(out_dir))


@pytest.fixture
def fake_manager(monkeypatch):
    from vrphoto.addons import manager

    def install(installed, outputs):
        fm = FakeManager(installed, outputs)
        monkeypatch.setattr(manager, "status", fm.status)
        monkeypatch.setattr(manager, "run", fm.run)
        return fm

    return install


def _box_alpha(h: int = 480, w: int = 640) -> np.ndarray:
    a = np.zeros((h, w), np.float32)
    a[300:460, 250:390] = 1.0  # the red "subject" of the gradient_photo fixture
    return a


@pytest.fixture
def box_segmenter(monkeypatch):
    """Segmenter returning exactly the subject box of ``gradient_photo``."""

    class BoxSeg:
        id = "birefnet-lite"
        calls = 0

        def segment(self, img, *, rect=None, progress=None, cancel=None):
            BoxSeg.calls += 1
            h, w = img.shape[:2]
            return _box_alpha(h, w)

    seg = BoxSeg()
    monkeypatch.setattr(context.RunContext, "segmenter", lambda self, *a, **k: seg)
    return seg


# --------------------------------------------------------------------------- object cut-outs
def _rgba_png(path: Path, h: int = 300, w: int = 240) -> Path:
    from PIL import Image

    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = (20, 20, 20)
    rgba[40:280, 70:170, :3] = (200, 150, 90)
    rgba[40:280, 70:170, 3] = 255
    Image.fromarray(rgba, "RGBA").save(path)
    return path


def test_object_uses_the_photos_own_transparency(make_job, tmp_path, monkeypatch, check_scene):
    def no_segmenter(*a, **k):
        raise AssertionError("a cut-out PNG must not be segmented again")

    monkeypatch.setattr(context.RunContext, "segmenter", no_segmenter)
    png = _rgba_png(tmp_path / "cutout person.png")
    res = run_job(make_job("object", [png], height_m=1.6))
    glb = Path(res.outputs["3D object (GLB)"])
    assert glb.is_file()
    m = check_scene(res, "object")
    assert m["object"]["height_m"] == 1.6


def test_opaque_alpha_is_ignored_and_segmenter_runs(make_job, tmp_path, box_segmenter):
    from PIL import Image

    rgba = np.full((480, 640, 4), 255, np.uint8)  # alpha channel present but fully opaque
    rgba[300:460, 250:390, :3] = (200, 60, 60)
    p = tmp_path / "opaque.png"
    Image.fromarray(rgba, "RGBA").save(p)
    res = run_job(make_job("object", [p], common_extra={"export_scene": False}))
    assert box_segmenter.calls == 1
    assert Path(res.outputs["3D object (GLB)"]).is_file()


def test_billboard_splats_are_capped(make_job, gradient_photo, monkeypatch, box_segmenter):
    from vrphoto.geometry.splat import read_splats

    monkeypatch.setattr(R, "MAX_OBJECT_SPLATS", 2000)
    res = run_job(make_job("object", [gradient_photo], common_extra={"export_scene": False}))
    sc = read_splats(res.outputs["3D object (Gaussian splat)"])
    assert 0 < len(sc) <= 2000


def test_object_back_option_is_validated_before_work(make_job, gradient_photo, monkeypatch):
    def no_segmenter(*a, **k):
        raise AssertionError("validation must happen first")

    monkeypatch.setattr(context.RunContext, "segmenter", no_segmenter)
    with pytest.raises(InputError, match="back texture"):
        run_job(make_job("object", [gradient_photo], back="shiny"))


# --------------------------------------------------------------------------- SHARP subject filter
def _sharp_cloud(f: float = 500.0, w: int = 640, h: int = 480, z: float = 2.0) -> SplatCloud:
    """Red splats that project onto the subject box, blue ones onto the background."""
    rng = np.random.default_rng(5)
    n = 1500
    us = np.concatenate([rng.uniform(262, 378, n), rng.uniform(20, 620, n)])
    vs = np.concatenate([rng.uniform(312, 448, n), rng.uniform(10, 200, n)])
    zs = np.full(2 * n, z) + rng.normal(0, 0.01, 2 * n)
    means = np.stack([(us - w / 2) * zs / f, (vs - h / 2) * zs / f, zs], 1).astype(np.float32)
    colors = np.concatenate([np.tile([1.0, 0.0, 0.0], (n, 1)), np.tile([0.0, 0.0, 1.0], (n, 1))]).astype(
        np.float32
    )
    return SplatCloud(
        means,
        np.full((2 * n, 3), 0.005, np.float32),
        np.tile([1, 0, 0, 0], (2 * n, 1)).astype(np.float32),
        colors,
        np.full(2 * n, 0.9, np.float32),
    )


def test_sharp_keeps_only_the_subject(make_job, gradient_photo, fake_manager, box_segmenter):
    from vrphoto.geometry.splat import read_splats, write_splats

    def outputs(addon_id, task, inputs, params, out_dir):
        p = write_splats(_sharp_cloud(), out_dir / "sharp.ply")
        return {"ply": str(p), "frame": "opencv", "f_px": 500.0, "width": 640, "height": 480}

    fake_manager({"sharp"}, outputs)
    res = run_job(make_job("object", [gradient_photo], method="sharp", common_extra={"export_scene": False}))
    sc = read_splats(next(p for k, p in res.outputs.items() if k.startswith("3D object (Gaussian splat")))
    red = int((sc.colors[:, 0] > 0.5).sum())
    blue = int((sc.colors[:, 2] > 0.5).sum())
    assert red > 1300  # (crop_splats trims a few percent of outliers)
    assert blue < 100
    assert not any("whole scene" in w for w in res.warnings)


def test_sharp_filter_uses_matte_resolution_when_sizes_differ():
    """The add-on may report its own image size: projections are rescaled to the matte."""
    run = type("RunStub", (), {"warnings": [], "warn": lambda self, m: self.warnings.append(m)})()
    alpha = _box_alpha(240, 320)
    alpha[:] = 0.0
    alpha[150:230, 125:195] = 1.0  # the subject box at half the add-on's resolution
    out = R._subject_splats(run, _sharp_cloud(), alpha, {"f_px": 500.0, "width": 640, "height": 480}, {})
    assert int((out.colors[:, 0] > 0.5).sum()) > 1400
    assert int((out.colors[:, 2] > 0.5).sum()) < 50
    assert not run.warnings


def test_sharp_filter_drops_background_behind_the_subject():
    run = type("RunStub", (), {"warnings": [], "warn": lambda self, m: self.warnings.append(m)})()
    front = _sharp_cloud(z=2.0)
    back = _sharp_cloud(z=6.0)  # same pixels, far behind (e.g. the wall behind the person)
    back.colors[:] = (0.0, 1.0, 0.0)
    both = R._concat_splats([front, back])
    out = R._subject_splats(run, both, _box_alpha(), {"f_px": 500.0, "width": 640, "height": 480}, {})
    assert int((out.colors[:, 0] > 0.5).sum()) > 1400
    assert int((out.colors[:, 1] > 0.5).sum()) == 0


def test_sharp_filter_without_camera_info_uses_the_job_fov():
    run = type("RunStub", (), {"warnings": [], "warn": lambda self, m: self.warnings.append(m)})()
    hfov = float(np.degrees(2 * np.arctan(320 / 500.0)))  # == f 500 px at 640 px width
    out = R._subject_splats(run, _sharp_cloud(), _box_alpha(), {}, {"hfov_deg": hfov})
    assert int((out.colors[:, 2] > 0.5).sum()) < 50
    behind = _sharp_cloud()
    behind.means[:, 2] *= -1.0  # everything behind the camera: nothing to keep -> fallback
    same = R._subject_splats(run, behind, _box_alpha(), {"f_px": 500.0}, {})
    assert same is behind and any("whole scene" in w for w in run.warnings)


def test_add_on_warnings_reach_the_result(make_job, gradient_photo, fake_manager, box_segmenter):
    from vrphoto.core.types import Mesh
    from vrphoto.geometry.gltf import write_glb

    def outputs(addon_id, task, inputs, params, out_dir):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        m = Mesh(
            v, np.array([[0, 1, 2]], np.uint32), np.zeros((3, 2), np.float32), np.zeros((2, 2, 3), np.uint8)
        )
        glb = write_glb([m], out_dir / "m.glb")
        return {"glb": str(glb), "warning": "Texture baking failed; the model is untextured."}

    fake_manager({"hunyuan"}, outputs)
    res = run_job(
        make_job("object", [gradient_photo], method="hunyuan", common_extra={"export_scene": False})
    )
    assert any("Texture baking failed" in w and "Hunyuan" in w for w in res.warnings)


# --------------------------------------------------------------------------- video
def test_video_writer_warnings_reach_the_result(make_job, gradient_photo, monkeypatch):
    from vrphoto.video import encode

    real = encode.VideoWriter

    class WarningWriter(real):  # type: ignore[misc, valid-type]
        def close(self):
            out = super().close()
            self.warnings.append("The sound could not be added (test).")
            return out

    monkeypatch.setattr(encode, "VideoWriter", WarningWriter)
    res = run_job(
        make_job("parallax", [gradient_photo], duration_s=0.2, fps=10, common_extra={"export_scene": False})
    )
    assert "The sound could not be added (test)." in res.warnings


def test_video_is_decoded_at_a_constant_frame_rate(make_job, video_path, monkeypatch):
    from vrphoto.video import decode

    seen = {}
    real = decode.iter_frames

    def spy(path, **kw):
        seen.update(kw)
        return real(path, **kw)

    monkeypatch.setattr(decode, "iter_frames", spy)
    run_job(
        make_job("video", [video_path], encoder_preset="h264_compat", common_extra={"export_scene": False})
    )
    assert seen.get("fps") == pytest.approx(10.0, rel=1e-3)


def test_video_half_tb(make_job, video_path):
    from vrphoto.video.decode import probe

    res = run_job(
        make_job(
            "video",
            [video_path],
            output="half_tb",
            encoder_preset="h264_compat",
            common_extra={"export_scene": False},
        )
    )
    v = Path(res.outputs["3D video (half top-bottom)"])
    assert v.name == "clip_TB.mp4"
    info = probe(v)
    assert (info.width, info.height) == (128, 96)


def test_parallax_bad_custom_path_fails_before_depth(make_job, gradient_photo, monkeypatch):
    def no_depth(*a, **k):
        raise AssertionError("the camera path must be validated first")

    monkeypatch.setattr(context.RunContext, "estimator", no_depth)
    with pytest.raises(InputError, match="camera path"):
        run_job(make_job("parallax", [gradient_photo], path="spiral"))


# --------------------------------------------------------------------------- pano
def test_pano_mono_depth_range_matches_the_depth_scale(make_job, equirect_path, check_scene):
    res = run_job(make_job("pano", [equirect_path], stereo=False, common_extra={"far_m": 60.0}))
    m = check_scene(res, "sphere")
    assert m["sphere"]["depth_far_m"] == 60.0
    assert m["sphere"]["depth_near_m"] == pytest.approx(1.2)


# --------------------------------------------------------------------------- capture
def test_failed_capture_keeps_its_log(make_job, gradient_photo, tmp_path, monkeypatch, out_dir):
    from vrphoto.addons import capture

    folder = tmp_path / "walk"
    folder.mkdir()
    for i in range(3):
        (folder / f"i{i}.jpg").write_bytes(gradient_photo.read_bytes())

    def failing_capture(inputs, out_dir_, **kw):
        (Path(out_dir_) / "capture_capture.log").write_text(
            "colmap: only 2 images registered\n", encoding="utf-8"
        )
        raise AddonError("COLMAP could not match enough photos.", hint="Take more overlapping photos.")

    monkeypatch.setattr(capture, "capture_to_splat", failing_capture)
    with pytest.raises(AddonError) as ei:
        run_job(make_job("capture", [folder]))
    hint = ei.value.hint or ""
    assert "Take more overlapping photos." in hint and "log was saved" in hint
    logs = list(out_dir.glob("*_capture_log.txt"))
    assert len(logs) == 1 and "only 2 images" in logs[0].read_text(encoding="utf-8")
    assert str(logs[0]) in hint


# --------------------------------------------------------------------------- batches / work folders
class OddError(VRPhotoError):
    def __init__(self, code: int) -> None:  # a subclass with another constructor
        super().__init__(f"odd failure {code}")
        self.code = code


def test_all_failed_batch_with_unusual_error_class(make_job, gradient_photo, photo_png, monkeypatch):
    def boom(run, item, rep):
        raise OddError(7)

    monkeypatch.setitem(R._HANDLERS, "stereo", boom)
    with pytest.raises(VRPhotoError, match="None of the 2 files could be processed.*odd failure 7"):
        run_job(make_job("stereo", [gradient_photo, photo_png]))


def test_stale_work_folders_are_swept(tmp_path, monkeypatch):
    root = tmp_path / "work"
    old = root / "job-old"
    new = root / "job-new"
    other = root / "keep-me"
    for d in (old, new, other):
        d.mkdir(parents=True)
        (d / "f.bin").write_bytes(b"x")
    past = time.time() - R.STALE_WORK_S - 3600
    os.utime(old, (past, past))
    os.utime(other, (past, past))
    monkeypatch.setattr(R, "_swept", set())
    assert R._sweep_stale_work(root) == 1
    assert not old.exists() and new.exists() and other.exists()
    assert R._sweep_stale_work(root) == 0  # once per process


def test_work_folder_creation_sweeps(make_job, gradient_photo, settings, monkeypatch):
    root = Path(settings.cache_dir) / "work"
    stale = root / "job-stale"
    stale.mkdir(parents=True)
    past = time.time() - R.STALE_WORK_S - 10
    os.utime(stale, (past, past))
    monkeypatch.setattr(R, "_swept", set())
    spec = make_job("stereo", [gradient_photo])
    run = R._Run(spec, context.RunContext(settings), R.CancelToken())
    item = R._Item(0, gradient_photo, "photo")
    work = run.work_dir(item)
    assert work.is_dir() and not stale.exists()
    R._cleanup(item)


def test_high_frame_rate_video_is_converted_at_60_fps(make_job, tmp_path):
    from vrphoto.video.decode import probe
    from vrphoto.video.encode import VideoWriter

    src = tmp_path / "slowmo.mp4"
    with VideoWriter(src, 96, 64, 120.0, "h264_compat") as vw:
        for i in range(60):  # 0.5 s at 120 fps
            f = np.zeros((64, 96, 3), np.uint8)
            f[..., 1] = 100
            f[20:44, i % 70 : i % 70 + 20] = (230, 40, 40)
            vw.write(f)
    res = run_job(
        make_job("video", [src], encoder_preset="h264_compat", common_extra={"export_scene": False})
    )
    info = probe(res.outputs["3D video (side-by-side)"])
    assert info.fps == pytest.approx(60.0, rel=1e-3)
    assert 28 <= info.n_frames <= 32
    assert info.duration == pytest.approx(0.5, abs=0.06)
