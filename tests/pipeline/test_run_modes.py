"""End-to-end job dispatch for every mode on small synthetic inputs (no AI models)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pipeline_helpers import decode_frames, image_size

from vrphoto.pipeline import run as R
from vrphoto.pipeline.run import run_job


def _out(result, label: str) -> Path:
    matches = [Path(p) for k, p in result.outputs.items() if k == label or k.endswith(": " + label)]
    assert matches, f"{label!r} not in {list(result.outputs)}"
    p = matches[0]
    assert p.exists(), p
    return p


# --------------------------------------------------------------------------- depth
def test_depth_mode(make_job, gradient_photo, out_dir):
    import cv2

    res = run_job(make_job("depth", [gradient_photo]))
    p16 = _out(res, "Depth map (16-bit PNG)")
    pc = _out(res, "Depth map (colour preview)")
    assert p16.name == "photo_depth.png" and pc.name == "photo_depth_color.jpg"
    assert p16.parent == out_dir
    d = cv2.imdecode(np.fromfile(str(p16), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert d.dtype == np.uint16 and d.shape == (480, 640)
    assert d.max() > d.min()
    assert image_size(pc) == (640, 480)
    assert res.scene_id is None  # depth maps are not viewer scenes
    assert res.elapsed_s >= 0


def test_depth_mode_needs_an_output(make_job, gradient_photo):
    from vrphoto.core.errors import InputError

    with pytest.raises(InputError, match="Nothing to save"):
        run_job(make_job("depth", [gradient_photo], save_16bit=False, save_color=False))


# --------------------------------------------------------------------------- stereo
@pytest.mark.parametrize(
    ("layout", "suffix", "size", "scene"),
    [
        ("sbs", "_LR.jpg", (1280, 480), "sbs"),
        ("half_sbs", "_LR.jpg", (640, 480), "sbs"),
        ("tb", "_TB.jpg", (640, 960), "tb"),
        ("half_tb", "_TB.jpg", (640, 480), "tb"),
        ("anaglyph", "_anaglyph.jpg", (640, 480), None),
        ("cross", "_cross.jpg", (1280, 480), None),
    ],
)
def test_stereo_layouts(make_job, gradient_photo, check_scene, layout, suffix, size, scene):
    res = run_job(make_job("stereo", [gradient_photo], layout=layout))
    img = next(Path(p) for k, p in res.outputs.items() if k.startswith("3D photo"))
    assert img.name == "photo" + suffix
    assert image_size(img) == size
    if scene:
        m = check_scene(res, "stereo")
        assert m["stereo"]["layout"] == scene and m["stereo"]["kind"] == "image"
        assert m["stereo"]["aspect"] == pytest.approx(640 / 480, rel=1e-3)
        assert m["title"] == "photo" and m["source"] == "photo.jpg"
    else:
        assert res.scene_id is None


def test_stereo_eyes_differ_and_png_and_wiggle(make_job, gradient_photo):
    from PIL import Image

    from vrphoto.core.imaging import load_image

    res = run_job(make_job("stereo", [gradient_photo], image_format="png", wiggle_gif=True, strength=1.5))
    img = _out(res, "3D photo (side-by-side)")
    assert img.suffix == ".png"
    sbs = load_image(img)
    left, right = sbs[:, :640], sbs[:, 640:]
    assert np.abs(left.astype(int) - right.astype(int)).mean() > 0.5  # real parallax
    gif = _out(res, "Wiggle animation (GIF)")
    assert gif.name == "photo_wiggle.gif"
    with Image.open(gif) as g:
        assert g.n_frames == R.WIGGLE_FRAMES


def test_stereo_no_scene_when_disabled(make_job, gradient_photo, settings):
    res = run_job(make_job("stereo", [gradient_photo], common_extra={"export_scene": False}))
    assert res.scene_id is None
    assert not any("viewer" in k for k in res.outputs)
    assert not list(Path(settings.library_dir).iterdir())


def test_outputs_are_never_overwritten(make_job, gradient_photo):
    a = run_job(make_job("stereo", [gradient_photo]))
    b = run_job(make_job("stereo", [gradient_photo]))
    pa, pb = _out(a, "3D photo (side-by-side)"), _out(b, "3D photo (side-by-side)")
    assert pa.name == "photo_LR.jpg" and pb.name == "photo (2)_LR.jpg"
    assert a.scene_id != b.scene_id


def test_custom_name_and_default_output_dir(make_job, gradient_photo, settings):
    res = run_job(make_job("stereo", [gradient_photo], common_extra={"output_dir": None, "name": "My: trip"}))
    p = _out(res, "3D photo (side-by-side)")
    assert p.name == "My_ trip_LR.jpg"
    assert p.parent == Path(settings.output_dir)


# --------------------------------------------------------------------------- parallax
def test_parallax_flat(make_job, gradient_photo, check_scene):
    res = run_job(
        make_job("parallax", [gradient_photo], duration_s=0.5, fps=10, encoder_preset="h264_compat")
    )
    v = _out(res, "Living photo video")
    assert v.name == "photo_parallax.mp4"
    frames = decode_frames(v)
    assert len(frames) == 5
    assert frames[0].shape == (480, 640, 3)
    assert np.abs(frames[0].astype(int) - frames[2].astype(int)).mean() > 0.2  # the camera moves
    m = check_scene(res, "stereo")
    assert m["stereo"]["kind"] == "video" and m["stereo"]["layout"] == "mono"


def test_parallax_sbs_with_width(make_job, gradient_photo, check_scene):
    res = run_job(
        make_job(
            "parallax",
            [gradient_photo],
            duration_s=0.3,
            fps=10,
            output="sbs",
            width=320,
            path="swing",
            encoder_preset="h264_compat",
        )
    )
    v = _out(res, "Living photo video (3D side-by-side)")
    assert v.name == "photo_parallax_LR.mp4"
    f = decode_frames(v)
    assert len(f) == 3 and f[0].shape == (120, 320, 3)
    m = check_scene(res, "stereo")
    assert m["stereo"]["layout"] == "sbs" and m["stereo"]["aspect"] == pytest.approx(4 / 3, rel=0.02)


def test_parallax_vr180_metadata(make_job, gradient_photo, check_scene):
    from vrphoto.video.spatial import read

    res = run_job(
        make_job(
            "parallax",
            [gradient_photo],
            duration_s=0.2,
            fps=10,
            output="vr180",
            width=512,
            encoder_preset="h264_compat",
        )
    )
    v = _out(res, "Living photo video (VR180)")
    assert v.name == "photo_parallax_180_LR.mp4"
    f = decode_frames(v)
    assert f[0].shape == (256, 512, 3)
    meta = read(v)
    assert meta["spherical"] and meta["coverage"] == 180 and meta["stereo"] == "lr"
    m = check_scene(res, "sphere")
    assert m["sphere"] == {**m["sphere"], "coverage": 180, "layout": "sbs", "kind": "video"}


def test_parallax_custom_keyframes(make_job, gradient_photo):
    kf = [{"t": 0, "x": -0.03}, {"t": 0.5, "x": 0.03, "yaw": 2}, {"t": 1, "x": -0.03}]
    res = run_job(
        make_job(
            "parallax",
            [gradient_photo],
            duration_s=0.3,
            fps=10,
            path="custom",
            keyframes=kf,
            encoder_preset="h264_compat",
            common_extra={"export_scene": False},
        )
    )
    assert len(decode_frames(_out(res, "Living photo video"))) == 3


# --------------------------------------------------------------------------- vr180
def test_vr180_photo(make_job, gradient_photo, check_scene):
    res = run_job(make_job("vr180", [gradient_photo], eye_px=256))
    p = _out(res, "VR180 photo")
    assert p.name == "photo_180_LR.jpg"
    assert image_size(p) == (512, 256)
    data = p.read_bytes()
    assert b"GPano:ProjectionType" in data and b'FullPanoWidthPixels="512"' in data
    m = check_scene(res, "sphere")
    assert (
        m["sphere"]["coverage"] == 180 and m["sphere"]["layout"] == "sbs" and m["sphere"]["kind"] == "image"
    )


def test_vr180_default_eye_from_preset_and_still_video(make_job, gradient_photo, monkeypatch):
    from vrphoto.video.spatial import read

    monkeypatch.setitem(R.VR180_EYE_PX, "quest2", 192)
    res = run_job(
        make_job(
            "vr180",
            [gradient_photo],
            as_video_s=0.5,
            fill="black",
            common_extra={"quest_model": "quest2", "export_scene": False},
        )
    )
    assert image_size(_out(res, "VR180 photo")) == (384, 192)
    v = _out(res, "VR180 video (still)")
    assert v.name == "photo_180_LR.mp4"
    frames = decode_frames(v)
    assert len(frames) == round(0.5 * R.STILL_VIDEO_FPS)
    assert read(v)["coverage"] == 180


def test_vr180_black_fill_has_black_surroundings(make_job, gradient_photo):
    from vrphoto.core.imaging import load_image

    res = run_job(
        make_job(
            "vr180",
            [gradient_photo],
            eye_px=256,
            fill="black",
            image_format="png",
            common_extra={"export_scene": False},
        )
    )
    img = load_image(_out(res, "VR180 photo"))
    assert img[:8, :8].max() < 10  # corner of the 180 field is outside the photo


# --------------------------------------------------------------------------- pano
def test_pano_from_normal_photo(make_job, gradient_photo, check_scene):
    res = run_job(make_job("pano", [gradient_photo], width=512))
    p = _out(res, "360 photo")
    assert p.name == "photo_360.jpg"
    assert image_size(p) == (512, 256)
    assert b"GPano" in p.read_bytes()
    m = check_scene(res, "sphere")
    assert m["sphere"]["coverage"] == 360 and m["sphere"]["layout"] == "mono"


def test_pano_from_360_input_is_stereo_tb(make_job, equirect_path, check_scene):
    res = run_job(make_job("pano", [equirect_path], width=512))
    p = _out(res, "360 stereo photo (top-bottom)")
    assert p.name == "pano360_360_TB.jpg"
    assert image_size(p) == (512, 512)
    m = check_scene(res, "sphere")
    assert m["sphere"]["coverage"] == 360 and m["sphere"]["layout"] == "tb"


def test_pano_360_mono_with_depth_and_video(make_job, equirect_path, check_scene, monkeypatch):
    from vrphoto.video.spatial import read

    monkeypatch.setitem(R.PANO_WIDTH, "quest3", 384)
    res = run_job(make_job("pano", [equirect_path], stereo=False, as_video_s=0.3))
    p = _out(res, "360 photo")
    assert image_size(p) == (384, 192)
    v = _out(res, "360 video (still)")
    assert v.name == "pano360_360.mp4"
    meta = read(v)
    assert meta["coverage"] == 360 and meta["stereo"] == "mono"
    m = check_scene(res, "sphere")
    assert m["sphere"]["depth"] == "depth.png"
    assert m["sphere"]["depth_near_m"] > 0 and m["sphere"]["depth_far_m"] > m["sphere"]["depth_near_m"]


def test_pano_force_360_on_wrong_aspect_fails(make_job, gradient_photo):
    from vrphoto.core.errors import InputError

    with pytest.raises(InputError, match="not a 360 photo"):
        run_job(make_job("pano", [gradient_photo], input_is_360=True))


# --------------------------------------------------------------------------- scene
def test_scene_photo3d(make_job, gradient_photo, check_scene):
    import pygltflib

    res = run_job(
        make_job("scene", [gradient_photo], target_faces=20_000, export_obj=True, max_baseline_m=0.3)
    )
    glb = _out(res, "3D scene (GLB)")
    assert glb.name == "photo_3d.glb"
    g = pygltflib.GLTF2().load(str(glb))
    assert len(g.meshes) >= 2  # backdrop/background + foreground
    obj = _out(res, "3D scene (OBJ)")
    assert obj.with_suffix(".mtl").is_file()
    m = check_scene(res, "photo3d")
    assert m["photo3d"]["meshes"] == ["scene.glb"]
    assert m["photo3d"]["anchor_distance_m"] == 2.0 and m["photo3d"]["eye_height_m"] == R.EYE_HEIGHT_M


def test_scene_glb_only_in_library_when_not_exported(make_job, gradient_photo, out_dir, check_scene):
    res = run_job(make_job("scene", [gradient_photo], target_faces=5000, export_glb=False))
    assert not list(out_dir.glob("*.glb"))
    check_scene(res, "photo3d")


def test_scene_cleanplate_with_segmentation(make_job, gradient_photo, check_scene):
    res = run_job(make_job("scene", [gradient_photo], target_faces=5000, look_behind="cleanplate"))
    check_scene(res, "photo3d")


# --------------------------------------------------------------------------- object
def test_object_billboard(make_job, gradient_photo, check_scene):
    from vrphoto.geometry.gltf import read_glb_meshes
    from vrphoto.geometry.splat import read_splats

    res = run_job(make_job("object", [gradient_photo], height_m=1.6))
    glb = _out(res, "3D object (GLB)")
    assert glb.name == "photo_object.glb"
    meshes = read_glb_meshes(glb)
    ys = np.concatenate([m.vertices[:, 1] for m in meshes])
    assert ys.min() == pytest.approx(0.0, abs=0.02) and ys.max() == pytest.approx(1.6, rel=0.05)
    sp = _out(res, "3D object (Gaussian splat)")
    assert sp.suffix == ".splat" and len(read_splats(sp)) > 100
    m = check_scene(res, "object")
    assert m["object"]["format"] == "glb" and m["object"]["alt_assets"] == ["object.splat"]
    assert m["object"]["height_m"] == 1.6 and m["object"]["pivot"] == "base"
    assert m["env"]["background"] == "passthrough"


def test_object_without_splat(make_job, gradient_photo, check_scene):
    res = run_job(
        make_job("object", [gradient_photo], export_splat=False, back="mirror", placement="front-floor")
    )
    m = check_scene(res, "object")
    assert "alt_assets" not in m["object"] and m["object"]["placement"] == "front-floor"


def test_object_empty_matte_is_friendly(make_job, gradient_photo, settings, monkeypatch):
    from vrphoto.core.errors import InputError
    from vrphoto.pipeline import context

    class NoSubject:
        id = "grabcut"

        def segment(self, img, **kw):
            return np.zeros(img.shape[:2], np.float32)

    monkeypatch.setattr(context.RunContext, "segmenter", lambda self, *a, **k: NoSubject())
    with pytest.raises(InputError, match="No person or object"):
        run_job(make_job("object", [gradient_photo]))


# --------------------------------------------------------------------------- video
def test_video_sbs_keeps_audio(make_job, video_path, check_scene):
    from vrphoto.video.decode import probe

    res = run_job(make_job("video", [video_path], encoder_preset="h264_compat"))
    v = _out(res, "3D video (side-by-side)")
    assert v.name == "clip_LR.mp4"
    info = probe(v)
    assert (info.width, info.height) == (256, 96)
    assert 9 <= info.n_frames <= 11
    assert info.has_audio
    m = check_scene(res, "stereo")
    assert m["stereo"]["kind"] == "video" and m["stereo"]["layout"] == "sbs"
    assert m["stereo"]["aspect"] == pytest.approx(128 / 96, rel=1e-3)


def test_video_tb_without_audio_and_time_range(make_job, video_path):
    from vrphoto.video.decode import probe

    res = run_job(
        make_job(
            "video",
            [video_path],
            output="tb",
            keep_audio=False,
            start_s=0.2,
            end_s=0.7,
            encoder_preset="h264_compat",
            common_extra={"export_scene": False},
        )
    )
    v = _out(res, "3D video (top-bottom)")
    assert v.name == "clip_TB.mp4"
    info = probe(v)
    assert (info.width, info.height) == (128, 192)
    assert not info.has_audio
    assert 4 <= info.n_frames <= 6


def test_video_vr180(make_job, video_path, check_scene):
    from vrphoto.video.spatial import read

    res = run_job(make_job("video", [video_path], output="vr180", end_s=0.3, encoder_preset="h264_compat"))
    v = _out(res, "VR180 video")
    assert v.name == "clip_180_LR.mp4"
    f = decode_frames(v)
    assert f[0].shape[1] == 2 * f[0].shape[0]
    assert read(v)["coverage"] == 180
    check_scene(res, "sphere")


def test_video_mode_rejects_photos_and_photo_modes_reject_videos(make_job, gradient_photo, video_path):
    from vrphoto.core.errors import InputError

    with pytest.raises(InputError, match="not a video"):
        run_job(make_job("video", [gradient_photo]))
    with pytest.raises(InputError, match="is a video"):
        run_job(make_job("stereo", [video_path]))


def test_video_empty_time_range(make_job, video_path):
    from vrphoto.core.errors import InputError

    with pytest.raises(InputError):
        run_job(make_job("video", [video_path], start_s=5.0, end_s=6.0, encoder_preset="h264_compat"))
