"""Add-on driven modes: object (TRELLIS/TRELLIS.2/Hunyuan/SHARP), AI outpainting, capture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vrphoto.core.errors import AddonError, AddonNotInstalled
from vrphoto.core.types import Mesh, SplatCloud
from vrphoto.pipeline import run as R
from vrphoto.pipeline.run import fit_glb_height, glb_bounds, run_job


def _box_mesh(lo=(-0.2, 0.3, -0.1), hi=(0.4, 0.8, 0.1)) -> Mesh:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        np.float32,
    )
    f = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3],
         [1, 2, 6], [1, 6, 5]],
        np.uint32,
    )  # fmt: skip
    uv = np.zeros((8, 2), np.float32)
    tex = np.full((4, 4, 3), 128, np.uint8)
    return Mesh(v, f, uv, tex, name="box")


def _write_box_glb(path: Path, **kw) -> Path:
    from vrphoto.geometry.gltf import write_glb

    return write_glb([_box_mesh(**kw)], path)


class FakeManager:
    """Stands in for vrphoto.addons.manager.status/run."""

    def __init__(self, installed: set[str], outputs=None):
        self.installed = installed
        self.calls: list[dict] = []
        self.outputs = outputs  # callable(addon_id, task, inputs, params, out_dir) -> dict

    def status(self, addon_id):
        return {"installed": addon_id in self.installed, "version": "1", "path": "", "size_mb": 0.0}

    def run(self, addon_id, task, inputs, params, out_dir, *, progress=None, cancel=None, timeout=None):
        self.calls.append(
            dict(
                addon_id=addon_id, task=task, inputs=dict(inputs), params=dict(params), out_dir=Path(out_dir)
            )
        )
        if progress is not None:
            progress(0.5, "fake add-on working")
        return self.outputs(addon_id, task, inputs, params, Path(out_dir))


@pytest.fixture
def fake_manager(monkeypatch):
    from vrphoto.addons import manager

    def install(installed=(), outputs=None) -> FakeManager:
        fm = FakeManager(set(installed), outputs)
        monkeypatch.setattr(manager, "status", fm.status)
        monkeypatch.setattr(manager, "run", fm.run)
        return fm

    return install


# --------------------------------------------------------------------------- object add-ons
@pytest.mark.parametrize("method", ["trellis", "trellis2", "hunyuan", "sharp"])
def test_object_addon_not_installed_fails_fast(make_job, gradient_photo, fake_manager, monkeypatch, method):
    from vrphoto.pipeline import context

    fake_manager(installed=())

    def no_segmentation(*a, **k):
        raise AssertionError("segmentation must not run when the add-on is missing")

    monkeypatch.setattr(context.RunContext, "segmenter", no_segmentation)
    with pytest.raises(AddonNotInstalled) as ei:
        run_job(make_job("object", [gradient_photo], method=method))
    assert "Add-ons" in (ei.value.hint or "")
    assert "not installed" in str(ei.value)


def test_object_addon_not_installed_real_manager(make_job, gradient_photo):
    """Without monkeypatching: a fresh data folder has no add-ons installed."""
    with pytest.raises(AddonNotInstalled, match="TRELLIS"):
        run_job(make_job("object", [gradient_photo], method="trellis"))


@pytest.mark.parametrize("method", ["trellis", "trellis2", "hunyuan"])
def test_object_image_to_3d_addon(make_job, gradient_photo, fake_manager, check_scene, method):
    from PIL import Image

    def outputs(addon_id, task, inputs, params, out_dir):
        glb = _write_box_glb(out_dir / "model.glb")
        res = {"glb": str(glb)}
        if addon_id == "trellis":
            ply = out_dir / "model.ply"
            ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
            res["ply"] = str(ply)
        return res

    fm = fake_manager(installed={method}, outputs=outputs)
    res = run_job(make_job("object", [gradient_photo], method=method, seed=7, height_m=1.5))
    assert len(fm.calls) == 1
    call = fm.calls[0]
    assert call["addon_id"] == method and call["task"] == "image_to_3d"
    assert call["params"]["seed"] == 7
    img = Path(call["inputs"]["image"])
    assert img.is_file() and img.suffix == ".png"
    with Image.open(img) as im:
        assert im.mode == "RGBA"
        a = np.asarray(im)[..., 3]
    assert a.min() == 0 and a.max() == 255  # transparent background, opaque subject
    cut = Path(res.outputs["Cut-out (PNG with transparency)"])
    assert cut.name == "photo_cutout.png"
    glb = next(Path(p) for k, p in res.outputs.items() if k.startswith("3D object (GLB"))
    assert glb.name == f"photo_{method}.glb"
    doc = _glb_json(glb)
    lo, hi = glb_bounds(doc)
    assert hi[1] - lo[1] == pytest.approx(1.5, rel=1e-4)
    assert lo[1] == pytest.approx(0.0, abs=1e-5)
    assert (lo[0] + hi[0]) / 2 == pytest.approx(0.0, abs=1e-5)
    m = check_scene(res, "object")
    assert m["object"]["format"] == "glb" and m["object"]["height_m"] == 1.5
    if method in ("trellis", "trellis2"):  # non-commercial add-ons carry a notice
        assert any("non-commercial" in n for n in m["notices"])
        assert any("non-commercial" in w for w in res.warnings)
    if method == "hunyuan":
        assert any("license is not valid" in n for n in m.get("notices", []))
    if method == "trellis":
        assert any(k.startswith("3D Gaussians") for k in res.outputs)


def test_object_sharp_splat(make_job, gradient_photo, fake_manager, check_scene):
    from vrphoto.geometry.splat import read_splats, write_splats

    def outputs(addon_id, task, inputs, params, out_dir):
        assert Path(inputs["image"]).is_file()  # the full photo (SHARP models the whole scene)
        n = 2000
        rng = np.random.default_rng(0)
        means = rng.normal(0, 0.3, (n, 3)).astype(np.float32)
        means[:, 2] += 2.0  # in front of the camera (OpenCV frame, z forward)
        sc = SplatCloud(
            means,
            np.full((n, 3), 0.01, np.float32),
            np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
            rng.random((n, 3)).astype(np.float32),
            np.full(n, 0.9, np.float32),
        )
        p = write_splats(sc, out_dir / "sharp.ply")
        return {"ply": str(p), "frame": "opencv", "f_px": 500.0}

    fm = fake_manager(installed={"sharp"}, outputs=outputs)
    res = run_job(make_job("object", [gradient_photo], method="sharp"))
    call = fm.calls[0]
    assert call["addon_id"] == "sharp" and call["task"] == "image_to_splat"
    assert "hfov_deg" in call["params"]
    sp = next(Path(p) for k, p in res.outputs.items() if k.startswith("3D object (Gaussian splat"))
    assert sp.name == "photo_sharp.splat"
    sc = read_splats(sp)
    assert 1000 < len(sc) <= 2000
    # the random cloud is not on the subject's cut-out: the whole reconstruction is kept
    assert any("whole scene was kept" in w for w in res.warnings)
    m = check_scene(res, "object")
    assert m["object"]["format"] == "splat" and m["object"]["asset"] == "object.splat"
    assert any("non-commercial" in n for n in m["notices"])


def test_object_addon_without_model_output_fails(make_job, gradient_photo, fake_manager):
    fake_manager(installed={"trellis"}, outputs=lambda *a: {"preview": "x.png"})
    with pytest.raises(AddonError, match="did not return a 3D model"):
        run_job(make_job("object", [gradient_photo], method="trellis"))


def test_object_addon_unfittable_glb_is_copied(make_job, gradient_photo, fake_manager):
    def outputs(addon_id, task, inputs, params, out_dir):
        p = out_dir / "model.glb"
        p.write_bytes(b"not really a glb")
        return {"glb": str(p)}

    fake_manager(installed={"trellis"}, outputs=outputs)
    res = run_job(
        make_job("object", [gradient_photo], method="trellis", common_extra={"export_scene": False})
    )
    glb = next(Path(p) for k, p in res.outputs.items() if k.startswith("3D object (GLB"))
    assert glb.read_bytes() == b"not really a glb"


def test_addon_work_folder_is_cleaned_up(make_job, gradient_photo, fake_manager, settings):
    seen = {}

    def outputs(addon_id, task, inputs, params, out_dir):
        seen["dir"] = out_dir
        return {"glb": str(_write_box_glb(out_dir / "m.glb"))}

    fake_manager(installed={"hunyuan"}, outputs=outputs)
    run_job(make_job("object", [gradient_photo], method="hunyuan"))
    assert Path(settings.cache_dir) in seen["dir"].parents
    assert not seen["dir"].exists()


# --------------------------------------------------------------------------- outpainting
def _green_outpaint(addon_id, task, inputs, params, out_dir):
    from PIL import Image

    with Image.open(inputs["image"]) as im:
        w, h = im.size
    with Image.open(inputs["mask"]) as mk:
        assert mk.size == (w, h)
        mask = np.asarray(mk.convert("L")) > 127
    assert mask.any() and not mask.all()
    p = out_dir / "outpainted.png"
    Image.new("RGB", (w, h), (0, 200, 0)).save(p)
    return {"image": str(p)}


def test_vr180_outpaint_with_addon(make_job, gradient_photo, fake_manager):
    from vrphoto.core.imaging import load_image

    fm = fake_manager(installed={"sdxl360"}, outputs=_green_outpaint)
    res = run_job(
        make_job(
            "vr180",
            [gradient_photo],
            eye_px=256,
            fill="outpaint",
            image_format="png",
            common_extra={"export_scene": False},
        )
    )
    call = fm.calls[0]
    assert (call["addon_id"], call["task"]) == ("sdxl360", "outpaint_360")
    assert set(call["inputs"]) == {"image", "mask"} and call["params"]["wrap"] == "no"
    img = load_image(res.outputs["VR180 photo"])
    corner = img[:10, :10].reshape(-1, 3).mean(0)
    assert corner[1] > 150 and corner[0] < 60  # surroundings come from the (green) outpainting
    assert not res.warnings


def test_vr180_outpaint_without_addon_falls_back_to_blur(make_job, gradient_photo, fake_manager):
    fm = fake_manager(installed=())
    res = run_job(
        make_job("vr180", [gradient_photo], eye_px=256, fill="outpaint", common_extra={"export_scene": False})
    )
    assert not fm.calls
    assert Path(res.outputs["VR180 photo"]).is_file()
    assert any("AI outpainting" in w and "not installed" in w for w in res.warnings)


def test_pano_outpaint_with_addon(make_job, gradient_photo, fake_manager):
    from vrphoto.core.imaging import load_image

    fm = fake_manager(installed={"sdxl360"}, outputs=_green_outpaint)
    res = run_job(
        make_job("pano", [gradient_photo], width=512, fill="outpaint", common_extra={"export_scene": False})
    )
    assert fm.calls[0]["params"]["wrap"] == "yes"
    img = load_image(res.outputs["360 photo"])
    assert img[5:20, 5:20].reshape(-1, 3).mean(0)[1] > 150


def test_outpaint_addon_failure_falls_back(make_job, gradient_photo, fake_manager):
    def broken(*a):
        raise AddonError("CUDA out of memory", hint="Close other apps")

    fake_manager(installed={"sdxl360"}, outputs=broken)
    res = run_job(
        make_job("pano", [gradient_photo], width=512, fill="outpaint", common_extra={"export_scene": False})
    )
    assert Path(res.outputs["360 photo"]).is_file()
    assert any("AI outpainting failed" in w for w in res.warnings)


# --------------------------------------------------------------------------- capture
def test_capture_not_installed(make_job, gradient_photo, tmp_path):
    photos = []
    for i in range(3):
        p = tmp_path / f"cap{i}.jpg"
        p.write_bytes(gradient_photo.read_bytes())
        photos.append(p)
    with pytest.raises(AddonNotInstalled) as ei:
        run_job(make_job("capture", photos))
    assert ei.value.hint


@pytest.mark.parametrize("as_object", [True, False])
def test_capture_with_fake_reconstruction(
    make_job, gradient_photo, tmp_path, monkeypatch, check_scene, out_dir, as_object
):
    from vrphoto.addons import capture
    from vrphoto.geometry.splat import write_splats

    folder = tmp_path / "Garden walk"
    folder.mkdir()
    for i in range(3):
        (folder / f"img{i}.jpg").write_bytes(gradient_photo.read_bytes())
    calls = []

    def fake_capture(
        inputs,
        out_dir_,
        *,
        video_fps=2.0,
        max_images=200,
        steps=15000,
        max_splats=750_000,
        progress=None,
        cancel=None,
        crop_pct=98.0,
        name=None,
        **kw,
    ):
        calls.append(
            dict(
                inputs=list(inputs),
                out_dir=Path(out_dir_),
                video_fps=video_fps,
                max_images=max_images,
                steps=steps,
                max_splats=max_splats,
                crop_pct=crop_pct,
                name=name,
            )
        )
        n = 500
        rng = np.random.default_rng(1)
        sc = SplatCloud(
            rng.normal(0, 1, (n, 3)).astype(np.float32),
            np.full((n, 3), 0.02, np.float32),
            np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
            rng.random((n, 3)).astype(np.float32),
            np.ones(n, np.float32),
        )
        p = write_splats(sc, Path(out_dir_) / f"{name}.splat")
        write_splats(sc, Path(out_dir_) / f"{name}.ply")
        if progress is not None:
            progress(1.0, "trained")
        return p

    monkeypatch.setattr(capture, "capture_to_splat", fake_capture)
    res = run_job(
        make_job(
            "capture", [folder], steps=500, max_images=50, video_fps=3.0, crop_pct=95.0, as_object=as_object
        )
    )
    c = calls[0]
    assert c["inputs"] == [folder] and c["steps"] == 500 and c["max_images"] == 50
    assert c["video_fps"] == 3.0 and c["crop_pct"] == 95.0
    splat = Path(res.outputs["Captured 3D scene (Gaussian splat)"])
    assert splat.parent == out_dir and splat.name == "Garden walk_capture.splat"
    assert Path(res.outputs["Captured 3D scene (PLY)"]).name == "Garden walk_capture.ply"
    m = check_scene(res, "object" if as_object else "splat")
    block = m["object" if as_object else "splat"]
    assert block["format"] == "splat"
    assert m.get("thumbnail") == "thumb.jpg"


# --------------------------------------------------------------------------- GLB fitting
def _glb_json(path: Path) -> dict:
    import struct

    data = path.read_bytes()
    jlen = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20 : 20 + jlen])


def test_fit_glb_height_roundtrip_with_trimesh(tmp_path):
    import trimesh

    src = _write_box_glb(tmp_path / "in.glb", lo=(1.0, 2.0, 3.0), hi=(1.5, 2.25, 3.2))
    dst = tmp_path / "out.glb"
    assert fit_glb_height(src, dst, 1.8)
    scene = trimesh.load(dst, force="scene")
    lo, hi = scene.bounds
    assert hi[1] - lo[1] == pytest.approx(1.8, rel=1e-4)
    assert lo[1] == pytest.approx(0.0, abs=1e-5)
    assert (lo[0] + hi[0]) / 2 == pytest.approx(0.0, abs=1e-5)
    assert (lo[2] + hi[2]) / 2 == pytest.approx(0.0, abs=1e-5)


def test_fit_glb_height_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"\x00" * 40)
    assert not fit_glb_height(bad, tmp_path / "o.glb", 1.0)
    assert not (tmp_path / "o.glb").exists()
    assert not fit_glb_height(tmp_path / "missing.glb", tmp_path / "o.glb", 1.0)


def test_glb_bounds_with_node_transforms():
    doc = {
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"translation": [0, 1, 0], "children": [1]},
            {"mesh": 0, "scale": [2, 2, 2], "rotation": [0, 0, 0.7071068, 0.7071068]},  # 90 deg about z
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"min": [0, 0, 0], "max": [1, 0.5, 0.25]}],
    }
    lo, hi = glb_bounds(doc)
    # x in [0,1] -> rotated to y in [0,2] (scaled), y in [0,0.5] -> x in [-1, 0]
    np.testing.assert_allclose(lo, [-1.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(hi, [0.0, 3.0, 0.5], atol=1e-5)
    doc["accessors"][0].pop("min")
    assert glb_bounds(doc) is None


def test_splats_cv_to_gltf_matches_rotation():
    from vrphoto.geometry.splat import quat_to_rotmat

    rng = np.random.default_rng(3)
    q = rng.normal(size=(5, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sc = SplatCloud(
        rng.normal(size=(5, 3)).astype(np.float32),
        np.ones((5, 3), np.float32),
        q,
        np.zeros((5, 3), np.float32),
        np.ones(5, np.float32),
    )
    out = R._splats_cv_to_gltf(sc)
    F = np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(out.means, sc.means * [1, -1, -1], atol=1e-6)
    np.testing.assert_allclose(quat_to_rotmat(out.quats), F @ quat_to_rotmat(q) @ F, atol=1e-5)
