"""SHARP 360 jobs end to end (docs/design/SHARP360.md section 5.3; the TRELLIS back has its
own file, test_sharp360_trellis.py).

A fake SHARP add-on returns a SHARP-like cloud of the analytic synthetic person
(tests/geometry/synth_person.py); the photo is that person's render and the segmenter
returns its matte, so the front clean-up, the built-in back, the stitching, the life-size
export and the previews all run for real on consistent data.  The "your own back photo"
tests write orthographic renders of the same person from behind / the side as the user's
photos (on a grey background, which the fake segmenter keys out).
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vrphoto.core.errors import AddonError, AddonNotInstalled, Cancelled, InputError
from vrphoto.core.types import CancelToken
from vrphoto.geometry.back_mould import BackPart
from vrphoto.pipeline import context
from vrphoto.pipeline import run as R
from vrphoto.pipeline import sharp360 as S
from vrphoto.pipeline.run import run_job

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geometry"))
import synth_person as sp  # noqa: E402

#: Portrait photo, big enough that the 1.5 cm voxel floater filter sees a dense back.
W, H, HFOV, G = 240, 320, 40.0, 128
CUT_FEET = (("cam_height", 1.4),)  # a high camera: the legs leave the photo at the bottom
#: Inputs and output of the 0.1.1 plain ``sharp`` method (commit 76cebb9), see
#: test_plain_sharp_matches_the_0_1_1_output.
REF = Path(__file__).parent / "data" / "plain_sharp_ref.npz"


@functools.lru_cache(maxsize=4)
def _scene(body_kw: tuple = ()) -> sp.SharpScene:
    body = sp.make_body(**dict(body_kw))
    return sp.sharp_like_scene(body, sp.make_cam(W, H, HFOV, grid=G), G=G)


def _ref_scene() -> SimpleNamespace:
    """The inputs pinned in data/plain_sharp_ref.npz (a small render of the synthetic person
    and SHARP-like cloud) as a ``person(scene=...)`` scene."""
    from vrphoto.core.types import SplatCloud
    from vrphoto.geometry.splat_fusion import SharpCam

    ref = np.load(REF)
    sc = SplatCloud(ref["in_means"], ref["in_scales"], ref["in_quats"], ref["in_colors"], ref["in_opacity"])
    cam = SharpCam(f=float(ref["f_px"]), width=int(ref["width"]), height=int(ref["height"]))
    return SimpleNamespace(sc=sc, alpha=ref["alpha"], rgb=ref["rgb"], cam=cam, ref=ref)


def _opaque_means(path: Path) -> np.ndarray:
    from vrphoto.geometry.splat import read_splats

    sc = read_splats(path)
    return np.asarray(sc.means, np.float64)[np.asarray(sc.opacity) > 0.5]


def _robust(path: Path) -> tuple[float, float]:
    """(floor, height) of a written splat as the life-size export measures it."""
    from vrphoto.geometry.splat import read_splats

    sc = read_splats(path)
    y = np.asarray(sc.means)[:, 1][np.asarray(sc.opacity) > 0.5]
    return float(np.percentile(y, 0.5)), float(np.percentile(y, 99.8))


class FakeManager:
    """Stands in for vrphoto.addons.manager.status/run; ``handlers`` maps add-on id ->
    ``fn(task, inputs, params, out_dir) -> outputs``."""

    def __init__(self, installed, handlers, events):
        self.installed = set(installed)
        self.handlers = handlers
        self.events = events
        self.calls: list[dict] = []

    def status(self, addon_id):
        return {"installed": addon_id in self.installed, "version": "1", "path": "", "size_mb": 0.0}

    def run(self, addon_id, task, inputs, params, out_dir, *, progress=None, cancel=None, timeout=None):
        self.events.append(f"run:{addon_id}")
        self.calls.append(dict(addon_id=addon_id, task=task, inputs=dict(inputs), params=dict(params)))
        if progress is not None:
            progress(0.5, "fake add-on working")
        return self.handlers[addon_id](task, inputs, params, Path(out_dir))


def _sharp_handler(scene: sp.SharpScene):
    def handle(task, inputs, params, out_dir):
        from vrphoto.geometry.splat import write_splats

        assert task == "image_to_splat" and Path(inputs["image"]).is_file()
        p = write_splats(scene.sc, out_dir / "model.ply")
        cam = scene.cam
        return {"ply": str(p), "frame": "opencv", "f_px": cam.f, "width": cam.width, "height": cam.height}

    return handle


@pytest.fixture
def events():
    return []


@pytest.fixture
def person(tmp_path, monkeypatch, events):
    """``person(body_kw=(), installed={"sharp"}, sharp=None, scene=None) -> (photo path,
    FakeManager)``: the synthetic photo, a segmenter returning its matte and a fake SHARP
    (``scene``: anything with ``sc``, ``alpha``, ``rgb`` and ``cam`` instead of the person)."""
    from PIL import Image

    from vrphoto.addons import manager

    def make(body_kw=(), installed=("sharp",), sharp=None, scene=None):
        scene = scene or _scene(tuple(body_kw))
        path = tmp_path / "photo.png"
        Image.fromarray(scene.rgb).save(path)

        class Seg:
            id = "birefnet-lite"

            def segment(self, img, *, rect=None, progress=None, cancel=None):
                events.append("segment")
                if img.shape[:2] != scene.alpha.shape:  # a back / side photo on its grey background
                    return (np.abs(img.astype(np.int16) - 128).max(axis=2) > 15).astype(np.float32)
                return scene.alpha.copy()

        seg = Seg()
        monkeypatch.setattr(context.RunContext, "segmenter", lambda self, *a, **k: seg)
        fm = FakeManager(installed, {"sharp": sharp or _sharp_handler(scene)}, events)
        monkeypatch.setattr(manager, "status", fm.status)
        monkeypatch.setattr(manager, "run", fm.run)
        return path, fm

    return make


def _splat_output(res, prefix="3D object (Gaussian splat") -> tuple[str, Path]:
    key, p = next((k, Path(v)) for k, v in res.outputs.items() if k.startswith(prefix))
    return key, p


# --------------------------------------------------------------------------- the job
def test_sharp360_writes_a_life_size_splat(make_job, person, check_scene):
    photo, fm = person()
    res = run_job(make_job("object", [photo], method="sharp360", height_m=1.6))
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]
    call = fm.calls[0]
    assert call["task"] == "image_to_splat"
    assert set(call["params"]) == {"hfov_deg", "name"}  # the same request as plain SHARP
    assert Path(call["inputs"]["image"]).name == "sharp_input.png"
    key, splat = _splat_output(res)
    assert splat.name == "photo_sharp360.splat" and splat.is_file()
    assert "SHARP 360°" in key and "built-in back" in key
    assert Path(res.outputs["Cut-out (PNG with transparency)"]).name == "photo_cutout.png"
    m = check_scene(res, "object")
    block = m["object"]
    assert block["asset"] == "object.splat" and block["format"] == "splat"
    assert block["scale"] == 1.0 and block["pivot"] == "base"
    assert block["height_m"] == pytest.approx(1.6, rel=1e-3)
    floor, top = _robust(splat)
    assert abs(floor) < 0.02
    assert top == pytest.approx(1.6, rel=0.01)
    # SHARP's research-only licence travels with the result
    assert any("non-commercial" in n for n in m["notices"])
    assert any("non-commercial" in w for w in res.warnings)
    # nothing fell back
    assert not [w for w in res.warnings if "non-commercial" not in w], res.warnings


def test_sharp360_uses_variation(make_job, person):
    """'Variation' drives the AI backs and the seam dither (0.2.0): no "not used" warning."""
    photo, _ = person()
    res = run_job(
        make_job("object", [photo], method="sharp360", seed=5, common_extra={"export_scene": False})
    )
    assert not any("Variation" in w for w in res.warnings), res.warnings


def test_sharp360_save_extras(make_job, person):
    from vrphoto.geometry.splat import read_splats

    photo, _ = person()
    res = run_job(
        make_job("object", [photo], method="sharp360", save_extras=True, common_extra={"export_scene": False})
    )
    names = {Path(p).name for p in res.outputs.values()}
    for n in (
        "photo_sharp360.splat",
        "photo_sharp360.ply",
        "photo_sharp360_front.splat",
        "photo_sharp360_back.splat",
        "photo_sharp360_debug.json",
    ):
        assert n in names, sorted(names)
    out = {Path(p).name: Path(p) for p in res.outputs.values()}
    main = read_splats(out["photo_sharp360.splat"])
    assert len(read_splats(out["photo_sharp360.ply"])) == len(main)
    front = read_splats(out["photo_sharp360_front.splat"])
    back = read_splats(out["photo_sharp360_back.splat"])
    assert len(front) > 500 and len(back) > 500
    assert len(front) + len(back) >= len(main)
    # same frame as the main splat: the photographed front faces +z, the back is behind it
    assert np.median(front.means[:, 2]) > np.median(back.means[:, 2]) + 0.05
    dbg = json.loads(out["photo_sharp360_debug.json"].read_text(encoding="utf-8"))
    assert dbg["options"]["sources"] == ["shell"]
    assert dbg["sources"] == [dict(dbg["sources"][0], source="shell", ok=True)]
    assert dbg["camera"]["grid"] == G
    assert dbg["canon"]["up_source"] == "floor"
    assert dbg["export"]["size_used"] == "height"
    assert dbg["back"]["source"] == "shell" and dbg["merge"]["gen_kept"] > 0
    assert dbg["seconds"]["total"] > 0
    # the floater filter's voxels follow the back's sampling
    spacing = dbg["back"]["diag"]["spacing_m"]
    assert dbg["floaters"]["voxel_m"] == pytest.approx(max(0.015, 1.5 * spacing))


def test_sharp360_fast_quality_keeps_its_coarse_back(make_job, person):
    """'fast' builds the back every 2 mould cells (here ~1.6 cm apart, coarser than the 1.5 cm
    floater voxels): the floater filter must keep it."""
    photo, _ = person()
    res = run_job(
        make_job(
            "object",
            [photo],
            method="sharp360",
            back_quality="fast",
            save_extras=True,
            common_extra={"export_scene": False},
        )
    )
    out = {Path(p).name: Path(p) for p in res.outputs.values()}
    dbg = json.loads(out["photo_sharp360_debug.json"].read_text(encoding="utf-8"))
    assert dbg["back"]["diag"]["stride"] == 2 and dbg["back"]["diag"]["spacing_m"] > 0.015
    assert dbg["floaters"]["voxel_m"] > 0.02
    assert dbg["floaters"]["gen_dropped"] <= 0.02 * dbg["merge"]["gen_kept"], dbg["floaters"]


def _band_centroid(P: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mean x / z of the points whose height lies between the ``lo`` and ``hi`` percentiles."""
    y0, y1 = np.percentile(P[:, 1], [lo, hi])
    band = P[(P[:, 1] >= y0) & (P[:, 1] <= y1)]
    return band[:, [0, 2]].mean(axis=0)


def test_sharp360_exports_a_pitched_photo_upright(make_job, person):
    """A photo taken looking down 15 deg (design 5.6 (f)): the floor gives "up", and the
    exported person stands upright (head straight above the feet), life-size, facing +z."""
    photo, _ = person(body_kw=(("pitch_deg", 15.0),))
    res = run_job(
        make_job(
            "object",
            [photo],
            method="sharp360",
            height_m=1.6,
            save_extras=True,
            common_extra={"export_scene": False},
        )
    )
    out = {Path(p).name: Path(p) for p in res.outputs.values()}
    dbg = json.loads(out["photo_sharp360_debug.json"].read_text(encoding="utf-8"))
    assert dbg["canon"]["up_source"] == "floor"
    assert dbg["canon"]["pitch_deg"] == pytest.approx(15.0, abs=1.5)
    P = _opaque_means(out["photo_sharp360.splat"])
    lean = _band_centroid(P, 80, 95) - _band_centroid(P, 5, 20)  # chest/arm/head minus legs
    assert abs(lean[1]) < 0.06, lean  # z: not leaning toward or away from the camera
    # x: the raised arm moves the upper band sideways; it must do so as in a level photo
    level, _ = person()
    res0 = run_job(
        make_job("object", [level], method="sharp360", height_m=1.6, common_extra={"export_scene": False})
    )
    P0 = _opaque_means(_splat_output(res0)[1])
    lean0 = _band_centroid(P0, 80, 95) - _band_centroid(P0, 5, 20)
    assert abs(lean[0] - lean0[0]) < 0.04, (lean, lean0)
    # the photographed front faces +z: it is in front of the made-up back
    front, back = (
        _opaque_means(out["photo_sharp360_front.splat"]),
        _opaque_means(out["photo_sharp360_back.splat"]),
    )
    assert np.median(front[:, 2]) > np.median(back[:, 2]) + 0.05
    # and the shape is the body's: its width / height as the true body's (raised arm included)
    truth = sp.synth_canon(sp.make_body(pitch_deg=15.0), sp.make_cam(W, H, HFOV, grid=G))
    x0, x1 = np.percentile(P[:, 0], [0.5, 99.5])
    y0, y1 = np.percentile(P[:, 1], [0.5, 99.8])
    assert (x1 - x0) / (y1 - y0) == pytest.approx(truth.width_s / truth.height_s, rel=0.08)


@pytest.mark.parametrize("size_from", ["height", "photo"])
def test_sharp360_size_from(make_job, person, check_scene, size_from):
    photo, _ = person()
    res = run_job(make_job("object", [photo], method="sharp360", height_m=1.5, size_from=size_from))
    block = check_scene(res, "object")["object"]
    _, top = _robust(_splat_output(res)[1])
    assert block["scale"] == 1.0
    assert top == pytest.approx(block["height_m"], rel=0.01)
    if size_from == "height":
        assert block["height_m"] == pytest.approx(1.5, rel=1e-3)
    else:  # SHARP's metric size: the synthetic person is 1.69 m tall
        assert block["height_m"] == pytest.approx(sp.BODY_HEIGHT, rel=0.04)
    assert not any("cut off" in w for w in res.warnings)


def test_sharp360_auto_size_of_a_cut_off_subject(make_job, person, check_scene):
    photo, _ = person(body_kw=CUT_FEET)
    res = run_job(make_job("object", [photo], method="sharp360", height_m=1.9))
    block = check_scene(res, "object")["object"]
    assert any("cut off at the bottom" in w and "Size from" in w for w in res.warnings), res.warnings
    # SHARP's metric size of what is in the photo (the legs are partly outside), not 1.9 m
    assert 1.2 < block["height_m"] < 1.7
    res = run_job(make_job("object", [photo], method="sharp360", height_m=1.9, size_from="height"))
    assert check_scene(res, "object")["object"]["height_m"] == pytest.approx(1.9, rel=1e-3)
    assert not any("cut off" in w for w in res.warnings)


def test_sharp360_front_is_untouched_with_the_built_in_back(make_job, person, monkeypatch):
    """Design 5.2 "front untouched": seen from the photo's camera, the fused result (front +
    built-in back, stitched, colours feathered, floaters removed) is SHARP's cleaned front."""
    from pipeline_helpers import capture_stitch, front_render_difference, generated_on_the_front

    seen = capture_stitch(monkeypatch)
    photo, _ = person()
    run_job(make_job("object", [photo], method="sharp360", common_extra={"export_scene": False}))
    assert seen["part"].source == "shell" and int((~seen["is_front"]).sum()) > 1000
    diff, n = front_render_difference(seen)
    assert n > 5000 and diff < 2.0, diff
    share, n_gen = generated_on_the_front(seen)  # nothing generated lies on the photographed side
    assert n_gen > 1000 and share < 0.02, share


def test_plain_sharp_is_unchanged(make_job, person, check_scene):
    """Regression: the plain ``sharp`` method gets exactly the old request and scene block."""
    photo, fm = person()
    res = run_job(make_job("object", [photo], method="sharp", height_m=1.6))
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]
    assert set(fm.calls[0]["params"]) == {"hfov_deg", "name"}
    assert Path(fm.calls[0]["inputs"]["image"]).name == "sharp_input.png"
    key, splat = _splat_output(res)
    assert splat.name == "photo_sharp.splat" and "SHARP 360" not in key
    block = check_scene(res, "object")["object"]
    assert "scale" not in block and block["height_m"] == 1.6


def test_plain_sharp_matches_the_0_1_1_output(make_job, person):
    """The plain ``sharp`` method's splat is the one 0.1.1 wrote for the same SHARP output and
    cut-out (design 0.2 #1).  The reference was made once by running commit 76cebb9 on the
    inputs stored next to it, so a change in any helper of that path (subject filter, glTF
    conversion, crop, orientation, decimation) shows up here.  Tolerances only absorb
    floating-point noise between platforms (.splat stores colours, opacity and rotations in
    8 bits)."""
    from vrphoto.geometry.splat import read_splats

    scene = _ref_scene()
    ref = scene.ref
    photo, fm = person(scene=scene)
    res = run_job(
        make_job("object", [photo], method="sharp", height_m=1.6, common_extra={"export_scene": False})
    )
    assert fm.calls[0]["params"]["hfov_deg"] == pytest.approx(51.077)
    out = read_splats(_splat_output(res)[1])
    assert len(out) == len(ref["out_means"])
    np.testing.assert_allclose(out.means, ref["out_means"], rtol=0, atol=2e-6)
    np.testing.assert_allclose(out.scales, ref["out_scales"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(out.colors, ref["out_colors"], rtol=0, atol=1.01 / 255)
    np.testing.assert_allclose(out.opacity, ref["out_opacity"], rtol=0, atol=1.01 / 255)
    np.testing.assert_allclose(out.quats, ref["out_quats"], rtol=0, atol=1.01 / 128)


# --------------------------------------------------------------------------- failures
def test_sharp360_sharp_failure_fails_the_job(make_job, person):
    def broken(task, inputs, params, out_dir):
        raise AddonError("SHARP crashed (test)")

    photo, _ = person(sharp=broken)
    with pytest.raises(AddonError, match="SHARP crashed"):
        run_job(make_job("object", [photo], method="sharp360"))
    photo, _ = person(sharp=lambda *a: {"frame": "opencv"})
    with pytest.raises(AddonError, match="did not return a 3D model"):
        run_job(make_job("object", [photo], method="sharp360"))

    def empty(task, inputs, params, out_dir):
        from vrphoto.geometry.splat import subset_splats, write_splats

        return {"ply": str(write_splats(subset_splats(_scene().sc, []), out_dir / "model.ply"))}

    photo, _ = person(sharp=empty)
    with pytest.raises(AddonError, match="empty 3D result"):
        run_job(make_job("object", [photo], method="sharp360"))


def test_sharp360_cancel_during_the_back_propagates(make_job, person, monkeypatch):
    warned: list[str] = []
    real_warn = R._Run.warn
    monkeypatch.setattr(R._Run, "warn", lambda self, m: (warned.append(m), real_warn(self, m)))
    token = CancelToken()
    real = S.SOURCES["shell"]

    def cancel_then_build(ctx, rep):
        token.cancel()  # the user presses Cancel while the back is being built
        return real(ctx, rep)

    monkeypatch.setitem(S.SOURCES, "shell", cancel_then_build)
    photo, _ = person()
    with pytest.raises(Cancelled):
        run_job(make_job("object", [photo], method="sharp360"), cancel=token)
    assert not [w for w in warned if "instead" in w], warned


def test_sharp360_stitch_failure_saves_the_front(make_job, person, monkeypatch, check_scene):
    def broken(*a, **k):
        raise RuntimeError("merge bug (test)")

    monkeypatch.setattr(S, "merge_parts", broken)
    photo, _ = person()
    res = run_job(make_job("object", [photo], method="sharp360", height_m=1.6))
    assert any("Stitching the front and back failed" in w for w in res.warnings), res.warnings
    key, splat = _splat_output(res)
    assert "front only" in key and splat.name == "photo_sharp360.splat"
    block = check_scene(res, "object")["object"]
    assert block["scale"] == 1.0 and block["height_m"] == pytest.approx(1.6, rel=1e-3)
    assert _robust(splat)[1] == pytest.approx(1.6, rel=0.01)


def test_sharp360_back_failure_falls_back_to_the_front(make_job, person, monkeypatch, check_scene):
    def broken(*a, **k):
        raise ValueError("shell bug (test)")

    monkeypatch.setattr(S, "shell_back", broken)
    photo, _ = person()
    res = run_job(make_job("object", [photo], method="sharp360"))
    assert any(
        "The built-in back failed unexpectedly (ValueError); the photo front alone was used instead." == w
        for w in res.warnings
    ), res.warnings
    assert "front only" in _splat_output(res)[0]
    check_scene(res, "object")


def test_sharp360_front_failure_saves_plain_sharp(make_job, person, monkeypatch, check_scene):
    def broken(*a, **k):
        raise RuntimeError("front bug (test)")

    monkeypatch.setattr(S, "sharp_front", broken)
    photo, _ = person()
    res = run_job(make_job("object", [photo], method="sharp360", height_m=1.6))
    assert any("could not clean SHARP's result (RuntimeError)" in w for w in res.warnings), res.warnings
    key, splat = _splat_output(res)
    assert splat.name == "photo_sharp360.splat" and "plain SHARP" in key
    block = check_scene(res, "object")["object"]
    assert "scale" not in block and block["height_m"] == 1.6  # fitted to the height by the viewer


def test_app_models_are_released_before_the_addon(make_job, person, monkeypatch, events):
    from vrphoto.models import ort

    calls = []

    def spy(model_id=None, *, wait=True):
        calls.append((model_id, wait))
        events.append("unload")

    monkeypatch.setattr(ort, "unload_model", spy)
    photo, _ = person()
    run_job(make_job("object", [photo], method="sharp360", common_extra={"export_scene": False}))
    assert calls == [(None, True)]
    assert events.index("segment") < events.index("unload") < events.index("run:sharp")


def test_releasing_app_models_never_fails_the_job(make_job, person, monkeypatch):
    from vrphoto.models import ort

    def broken(*a, **k):
        raise RuntimeError("driver hiccup (test)")

    monkeypatch.setattr(ort, "unload_model", broken)
    photo, _ = person()
    res = run_job(make_job("object", [photo], method="sharp", common_extra={"export_scene": False}))
    assert _splat_output(res)[1].is_file()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("back_source", "magic", "back source"),
        ("back_quality", "ultra", "back quality"),
        ("seam_deg", 20, "Keep the photo up to"),
        ("seam_deg", 85, "Keep the photo up to"),
        ("seam_deg", "wide", "Keep the photo up to"),
        ("size_from", "tape", "Size from"),
        ("back", "shiny", "back texture"),
        ("thickness", -0.1, "Thickness"),
        ("back_relief", "bumpy", "Back detail"),
    ],
)
def test_bad_sharp360_options_fail_before_any_work(make_job, person, events, field, value, match):
    photo, fm = person()
    with pytest.raises(InputError, match=match):
        run_job(make_job("object", [photo], method="sharp360", **{field: value}))
    assert not fm.calls and not events


# --------------------------------------------------------------------------- options and the source chain
def _spec(**kw):
    from vrphoto.pipeline.jobs import ObjectJob

    return ObjectJob(inputs=["x.jpg"], method="sharp360", **kw)


def test_resolve_options():
    o = S.resolve_options(_spec(), installed=lambda a: False)
    assert o.sources == ("shell",) and o.quality == "balanced"
    assert (o.mould_long, o.stride, o.seam, o.size_from) == (768, 1, 60.0, "auto")
    fast = S.resolve_options(_spec(back_quality="FAST", back_source="shell"), installed=lambda a: True)
    assert fast.sources == ("shell",) and (fast.mould_long, fast.stride) == (512, 2)
    best = S.resolve_options(_spec(back_quality="best", seam_deg=35, save_extras=1, seed=7), installed=bool)
    assert (best.mould_long, best.seam, best.save_extras, best.seed) == (1024, 35.0, True, 7)
    assert S.resolve_options(_spec(seam_deg="80"), installed=bool).seam == 80.0


def test_auto_skips_sources_whose_addon_is_missing(monkeypatch):
    """The extension point for the AI sources of later versions."""
    monkeypatch.setitem(S.SOURCES, "fake", lambda ctx, rep: None)
    monkeypatch.setitem(S.SOURCE_ADDONS, "fake", "fakeaddon")
    monkeypatch.setitem(S.QUALITY, "balanced", S.Quality(768, 1, ("fake", "shell")))
    asked = []

    def installed(addon_id):
        asked.append(addon_id)
        return addon_id == "fakeaddon"

    assert S.resolve_options(_spec(), installed=installed).sources == ("fake", "shell")
    assert asked == ["fakeaddon"]
    assert S.resolve_options(_spec(), installed=lambda a: False).sources == ("shell",)


class _RunStub:
    def __init__(self):
        self.warnings: list[str] = []
        self.cancel = CancelToken()

    def warn(self, msg):
        self.warnings.append(msg)

    def check(self):
        self.cancel.check()


def test_resolve_options_clamps_a_billboard_thickness():
    """The object presets set Thickness for Billboard (up to 0.24): SHARP 360's life-size back
    keeps it within a person's range, and says so."""
    lo, hi = S.THICKNESS_RANGE
    for asked, used in ((0.24, hi), (0.5, hi), (0.0, lo), (0.03, lo)):
        o = S.resolve_options(_spec(thickness=asked), installed=bool)
        assert (o.thickness, o.thickness_asked) == (used, asked)
    o = S.resolve_options(_spec(thickness=0.11), installed=bool)
    assert (o.thickness, o.thickness_asked) == (0.11, None)
    assert lo <= 0.08 and 0.15 <= hi  # the user guide's recommended range fits


def test_sharp360_warns_when_the_thickness_is_clamped(make_job, person):
    photo, _ = person()
    res = run_job(
        make_job(
            "object",
            [photo],
            method="sharp360",
            thickness=0.24,
            save_extras=True,
            common_extra={"export_scene": False},
        )
    )
    assert any("Thickness of 0.18 instead of 0.24" in w for w in res.warnings), res.warnings
    out = {Path(p).name: Path(p) for p in res.outputs.values()}
    dbg = json.loads(out["photo_sharp360_debug.json"].read_text(encoding="utf-8"))
    assert dbg["options"]["thickness"] == 0.18 and dbg["options"]["thickness_asked"] == 0.24
    assert dbg["canon"]["T_prior"] == pytest.approx(0.18 * dbg["canon"]["height_s"])


def test_export_never_crops_front_rows():
    """The life-size box (P0.2-P99.8 + 3 cm) is for the generated part: a thin protected part
    of the photo front held out from the body (a stick, a pointing hand) stays whole."""
    from vrphoto.core.types import ProgressReporter, SplatCloud
    from vrphoto.geometry.splat_fusion import Canon

    rng = np.random.default_rng(0)
    n = 30_000  # a 1.7 m upright "body" in frame C (y down), 2.5 m from the camera
    a = rng.uniform(0, 2 * np.pi, n)
    body = np.c_[0.2 * np.cos(a), rng.uniform(-0.85, 0.85, n), 2.5 + 0.12 * np.sin(a)]
    t = np.linspace(0.22, 0.72, 250)  # a stick held out sideways, 0.8 % of the rows
    stick = np.c_[t, np.full_like(t, -0.3), np.full_like(t, 2.5)] + rng.normal(0, 0.005, (250, 3))
    junk = np.c_[np.full(10, 1.5), np.zeros(10), np.full(10, 2.6)]  # generated, far beside it
    P = np.concatenate([body, stick, junk])
    m = len(P)
    sc = SplatCloud(
        P.astype(np.float32),
        np.full((m, 3), 0.004, np.float32),
        np.tile([1.0, 0.0, 0.0, 0.0], (m, 1)).astype(np.float32),
        np.full((m, 3), 0.5, np.float32),
        np.full(m, 0.9, np.float32),
    )
    is_front = np.r_[body[:, 2] < 2.5, np.ones(len(stick), bool), np.zeros(len(junk), bool)]
    canon = Canon(
        R=np.eye(3),
        c=np.array([0.0, 0.0, 2.5]),
        up_C=np.array([0.0, -1.0, 0.0]),
        up_source="camera",
        height_s=1.7,
        width_s=0.4,
        T_prior=0.19,
        z_axis=2.5,
        pitch_deg=0.0,
    )
    o = S.resolve_options(_spec(size_from="photo"), installed=bool)
    ex = S._export_cloud(sc, is_front, canon, o, 1.7, _RunStub(), ProgressReporter.null())
    assert ex.info["front_outside_box"] > 0  # the box alone would cut the stick's end
    x = np.asarray(ex.sc.means)[:, 0]
    assert len(ex.sc) == m - len(junk) and len(ex.front) == int(is_front.sum())
    # the whole stick (normalize_orientation only moves the x / z centre)
    assert x.max() - x.min() == pytest.approx(stick[:, 0].max() - P[:, 0].min(), abs=1e-4)
    assert len(ex.back) == int((~is_front).sum()) - len(junk)


def _ctx(sources):
    o = S.resolve_options(_spec(), installed=bool)
    o = S.Opts(**{**o.__dict__, "sources": tuple(sources)})
    return S.Ctx(_RunStub(), None, None, None, None, None, None, o)


def test_source_chain_falls_back_with_one_warning_each(monkeypatch):
    from vrphoto.core.types import ProgressReporter, SplatCloud

    def part(ok=True, reason=""):
        n = 0 if not ok else 3
        sc = SplatCloud(
            np.zeros((n, 3), np.float32),
            np.full((n, 3), 0.01, np.float32),
            np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
            np.zeros((n, 3), np.float32),
            np.ones(n, np.float32),
        )
        return BackPart(sc=sc, source="x", ok=ok, reason=reason)

    def missing(ctx, rep):
        raise AddonNotInstalled("TRELLIS is not installed.")

    def oom(ctx, rep):
        raise AddonError("CUDA out of memory")

    def bug(ctx, rep):
        raise KeyError("x")

    for name, fn in {
        "a": missing,
        "b": lambda ctx, rep: part(ok=False, reason="the views did not line up"),
        "c": oom,
        "d": bug,
        "shell": lambda ctx, rep: part(),
    }.items():
        monkeypatch.setitem(S.SOURCES, name, fn)
    for name in "abcd":
        monkeypatch.setitem(S.SOURCE_NAMES, name, name.upper())
    ctx = _ctx(["a", "b", "c", "d", "shell", "zz"])
    attempts: list[dict] = []
    got = S.run_sources(ctx, ProgressReporter.null(), attempts)
    assert got is not None and got.ok and len(got.sc) == 3
    assert ctx.run.warnings == [
        "A is not installed; B was used instead.",
        "B could not make a usable back (the views did not line up); C was used instead.",
        "C could not make the back (CUDA out of memory); D was used instead.",
        "D failed unexpectedly (KeyError); the built-in back was used instead.",
    ]
    assert [a["source"] for a in attempts] == ["a", "b", "c", "d", "shell"]
    assert [a["ok"] for a in attempts] == [False, False, False, False, True]
    # nothing usable at all: None (the job then exports the front alone)
    monkeypatch.setitem(S.SOURCES, "shell", lambda ctx, rep: part(ok=False, reason="too small"))
    ctx = _ctx(["shell"])
    assert S.run_sources(ctx, ProgressReporter.null(), []) is None
    assert ctx.run.warnings == [
        "The built-in back could not make a usable back (too small); the photo front alone was used instead."
    ]


def test_source_chain_never_swallows_a_cancel(monkeypatch):
    from vrphoto.core.types import ProgressReporter

    def cancelled(ctx, rep):
        raise Cancelled("Operation cancelled")

    monkeypatch.setitem(S.SOURCES, "a", cancelled)
    ctx = _ctx(["a", "shell"])
    with pytest.raises(Cancelled):
        S.run_sources(ctx, ProgressReporter.null(), [])
    assert not ctx.run.warnings


def test_sharp360_is_an_object_method_needing_only_sharp():
    assert "sharp360" in R.OBJECT_METHODS
    assert R.ADDON_METHODS["sharp360"] == ("sharp", "image_to_splat")
    assert set(S.SOURCES) >= set(R.BACK_SOURCES) - {"auto"}
    assert set(S.QUALITY) == set(R.BACK_QUALITIES)


def test_sharp360_survives_a_cloud_that_misses_the_subject(make_job, person, check_scene):
    """SHARP's result does not match the cut-out at all (random blobs): still an output."""
    from vrphoto.core.types import SplatCloud

    def garbage(task, inputs, params, out_dir):
        from vrphoto.geometry.splat import write_splats

        n = 2000
        rng = np.random.default_rng(0)
        sc = SplatCloud(
            rng.normal(0, 0.3, (n, 3)).astype(np.float32) + np.array([0, 0, 2], np.float32),
            np.full((n, 3), 0.01, np.float32),
            np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
            rng.random((n, 3)).astype(np.float32),
            np.full(n, 0.9, np.float32),
        )
        return {"ply": str(write_splats(sc, out_dir / "model.ply")), "frame": "opencv", "f_px": 500.0}

    photo, _ = person(sharp=garbage)
    res = run_job(make_job("object", [photo], method="sharp360", height_m=1.6))
    assert _splat_output(res)[1].is_file()
    check_scene(res, "object")
    assert any("could not be separated from the background" in w for w in res.warnings), res.warnings


def test_sharp360_resolves_the_back_photo_and_relief(tmp_path):
    back = _views_folder(tmp_path, names=("back",))
    o = S.resolve_options(_spec(back_source="images", back_images=str(back)), installed=lambda a: True)
    assert o.sources == ("images", "shell") and o.back_images == str(back) and o.relief == "auto"
    # a path pasted with Windows Explorer's 'Copy as path' keeps its quotes; a folder is fine too
    for value, want in ((f'"{back}"', back), (f"  '{back}' ", back), (str(back.parent), back.parent)):
        o = S.resolve_options(_spec(back_source="images", back_images=value), installed=bool)
        assert o.back_images == str(want), value
    # "auto" never picks the back photo (it needs the user's picture), whatever is installed
    for quality in R.BACK_QUALITIES:
        o = S.resolve_options(_spec(back_quality=quality, back_images=str(back)), installed=lambda a: True)
        assert "images" not in o.sources and o.back_images is None
    # a remembered (even missing) back photo is ignored by the other sources
    o = S.resolve_options(_spec(back_source="shell", back_images=str(tmp_path / "gone.jpg")), installed=bool)
    assert o.back_images is None
    assert S.resolve_options(_spec(back_relief=" FLAT "), installed=bool).relief == "flat"
    # 'Back detail' Automatic: the depth model, except with "fast" (smooth, design 1.3)
    for quality, relief, mode in (
        ("balanced", "auto", "auto"),
        ("best", "depth", "depth"),
        ("fast", "auto", "flat"),
        ("fast", "depth", "depth"),
        ("balanced", "flat", "flat"),
    ):
        o = S.resolve_options(_spec(back_quality=quality, back_relief=relief), installed=bool)
        got, fn, used = S._relief_fn(SimpleNamespace(opts=o, run=None))
        assert got == mode and (fn is None) == (mode == "flat") and used == []
    assert S.QUALITY["fast"].fill_holes is False and S.QUALITY["balanced"].fill_holes is True


# --------------------------------------------------------------------------- previews (design 1.5 / 3.13)
def _outputs(res) -> dict[str, tuple[str, Path]]:
    """``file name -> (label, path)`` of a job's outputs."""
    return {Path(p).name: (k, Path(p)) for k, p in res.outputs.items()}


def _image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _tile_colours(sheet: np.ndarray, size: int, k: int) -> dict[str, float]:
    """Share of each side colour among the subject's pixels of turntable view ``k``."""
    r, c = divmod(k, 4)
    tile = sheet[r * size : (r + 1) * size, c * size : (c + 1) * size].reshape(-1, 3)
    subject = np.abs(tile.astype(np.int16) - 32).max(axis=1) > 40  # not the dark grey background
    names = sp.classify_rgb(tile[subject], sp.SIDES)
    return {n: float((names == n).mean()) for n in sp.SIDES}


def test_sharp360_writes_a_turntable_preview(make_job, person):
    photo, _ = person()
    res = run_job(
        make_job("object", [photo], method="sharp360", save_extras=True, common_extra={"export_scene": False})
    )
    out = _outputs(res)
    label, prev = out["photo_sharp360_preview.jpg"]
    assert label == "SHARP 360° turntable preview (JPEG)"
    sheet = _image(prev)
    assert sheet.shape == (2 * 384, 4 * 384, 3)  # 8 views of "balanced"'s 384 px, 4 per row
    assert "photo_sharp360_views.png" not in out  # no view-based source ran
    # the 0 deg view is the photographed (red) front; every view shows the subject
    assert _tile_colours(sheet, 384, 0)["front"] > 0.6
    for k in range(8):
        r, c = divmod(k, 4)
        tile = sheet[r * 384 : (r + 1) * 384, c * 384 : (c + 1) * 384]
        assert (np.abs(tile.astype(np.int16) - 32).max(axis=2) > 40).mean() > 0.02, k
    dbg = json.loads(out["photo_sharp360_debug.json"][1].read_text(encoding="utf-8"))
    assert dbg["previews"]["turntable"] == dict(dbg["previews"]["turntable"], views=8, px=384)
    assert dbg["seconds"]["previews"] > 0
    # "fast" makes smaller views
    res = run_job(
        make_job(
            "object", [photo], method="sharp360", back_quality="fast", common_extra={"export_scene": False}
        )
    )
    assert _image(Path(res.outputs["SHARP 360° turntable preview (JPEG)"])).shape == (2 * 256, 4 * 256, 3)


def test_a_failed_preview_is_only_a_warning(make_job, person, monkeypatch):
    from vrphoto.geometry import splat_render

    def broken(*a, **k):
        raise RuntimeError("renderer bug (test)")

    monkeypatch.setattr(splat_render, "turntable_sheet", broken)
    photo, _ = person()
    res = run_job(make_job("object", [photo], method="sharp360", common_extra={"export_scene": False}))
    assert (
        "The turntable preview could not be made (RuntimeError); the 3D result is not affected."
        in res.warnings
    )
    assert _splat_output(res)[1].is_file() and "photo_sharp360_preview.jpg" not in _outputs(res)
    # ... but a cancel while it renders stops the job
    token = CancelToken()

    def cancelled(*a, **k):
        token.cancel()
        raise Cancelled("Operation cancelled")

    monkeypatch.setattr(splat_render, "turntable_sheet", cancelled)
    with pytest.raises(Cancelled):
        run_job(
            make_job("object", [photo], method="sharp360", common_extra={"export_scene": False}), cancel=token
        )


# --------------------------------------------------------------------------- your own back photo (design 3.5.4)
#: Azimuth, metres per pixel and size of the synthetic "user photos" (each framed differently).
VIEW_SHOTS = {"back": (180.0, 0.006, 360), "right": (90.0, 0.005, 420), "left": (270.0, 0.007, 300)}


def _views_folder(folder: Path, names=("back", "right"), body=None) -> Path:
    """Synthetic photos of the person from behind / the side next to each other: orthographic
    renders on a grey studio background, each framed differently from the photo (the app
    lines them up).  ``body`` poses the person differently.  Returns the back photo's path."""
    from PIL import Image

    canon = sp.synth_canon(sp.make_body(), sp.make_cam(W, H, HFOV, grid=G))
    body = body or sp.make_body()
    folder = Path(folder) / "back_photos"
    folder.mkdir(exist_ok=True)
    for name in names:
        phi, m, n = VIEW_SHOTS[name]
        rgb, _, _ = sp.ortho_view(body, canon, phi, m=m, N=n)
        Image.fromarray(rgb).save(folder / f"{name}.png")
    return folder / "back.png"


def _far_sides(path: Path) -> dict[str, float]:
    """Share of each side colour among the opaque Gaussians of the far quarter of an exported
    splat (lowest z: the subject faces +z).  The true body there is mostly its blue back;
    the built-in back takes its colours from the photo's outline band (the red front)."""
    from vrphoto.geometry.splat import read_splats

    sc = read_splats(path)
    z = sc.means[:, 2]
    names = sp.classify_rgb(sc.colors[(z < np.percentile(z, 25)) & (sc.opacity > 0.5)], sp.SIDES)
    return {n: float((names == n).mean()) for n in sp.SIDES}


def test_sharp360_back_from_your_photos(make_job, person, check_scene, monkeypatch):
    from pipeline_helpers import capture_stitch, front_render_difference, generated_on_the_front

    seen = capture_stitch(monkeypatch)
    photo, fm = person()
    back = _views_folder(photo.parent)  # back.png + right.png
    res = run_job(
        make_job(
            "object",
            [photo],
            method="sharp360",
            back_source="images",
            back_images=str(back),
            back_quality="fast",  # the quickest mould: what is tested here is the source itself
            height_m=1.6,
            save_extras=True,
        )
    )
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]  # no AI add-on
    assert not [w for w in res.warnings if not w.startswith("Note:")], res.warnings
    key, splat = _splat_output(res)
    assert key == "3D object (Gaussian splat, SHARP 360°, back from your photo)"
    block = check_scene(res, "object")["object"]
    assert block["scale"] == 1.0 and block["height_m"] == pytest.approx(1.6, rel=1e-3)
    assert _robust(splat)[1] == pytest.approx(1.6, rel=0.01)

    out = _outputs(res)
    dbg = json.loads(out["photo_sharp360_debug.json"][1].read_text(encoding="utf-8"))
    assert dbg["options"]["sources"] == ["images", "shell"] and dbg["options"]["back_images"] == str(back)
    assert dbg["sources"] == [dict(dbg["sources"][0], source="images", ok=True)]
    assert dbg["back"]["source"] == "images" and dbg["back"]["kind"] == "mould"
    d = dbg["back"]["diag"]
    assert d["files"] == {"90": "right.png", "180": "back.png"}
    assert d["back"]["iou_locked"] >= 0.95 and d["back"]["pose_frac"] < 0.06, d["back"]
    assert [v["name"] for v in d["views"] if v.get("used")] == ["right"], d["views"]
    assert d["relief"]["source"] == "flat"  # "fast": a smooth back
    assert d["holes"] == {} and d["qc"]["cost"] >= 0  # "fast" fills no side holes
    # the back is the photo's (blue), not the built-in one's (red, from the outline band)
    sides = _far_sides(out["photo_sharp360_back.splat"][1])
    assert sides["back"] > 0.6 and sides["front"] < 0.05, sides
    # the turntable shows it from behind (180 deg: bottom-left view; 256 px tiles with "fast")
    assert _tile_colours(_image(out["photo_sharp360_preview.jpg"][1]), 256, 4)["back"] > 0.6
    # the contact sheet: the levelled photo, the back and the right photo
    label, views_png = out["photo_sharp360_views.png"]
    assert label == "SHARP 360° back photos as lined up (PNG)"
    assert _image(views_png).shape == (256, 3 * 256, 3)
    assert dbg["previews"]["views"] == dict(dbg["previews"]["views"], views=2)
    # the photographed front is untouched: seen from the photo's camera the fused result is
    # SHARP's cleaned front (design 5.2), and nothing generated lies on the photographed side
    diff, n = front_render_difference(seen)
    assert n > 5000 and diff < 2.0, diff
    share, n_gen = generated_on_the_front(seen)
    assert n_gen > 1000 and share < 0.02, share


def test_sharp360_back_relief_from_the_depth_model(make_job, person, monkeypatch):
    """A real depth model shapes the back from the back photo; its licence notice is added."""
    from vrphoto.core.types import DepthResult

    calls = []

    def depth(self, img, **kw):
        calls.append(img.shape)
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        bulge = 1.0 - np.hypot((xx - w / 2) / w, (yy - h / 2) / h)  # the middle is nearest
        return DepthResult(disparity=bulge.astype(np.float32), model_id="da2-base")

    monkeypatch.setattr(context.RunContext, "depth", depth)
    photo, _ = person()
    back = _views_folder(photo.parent, names=("back",))
    res = run_job(
        make_job(
            "object",
            [photo],
            method="sharp360",
            back_source="images",
            back_images=str(back),
            back_quality="fast",  # 'Back detail' asked for explicitly: the depth model runs even so
            back_relief="depth",
            save_extras=True,
            common_extra={"export_scene": False},
        )
    )
    assert calls == [(512, 512, 3)]  # the back photo, lined up in the levelled frame
    dbg = json.loads(_outputs(res)["photo_sharp360_debug.json"][1].read_text(encoding="utf-8"))
    assert dbg["back"]["source"] == "images"
    assert dbg["back"]["diag"]["relief"] == dict(dbg["back"]["diag"]["relief"], source="depth", mode="detail")
    notes = [w for w in res.warnings if w.startswith("Note:") and "non-commercial model" in w]
    assert len(notes) == 1 and "Depth Anything" in notes[0], res.warnings
    # ('Smooth' and "fast" never run it: test_sharp360_resolves_the_back_photo_and_relief)


def test_sharp360_back_photo_in_another_pose_falls_back(make_job, person):
    photo, fm = person()
    back = _views_folder(photo.parent, names=("back",), body=sp.make_body_pose(left_arm_deg=60.0))
    res = run_job(
        make_job(
            "object",
            [photo],
            method="sharp360",
            back_source="images",
            back_images=str(back),
            save_extras=True,
            common_extra={"export_scene": False},
        )
    )
    fb = [w for w in res.warnings if not w.startswith("Note:")]
    assert len(fb) == 1, res.warnings
    assert fb[0].startswith("Your back photo could not make a usable back (") and "different pose" in fb[0]
    assert fb[0].endswith("; the built-in back was used instead."), fb
    assert "built-in back" in _splat_output(res)[0]
    out = _outputs(res)
    dbg = json.loads(out["photo_sharp360_debug.json"][1].read_text(encoding="utf-8"))
    att = dbg["sources"]
    assert [(a["source"], a["ok"]) for a in att] == [("images", False), ("shell", True)]
    assert att[0]["diag"]["back"]["pose_frac"] > 0.06
    # the contact sheet still shows the back photo (outlined as not used): it says why
    assert _image(out["photo_sharp360_views.png"][1]).shape == (256, 2 * 256, 3)


def _half_copied(path: Path) -> Path:
    """The picture as a half-finished copy leaves it: a JPEG whose header is fine (the job
    accepts it before any work) but whose data ends half way (decoding fails)."""
    from PIL import Image

    with Image.open(path) as im:
        im.convert("RGB").save(path.with_suffix(".jpg"), quality=95)
    data = path.with_suffix(".jpg").read_bytes()
    path.unlink()
    out = path.with_suffix(".jpg")
    out.write_bytes(data[: len(data) // 2])
    return out


@pytest.mark.parametrize("case", ["back_unreadable", "side_unreadable", "nothing_in_either"])
def test_sharp360_back_photos_that_cannot_be_used_are_named(make_job, person, case):
    """An unreadable or empty back / side photo is named in the warning (not "there is no
    picture of the back", nor the side photo's problem in place of the back photo's)."""
    photo, _ = person()
    back = _views_folder(photo.parent)  # back.png + right.png
    if case == "back_unreadable":
        back = _half_copied(back)
    elif case == "side_unreadable":
        _half_copied(back.parent / "right.png")
    else:  # both pictures show only the grey studio background
        from PIL import Image

        for name in ("back.png", "right.png"):
            Image.fromarray(np.full((300, 300, 3), 128, np.uint8)).save(back.parent / name)
    res = run_job(
        make_job(
            "object",
            [photo],
            method="sharp360",
            back_source="images",
            back_images=str(back),
            back_quality="fast",
            save_extras=True,
            common_extra={"export_scene": False},
        )
    )
    fb = [w for w in res.warnings if not w.startswith("Note:")]
    dbg = json.loads(_outputs(res)["photo_sharp360_debug.json"][1].read_text(encoding="utf-8"))
    if case == "side_unreadable":  # the back photo is used; the side photo is named
        assert dbg["back"]["source"] == "images"
        assert len(fb) == 1 and "The right picture (right.jpg) could not be read" in fb[0], fb
        d = dbg["back"]["diag"]
        assert d["files"] == {"180": "back.png"} and "right.jpg" in d["dropped"][0]
        return
    assert len(fb) == 1 and fb[0].startswith("Your back photo could not make a usable back ("), fb
    assert "built-in back" in _splat_output(res)[0]
    if case == "back_unreadable":
        assert "the back picture (back.jpg) could not be read (" in fb[0], fb
    else:
        assert "the back picture (back.png) was not used: no subject was found" in fb[0], fb
    assert "right" not in fb[0] and "there is no picture of the back" not in fb[0]


@pytest.mark.parametrize("case", ["none", "empty", "missing", "not_a_picture", "no_back", "corrupt"])
def test_sharp360_back_photo_problems_fail_before_any_work(make_job, person, events, tmp_path, case):
    folder = tmp_path / "pics"
    folder.mkdir()
    (folder / "notes.txt").write_text("not a picture", encoding="utf-8")
    (folder / "broken.png").write_bytes(b"this is not a PNG")
    value, match = {
        "none": (None, "needs a picture of the back"),
        "empty": ("  ", "needs a picture of the back"),
        "missing": (str(folder / "gone.jpg"), "was not found"),
        "not_a_picture": (str(folder / "notes.txt"), "is not a picture"),
        "no_back": (str(_views_folder(tmp_path, names=("right",)).parent), "No picture of the back"),
        "corrupt": (str(folder / "broken.png"), "cannot be used"),
    }[case]
    photo, fm = person()
    with pytest.raises(InputError, match=match):
        run_job(make_job("object", [photo], method="sharp360", back_source="images", back_images=value))
    assert not fm.calls and not events  # before SHARP and before finding the subject
