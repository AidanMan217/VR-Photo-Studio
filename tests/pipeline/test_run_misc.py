"""run_job behaviour shared by all modes: validation, batches, cancel, progress, warnings, notices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError, VRPhotoError
from vrphoto.core.types import CancelToken, DepthResult, Intrinsics
from vrphoto.pipeline import context
from vrphoto.pipeline import run as R
from vrphoto.pipeline.jobs import StereoJob
from vrphoto.pipeline.run import run_job


class StepEstimator:
    """Fake model estimator: left half near, right half far (strong depth edge)."""

    def __init__(self, ident="da2-small", result_id=None, metric=False, K=None, reason=None):
        self.id = ident
        self.result_id = result_id or ident
        self.metric = metric
        self.K = K
        self.fallback_reason = reason
        self.calls = 0

    def estimate(self, img, *, progress=None, cancel=None):
        self.calls += 1
        h, w = img.shape[:2]
        d = np.zeros((h, w), np.float32)
        d[:, : w // 2] = 1.0
        metric = np.where(d > 0.5, 1.5, 6.0).astype(np.float32) if self.metric else None
        return DepthResult(d, self.result_id, metric_m=metric, intrinsics=self.K)


@pytest.fixture
def fake_estimator(monkeypatch):
    def install(est):
        monkeypatch.setattr(context.RunContext, "estimator", lambda self, *a, **k: est)
        return est

    return install


# --------------------------------------------------------------------------- validation
def test_rejects_non_specs_and_unknown_kinds():
    with pytest.raises(InputError):
        run_job({"kind": "stereo"})  # type: ignore[arg-type]

    class Weird(StereoJob):
        kind = "hologram"  # type: ignore[assignment]

    with pytest.raises(InputError, match="Unknown job kind"):
        run_job(Weird(inputs=["x.jpg"]))


def test_missing_and_empty_inputs(make_job, tmp_path):
    with pytest.raises(InputError, match="No input files"):
        run_job(make_job("stereo", []))
    with pytest.raises(InputError, match="File not found") as ei:
        run_job(make_job("stereo", [tmp_path / "nope.jpg"]))
    assert ei.value.hint


@pytest.mark.parametrize(
    ("fields", "extra", "match"),
    [
        ({"layout": "holo"}, {}, "stereo layout"),
        ({"look_behind": "xray"}, {}, "look-behind"),
        ({"image_format": "gif"}, {}, "image format"),
        ({"ipd_m": 0}, {}, "Eye distance"),
        ({"strength": -1}, {}, "3D strength"),
        ({}, {"near_m": 5.0, "far_m": 2.0}, "farthest distance"),
        ({}, {"near_m": 0}, "Nearest distance"),
        ({}, {"quest_model": "vive"}, "headset"),
        ({}, {"max_process_px": 100}, "at least 256"),
        ({}, {"hfov_deg": 400}, "Field of view"),
    ],
)
def test_option_validation(make_job, gradient_photo, fields, extra, match):
    with pytest.raises(InputError, match=match):
        run_job(make_job("stereo", [gradient_photo], common_extra=extra, **fields))


@pytest.mark.parametrize(
    ("kind", "fields", "match"),
    [
        ("parallax", {"output": "imax"}, "video type"),
        ("parallax", {"fps": 0}, "Frame rate"),
        ("parallax", {"encoder_preset": "av1_super"}, "video format"),
        ("vr180", {"fill": "lava"}, "fill style"),
        ("vr180", {"eye_px": 10}, "at least 64"),
        ("pano", {"fill": "lava"}, "fill style"),
        ("scene", {"look_behind": "xray"}, "look-behind"),
        ("scene", {"target_faces": 10}, "at least 1000"),
        ("object", {"method": "magic"}, "3D method"),
        ("object", {"placement": "ceiling"}, "placement"),
        ("object", {"height_m": 0}, "Height"),
        ("video", {"output": "imax"}, "video type"),
    ],
)
def test_mode_option_validation(make_job, gradient_photo, kind, fields, match):
    with pytest.raises(InputError, match=match):
        run_job(make_job(kind, [gradient_photo], **fields))


def test_uncreatable_output_folder(make_job, gradient_photo, tmp_path):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    with pytest.raises(InputError, match="output folder"):
        run_job(make_job("stereo", [gradient_photo], common_extra={"output_dir": str(blocker / "sub")}))


# --------------------------------------------------------------------------- batches
def test_batch_continues_after_a_bad_file(make_job, gradient_photo, tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"this is not a jpeg")
    res = run_job(make_job("stereo", [bad, gradient_photo]))
    keys = list(res.outputs)
    assert any(k.startswith("photo: 3D photo") for k in keys)
    assert not any(k.startswith("broken") for k in keys)
    assert any("broken.jpg could not be processed" in w for w in res.warnings)


def test_batch_fails_when_every_file_fails(make_job, tmp_path):
    files = []
    for i in range(2):
        p = tmp_path / f"bad{i}.jpg"
        p.write_bytes(b"garbage")
        files.append(p)
    with pytest.raises(InputError, match="None of the 2 files"):
        run_job(make_job("stereo", files))


def test_batch_names_with_custom_name(make_job, gradient_photo, photo_png):
    res = run_job(
        make_job("stereo", [gradient_photo, photo_png], common_extra={"name": "trip", "export_scene": False})
    )
    names = sorted(Path(p).name for p in res.outputs.values())
    assert names == ["trip_1_LR.jpg", "trip_2_LR.jpg"]
    assert set(res.outputs) == {"trip_1: 3D photo (side-by-side)", "trip_2: 3D photo (side-by-side)"}


def test_batch_progress_is_prefixed_and_reaches_one(make_job, gradient_photo, photo_png):
    seen: list[tuple[float, str]] = []
    run_job(make_job("depth", [gradient_photo, photo_png]), progress=lambda f, m: seen.append((f, m)))
    assert seen[-1][0] == 1.0
    assert any(m.startswith("[1/2] ") for _, m in seen) and any(m.startswith("[2/2] ") for _, m in seen)
    first_two = [f for f, m in seen if m.startswith("[2/2] ")]
    assert min(first_two) >= 0.5 - 1e-9  # second file maps onto the second half
    fr = [f for f, _ in seen]
    assert all(b >= a - 1e-6 for a, b in zip(fr, fr[1:]))  # never goes backwards


# --------------------------------------------------------------------------- cancel
def test_cancel_before_start(make_job, gradient_photo):
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        run_job(make_job("stereo", [gradient_photo]), cancel=tok)


def test_cancel_during_video_leaves_no_partial_file(make_job, gradient_photo, out_dir):
    tok = CancelToken()

    def progress(frac, msg):
        if "frame 3/" in msg:
            tok.cancel()

    with pytest.raises(Cancelled):
        run_job(
            make_job("parallax", [gradient_photo], duration_s=1.0, fps=10, encoder_preset="h264_compat"),
            progress=progress,
            cancel=tok,
        )
    assert list(out_dir.iterdir()) == []


def test_unexpected_error_after_cancel_is_cancelled(make_job, gradient_photo, monkeypatch):
    tok = CancelToken()
    import vrphoto.geometry.stereo as stereo

    def boom(*a, **k):
        tok.cancel()
        raise BrokenPipeError("pipe")

    monkeypatch.setattr(stereo, "compose", boom)
    with pytest.raises(Cancelled):
        run_job(make_job("stereo", [gradient_photo]), cancel=tok)


def test_memory_error_is_friendly(make_job, gradient_photo, monkeypatch):
    import vrphoto.geometry.stereo as stereo

    def oom(*a, **k):
        raise MemoryError()

    monkeypatch.setattr(stereo, "compose", oom)
    with pytest.raises(VRPhotoError, match="Not enough memory") as ei:
        run_job(make_job("stereo", [gradient_photo]))
    assert "Max processing size" in ei.value.hint


# --------------------------------------------------------------------------- warnings / notices
def test_heuristic_fallback_is_reported(make_job, gradient_photo, fake_estimator):
    fake_estimator(StepEstimator("da2-small", result_id="heuristic", reason="no internet connection"))
    res = run_job(make_job("stereo", [gradient_photo], common_extra={"depth_model": "auto"}))
    assert any("AI depth model could not be used" in w and "no internet" in w for w in res.warnings)


def test_explicit_heuristic_has_no_fallback_warning(make_job, gradient_photo):
    res = run_job(make_job("stereo", [gradient_photo]))
    assert not any("depth model" in w for w in res.warnings)


def test_noncommercial_model_notice_in_manifest(make_job, gradient_photo, fake_estimator, check_scene):
    fake_estimator(StepEstimator("da2-large"))
    res = run_job(make_job("stereo", [gradient_photo], common_extra={"depth_model": "da2-large"}))
    m = check_scene(res, "stereo")
    assert any("non-commercial" in n and "Depth Anything V2 Large" in n for n in m["notices"])
    assert any(w.startswith("Note:") for w in res.warnings)


def test_simple_inpaint_fallback_is_reported(make_job, gradient_photo, fake_estimator, monkeypatch):
    fake_estimator(StepEstimator("da2-small"))
    from vrphoto.inpaint import get_inpainter

    class FellBack:
        id = "lama"
        resolved = None

        def __init__(self):
            self.inner = get_inpainter("opencv")
            self.calls = 0

        def inpaint(self, img, mask, **kw):
            self.calls += 1
            self.resolved = "opencv"
            return self.inner.inpaint(img, mask)

    inp = FellBack()
    monkeypatch.setattr(context.RunContext, "inpainter", lambda self, *a, **k: inp)
    res = run_job(make_job("stereo", [gradient_photo], common_extra={"inpaint_model": "auto"}, strength=2.0))
    assert inp.calls >= 1
    assert any("look-behind model could not be used" in w for w in res.warnings)


def test_look_behind_none_never_creates_an_inpainter(make_job, gradient_photo, fake_estimator, monkeypatch):
    fake_estimator(StepEstimator("da2-small"))

    def forbidden(*a, **k):
        raise AssertionError("no inpainter needed")

    monkeypatch.setattr(context.RunContext, "inpainter", forbidden)
    run_job(make_job("stereo", [gradient_photo], look_behind="none"))


def test_depth_cache_is_used_across_jobs(make_job, gradient_photo, fake_estimator, settings):
    est = fake_estimator(StepEstimator("da2-small"))
    ctx = context.RunContext(settings)
    run_job(make_job("stereo", [gradient_photo], common_extra={"export_scene": False}), ctx)
    run_job(make_job("depth", [gradient_photo]), ctx)
    assert est.calls == 1


def test_metric_depth_and_model_intrinsics_are_used(settings, gradient_photo, fake_estimator):
    K_model = Intrinsics.from_hfov(640, 480, 90.0)
    fake_estimator(StepEstimator("moge2-vitl", metric=True, K=K_model))
    from vrphoto.pipeline.jobs import job_from_dict

    spec = job_from_dict({"kind": "depth", "inputs": [str(gradient_photo)], "common": {"depth_refine": 0}})
    run = R._Run(spec, context.RunContext(settings), CancelToken())
    item = R._Item(0, gradient_photo, "photo")
    photo = R._load_photo(run, item, R.ProgressReporter.null())
    d = R._estimate_depth(run, item, photo, R.ProgressReporter.null())
    assert d.K.hfov_deg == pytest.approx(90.0, abs=0.1)  # MoGe beats EXIF (26 mm) and the default
    assert d.inv_z[:, :10].mean() == pytest.approx(1 / 1.5, rel=1e-3)
    assert d.inv_z[:, -10:].mean() == pytest.approx(1 / 6.0, rel=1e-3)


def test_exif_and_user_fov(settings, gradient_photo):
    from vrphoto.pipeline.jobs import job_from_dict

    for common, expected in (({}, 69.4), ({"hfov_deg": 50.0}, 50.0)):
        spec = job_from_dict(
            {
                "kind": "depth",
                "inputs": [str(gradient_photo)],
                "common": {"depth_model": "heuristic", **common},
            }
        )
        run = R._Run(spec, context.RunContext(settings), CancelToken())
        item = R._Item(0, gradient_photo, "photo")
        photo = R._load_photo(run, item, R.ProgressReporter.null())
        d = R._estimate_depth(run, item, photo, R.ProgressReporter.null())
        assert d.K.hfov_deg == pytest.approx(expected, abs=0.2)


def test_large_photos_are_limited_to_max_process_px(make_job, tmp_path):
    from PIL import Image
    from pipeline_helpers import image_size

    p = tmp_path / "big.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (600, 1000, 3), dtype=np.uint8)).save(p)
    res = run_job(make_job("stereo", [p], common_extra={"max_process_px": 400, "export_scene": False}))
    assert image_size(Path(res.outputs["3D photo (side-by-side)"])) == (800, 240)


def test_scene_export_failure_is_a_warning(make_job, gradient_photo, monkeypatch):
    import vrphoto.webxr.scene as scene

    def broken(*a, **k):
        raise InputError("disk full")

    monkeypatch.setattr(scene, "export_scene", broken)
    res = run_job(make_job("stereo", [gradient_photo]))
    assert res.scene_id is None
    assert Path(res.outputs["3D photo (side-by-side)"]).is_file()
    assert any("could not be added to the VR viewer library" in w for w in res.warnings)


def test_call_opt_drops_unsupported_kwargs():
    def f(a, *, b=1):
        return a, b

    assert R._call_opt(f, 1, b=2, progress=3, cancel=4) == (1, 2)

    def g(a, **kw):
        return kw

    assert R._call_opt(g, 1, progress=3) == {"progress": 3}


def test_progress_accepts_reporter_or_callable(make_job, gradient_photo):
    from vrphoto.core.types import ProgressReporter

    seen = []
    run_job(make_job("depth", [gradient_photo]), progress=ProgressReporter(lambda f, m: seen.append(f)))
    assert seen and seen[-1] == 1.0
    with pytest.raises(TypeError):
        run_job(make_job("depth", [gradient_photo]), progress=42)  # type: ignore[arg-type]


def test_folder_input_is_rejected_for_photo_modes(make_job, tmp_path):
    folder = tmp_path / "album"
    folder.mkdir()
    with pytest.raises(InputError, match="is a folder") as ei:
        run_job(make_job("stereo", [folder]))
    assert "Capture" in ei.value.hint


def test_scene_title_keeps_the_readable_name(make_job, gradient_photo, check_scene):
    res = run_job(make_job("stereo", [gradient_photo], common_extra={"name": "Beach: day 1"}))
    assert Path(res.outputs["3D photo (side-by-side)"]).name == "Beach_ day 1_LR.jpg"
    assert check_scene(res, "stereo")["title"] == "Beach: day 1"


def test_failed_job_releases_reserved_names(make_job, gradient_photo, monkeypatch, out_dir):
    def fail(*a, **k):
        raise InputError("disk full")

    monkeypatch.setattr(R, "_save_jpeg_or_png", fail)
    with pytest.raises(InputError):
        run_job(make_job("stereo", [gradient_photo]))
    monkeypatch.undo()
    res = run_job(make_job("stereo", [gradient_photo], common_extra={"export_scene": False}))
    assert Path(res.outputs["3D photo (side-by-side)"]).name == "photo_LR.jpg"
