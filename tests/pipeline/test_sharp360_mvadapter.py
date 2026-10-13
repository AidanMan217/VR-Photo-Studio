"""SHARP 360 with MV-Adapter's AI views and the Brush polish, end to end (docs/design/SHARP360.md
5.3; M4 / M5).

A fake SHARP add-on returns a SHARP-like cloud of the analytic synthetic person
(tests/geometry/synth_person.py) and a fake MV-Adapter add-on returns
``synth_person.mvadapter_like_set``: exact orthographic renders of the same person at
MV-Adapter's six azimuths in the framing of the levelled input the app gave it, with the
``cameras.json`` the real worker writes.  So the levelled input, the view cache, the
orthographic lifting, the stitching, the export and the previews all run for real.  The Brush
polish runs the fake Brush (tests/addons/fakes/fake_brush.py), which "trains" by echoing
``init.ply``.  No GPU, no torch.
"""

from __future__ import annotations

import functools
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vrphoto.core.errors import AddonError, Cancelled, InputError
from vrphoto.core.types import CancelToken, ProgressReporter
from vrphoto.pipeline import context, view_cache
from vrphoto.pipeline import sharp360 as S
from vrphoto.pipeline.run import run_job

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geometry"))
import synth_person as sp  # noqa: E402

FAKES = Path(__file__).resolve().parents[1] / "addons" / "fakes"
W, H, HFOV, G = 240, 320, 40.0, 128  # as test_run_sharp360.py
MAGENTA = np.array([255, 0, 255], np.uint8)  # the bag, painted in the AI views to track its side
VIEW_PX = 256  # the fake worker's pictures (the real one makes 512; the app reads any square size)
TESTED = {"ok": True, "sh0": True, "masks": True, "sh_option": True}


@functools.lru_cache(maxsize=1)
def _scene() -> sp.SharpScene:
    return sp.sharp_like_scene(sp.make_body(), sp.make_cam(W, H, HFOV, grid=G), G=G)


class FakeManager:
    """``vrphoto.addons.manager.status/run`` for the tests: ``handlers`` maps add-on id ->
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
            progress(0.5, "Generating the views (step 10/20)" if addon_id == "mvadapter" else "working")
        return self.handlers[addon_id](task, inputs, params, Path(out_dir))


def _sharp_handler(task, inputs, params, out_dir):
    from vrphoto.geometry.splat import write_splats

    scene = _scene()
    assert task == "image_to_splat" and Path(inputs["image"]).is_file()
    p = write_splats(scene.sc, out_dir / "model.ply")
    cam = scene.cam
    return {"ply": str(p), "frame": "opencv", "f_px": cam.f, "width": cam.width, "height": cam.height}


@pytest.fixture
def events():
    return []


@pytest.fixture
def levelled(monkeypatch):
    """Records the levelled input the app makes (``ortho``) and the frame it was made in
    (``canon``): what an ideal generator turns around."""
    from vrphoto.geometry import views_lift
    from vrphoto.pipeline.cache import content_hash

    seen: dict = {"calls": 0, "digests": []}
    real = views_lift.levelled_ortho_input

    def spy(rgb, alpha, front, canon, **kw):
        oi = real(rgb, alpha, front, canon, **kw)
        seen.update(ortho=oi, canon=canon, calls=seen["calls"] + 1)
        seen["digests"].append(content_hash(np.ascontiguousarray(oi.rgba)))
        return oi

    monkeypatch.setattr(views_lift, "levelled_ortho_input", spy)
    return seen


def mv_handler(levelled: dict, *, garbage: bool = False, **perturb):
    """A fake MV-Adapter worker: one ``mvadapter_like_set`` per requested seed, written like
    the real worker (``<out>/<name>_<seed>/view_<k>.png`` + ``cameras.json``).  ``garbage``
    replaces every picture by a grey disc (a generator that lost the subject)."""

    def handle(task, inputs, params, out_dir):
        from PIL import Image

        assert task == "image_to_views"
        with Image.open(inputs["image"]) as im:
            assert im.size == (512, 512) and im.mode == "RGBA"
        sets = []
        for seed in params["seeds"]:
            mv = sp.mvadapter_like_set(
                sp.make_body(),
                levelled["canon"],
                levelled["ortho"],
                size=VIEW_PX,
                paint={"bag": MAGENTA},
                **perturb,
            )
            if garbage:
                yy, xx = np.mgrid[0:VIEW_PX, 0:VIEW_PX]
                disc = np.hypot(xx - VIEW_PX / 2, yy - VIEW_PX / 2) < VIEW_PX / 4
                for name in mv.images:
                    mv.images[name] = np.where(disc[..., None], 200, 128).astype(np.uint8).repeat(3, 2)
            mv.cameras.update(seed=seed, worker_version="1.0")  # as the real worker's cameras.json
            folder = out_dir / f"{params.get('name', 'views')}_{seed}"
            cams = mv.write(folder)
            sets.append({"seed": seed, "views": [str(folder / n) for n in mv.images], "cameras": str(cams)})
        return {"sets": sets, "frame": "opencv", "warnings": [], "offload": "model", "worker_version": "1.0"}

    return handle


@pytest.fixture
def person(tmp_path, monkeypatch, events, levelled):
    """``person(installed=("sharp", "mvadapter"), mvadapter=None, trellis=None) -> (photo,
    FakeManager)``: the synthetic photo, a segmenter returning its matte (and keying the AI
    views' grey background), the fake SHARP and the fake MV-Adapter (``mvadapter_like_set``)."""
    from PIL import Image

    from vrphoto.addons import manager

    def make(installed=("sharp", "mvadapter"), mvadapter=None, trellis=None):
        scene = _scene()
        path = tmp_path / "photo.png"
        Image.fromarray(scene.rgb).save(path)

        class Seg:
            id = "birefnet-lite"

            def segment(self, img, *, rect=None, progress=None, cancel=None):
                events.append("segment")
                if img.shape[:2] != scene.alpha.shape:  # an AI view on its grey background
                    return (np.abs(img.astype(np.int16) - 128).max(axis=2) > 15).astype(np.float32)
                return scene.alpha.copy()

        seg = Seg()
        monkeypatch.setattr(context.RunContext, "segmenter", lambda self, *a, **k: seg)
        handlers = {"sharp": _sharp_handler, "mvadapter": mvadapter or mv_handler(levelled)}
        if trellis is not None:
            handlers["trellis"] = trellis
        fm = FakeManager(installed, handlers, events)
        monkeypatch.setattr(manager, "status", fm.status)
        monkeypatch.setattr(manager, "run", fm.run)
        return path, fm

    return make


@pytest.fixture(autouse=True)
def small_previews(monkeypatch):
    """Small turntable previews (tested in test_run_sharp360.py)."""
    for name, q in list(S.QUALITY.items()):
        monkeypatch.setitem(S.QUALITY, name, replace(q, preview_px=64))


def _job(make_job, photo, **kw):
    kw.setdefault("common_extra", {"export_scene": False})
    kw.setdefault("back_source", "mvadapter")
    kw.setdefault("back_quality", "fast")  # the quickest mould: what is tested here is the source
    return make_job("object", [photo], method="sharp360", height_m=1.6, save_extras=True, **kw)


def _outputs(res) -> dict[str, tuple[str, Path]]:
    return {Path(p).name: (k, Path(p)) for k, p in res.outputs.items()}


def _debug(res) -> dict:
    """The job's debug.json (a re-run's outputs are named ``photo (2)_...``)."""
    path = next(
        Path(p) for p in res.outputs.values() if re.fullmatch(r"photo.*_sharp360_debug.*\.json", Path(p).name)
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _splat_key(res) -> str:
    return next(k for k in res.outputs if k.startswith("3D object (Gaussian splat"))


def _fallbacks(res) -> list[str]:
    return [w for w in res.warnings if not w.startswith("Note:")]


def _far_sides(path: Path) -> dict[str, float]:
    """Share of each side colour among the opaque Gaussians of the far quarter of an exported
    splat (lowest z: the subject faces +z) - the true body there is mostly its blue back.  The
    bag, painted magenta in the AI views, is left out (it would count as the red front)."""
    from vrphoto.geometry.splat import read_splats

    sc = read_splats(path)
    z = sc.means[:, 2]
    bag = np.abs(sc.colors - MAGENTA / 255.0).max(axis=1) < 0.25
    names = sp.classify_rgb(sc.colors[(z < np.percentile(z, 25)) & (sc.opacity > 0.5) & ~bag], sp.SIDES)
    return {n: float((names == n).mean()) for n in sp.SIDES}


# --------------------------------------------------------------------------- the MV-Adapter back
def test_mvadapter_back_end_to_end_and_the_view_cache(make_job, person, events, monkeypatch, levelled):
    from pipeline_helpers import capture_stitch, front_render_difference, generated_on_the_front

    from vrphoto.models import ort

    unloads: list[bool] = []

    def spy(model_id=None, *, wait=True):
        unloads.append(wait)
        events.append("unload")

    monkeypatch.setattr(ort, "unload_model", spy)
    seen = capture_stitch(monkeypatch)
    # the photo is 320 px long: a coarser "fast" mould grid makes the levelled picture differ
    # between the two qualities used here, as it does for real photos (512 / 768 / 1024 cells)
    monkeypatch.setitem(S.QUALITY, "fast", replace(S.QUALITY["fast"], mould_long=200))
    monkeypatch.setitem(
        S.QUALITY,
        "balanced",
        replace(S.QUALITY["balanced"], mould_long=320, stride=2, fill_holes=False, relief="flat"),
    )
    photo, fm = person()
    res = run_job(_job(make_job, photo, seed=4, back_steps=20))

    # calls: SHARP, then MV-Adapter, each after the app's own models were released
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "mvadapter"]
    runs = [i for i, e in enumerate(events) if e.startswith("run:")]
    assert [events[i - 1] for i in runs] == ["unload", "unload"], events
    call = fm.calls[1]
    assert call["task"] == "image_to_views"
    assert call["params"] == {
        "seeds": [4],
        "steps": 20,  # "fast"
        "guidance": 3.0,
        "prompt": S.MV_DEFAULT_PROMPT,
        "name": "views",
    }
    assert Path(call["inputs"]["image"]).name == "mvadapter_input.png"
    assert levelled["calls"] == 1

    # the back is MV-Adapter's: blue from behind, the bag on the body's right side
    assert not _fallbacks(res), res.warnings
    assert _splat_key(res) == "3D object (Gaussian splat, SHARP 360°, back by MV-Adapter)"
    out = _outputs(res)
    sides = _far_sides(out["photo_sharp360_back.splat"][1])
    assert sides["back"] > 0.6 and sides["front"] < 0.05, sides
    part = seen["part"]
    assert part.source == "mvadapter" and part.ok
    mag = np.abs(part.sc.colors - MAGENTA / 255.0).max(axis=1) < 0.25
    assert mag.sum() > 30
    bag_x = sp.make_body().to_B(np.asarray(part.sc.means)[mag])[:, 0]
    assert np.median(bag_x) < 0, "the bag is on the body's right (x_B < 0)"
    # the photographed front is untouched (design 5.2)
    diff, n = front_render_difference(seen)
    assert n > 5000 and diff < 2.0, diff
    share, n_gen = generated_on_the_front(seen)
    assert n_gen > 1000 and share < 0.02, share

    # debug data, contact sheet and the editable views folder
    dbg = _debug(res)
    assert dbg["options"]["sources"] == ["mvadapter", "shell"]
    assert dbg["sources"] == [dict(dbg["sources"][0], source="mvadapter", ok=True)]
    d = dbg["back"]["diag"]
    assert d["cameras"] == "orthographic" and d["chosen"] == 1 and d["seed"] == 4
    assert d["cache"]["hits"] == 0 and d["steps"] == 20 and d["worker"]["offload"] == "model"
    assert [t["ok"] for t in d["tries"]] == [True] and d["qc"]["cost"] >= 0
    label, sheet = out["photo_sharp360_views.png"]
    assert label == "SHARP 360° AI views as lined up (PNG)" and sheet.is_file()
    assert dbg["previews"]["views"]["views"] >= 5
    label, folder = out["photo_sharp360_views"]
    assert label.startswith("SHARP 360° AI views folder") and (folder / "back.png").is_file()
    doc = json.loads((folder / "views.json").read_text(encoding="utf-8"))
    assert doc["projection"] == "orthographic" and {v["file"] for v in doc["views"]} >= {
        "back.png",
        "right.png",
        "left.png",
    }
    assert dbg["previews"]["views_folder"] == "photo_sharp360_views"
    assert dbg["polish"] == dict(dbg["polish"], asked="auto", run=False)  # "fast": no polish
    entries = [p for p in view_cache.default_root().iterdir() if not p.name.startswith(".")]
    assert len(entries) == 1

    # a re-run with another seam and another back quality ('AI steps' set): the views come
    # from the cache although the levelled picture differs a little (another mould grid)
    fm.calls.clear()
    res2 = run_job(_job(make_job, photo, seed=4, back_steps=20, seam_deg=55, back_quality="balanced"))
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]
    assert levelled["calls"] == 2 and levelled["digests"][0] != levelled["digests"][1]
    assert _splat_key(res2) == "3D object (Gaussian splat, SHARP 360°, back by MV-Adapter)"
    d2 = _debug(res2)["back"]["diag"]
    assert (
        d2["cache"]["hits"] == 1
        and d2["tries"][0]["cached"] is True
        and d2["cache"]["input"] == d["cache"]["input"]
    )
    assert not _fallbacks(res2), res2.warnings


def test_mvadapter_tries_are_one_worker_call_and_the_best_set_is_kept(make_job, person, levelled):
    """'AI attempts' 2: one worker call with two seeds (the model loads once); the set with
    the lower QC cost wins - here the second: in the first, the left view came out 9 % too
    big and is dropped."""
    calls = {"n": 0}
    good = mv_handler(levelled)
    off = mv_handler(levelled, view_scale={270.0: 1.09})

    def handle(task, inputs, params, out_dir):
        calls["n"] += 1
        outs = {"sets": [], "frame": "opencv", "warnings": ["Repeated seeds were generated only once."]}
        for seed in params["seeds"]:
            h = off if seed == 0 else good
            outs["sets"] += h(task, inputs, {**params, "seeds": [seed]}, out_dir)["sets"]
        return outs

    photo, fm = person(mvadapter=handle)
    progress: list[tuple[float, str]] = []
    res = run_job(_job(make_job, photo, back_tries=2), progress=lambda f, m: progress.append((f, m)))
    # the worker's step messages, in the back's range (design 1.4)
    gen = [(f, m) for f, m in progress if m.startswith("Generating the back and sides with MV-Adapter")]
    assert "Generating the back and sides with MV-Adapter (step 10/20)" in [m for _, m in gen], progress
    assert all(0.30 <= f <= 0.72 for f, _ in gen)
    assert any(m == "Lining up the AI views (set 2/2)" for _, m in progress)
    assert [c["params"]["seeds"] for c in fm.calls if c["addon_id"] == "mvadapter"] == [[0, 1]]
    d = _debug(res)["back"]["diag"]
    assert [t["seed"] for t in d["tries"]] == [0, 1] and all(t["ok"] for t in d["tries"])
    assert d["tries"][1]["qc"] < d["tries"][0]["qc"] and d["chosen"] == 2 and d["seed"] == 1
    # the worker's own notes reach the job, named
    assert "MV-Adapter (AI back and side views, experimental): Repeated seeds were generated only once." in (
        res.warnings
    )


def test_mvadapter_out_of_memory_falls_back_to_trellis_then_the_shell(make_job, person, events):
    def oom(task, inputs, params, out_dir):
        raise AddonError("CUDA out of memory")

    def trellis_crash(task, inputs, params, out_dir):
        raise AddonError("TRELLIS stopped unexpectedly")

    photo, fm = person(installed=("sharp", "mvadapter", "trellis"), mvadapter=oom, trellis=trellis_crash)
    res = run_job(_job(make_job, photo, back_source="auto", back_quality="balanced"))
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "mvadapter", "trellis"]
    assert _fallbacks(res) == [
        "MV-Adapter could not make the back (CUDA out of memory); TRELLIS was used instead.",
        "TRELLIS could not make the back (TRELLIS stopped unexpectedly); the built-in back was used instead.",
    ]
    assert "built-in back" in _splat_key(res)
    dbg = _debug(res)
    assert [(a["source"], a["ok"]) for a in dbg["sources"]] == [
        ("mvadapter", False),
        ("trellis", False),
        ("shell", True),
    ]
    assert "photo_sharp360_views.png" not in _outputs(res)  # no views were made
    assert "photo_sharp360_views" not in _outputs(res)


def test_a_garbage_view_set_is_rejected_by_the_gates(make_job, person, levelled):
    photo, fm = person(mvadapter=mv_handler(levelled, garbage=True))
    res = run_job(_job(make_job, photo))
    fb = _fallbacks(res)
    assert len(fb) == 1 and fb[0].startswith("MV-Adapter could not make a usable back ("), fb
    assert "outline" in fb[0] and fb[0].endswith("; the built-in back was used instead."), fb
    assert "built-in back" in _splat_key(res)
    dbg = _debug(res)
    att = dbg["sources"][0]
    assert (att["source"], att["ok"]) == ("mvadapter", False) and att["diag"]["tries"][0]["ok"] is False
    # the contact sheet and the views folder still show what the AI made (to fix and re-run)
    out = _outputs(res)
    assert "photo_sharp360_views.png" in out and "photo_sharp360_views" in out


def test_mvadapter_not_installed_falls_back(make_job, person):
    photo, fm = person(installed=("sharp",))
    res = run_job(_job(make_job, photo))
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]
    assert _fallbacks(res) == ["MV-Adapter is not installed; the built-in back was used instead."]


def test_a_cancel_while_mvadapter_runs_propagates_and_keeps_finished_sets(make_job, person, levelled):
    token = CancelToken()
    good = mv_handler(levelled)

    def cancelled(task, inputs, params, out_dir):
        # the worker writes each set as soon as it is made: the first seed's set is on disk
        # when the user cancels during the second
        good(task, inputs, {**params, "seeds": params["seeds"][:1]}, out_dir)
        token.cancel()
        raise Cancelled("Operation cancelled")

    photo, fm = person(mvadapter=cancelled)
    with pytest.raises(Cancelled):
        run_job(_job(make_job, photo, back_tries=2), cancel=token)
    entries = [p for p in view_cache.default_root().iterdir() if not p.name.startswith(".")]
    assert len(entries) == 1  # salvaged: the next run does not generate it again


# --------------------------------------------------------------------------- the source on its own
@functools.lru_cache(maxsize=4)
def _front_and_frame(mould_long: int):
    """SHARP's cleaned front and the upright frame of the synthetic person (made once)."""
    from vrphoto.geometry.splat_fusion import canonical_frame, sharp_front

    scene = _scene()
    front = sharp_front(scene.sc, scene.alpha, scene.cam, scene.rgb, mould_long=mould_long)
    return front, canonical_frame(front, scene.sc, scene.alpha, thickness=0.11)


def _stub_lift(front, canon, ortho, views, *, source, **kw):
    """``view_based_back`` stand-in: what the cache tests check happens before lifting."""
    from vrphoto.geometry.back_mould import BackPart

    return BackPart(
        sc=S._empty_cloud(),
        kind="mould",
        source=source,
        ok=True,
        views=list(views),
        diag={"qc": {"cost": 0.1}},
    )


def quick_mv_handler(levelled: dict):
    """:func:`mv_handler` for the tests with the lifting stubbed: the six pictures are rendered
    once and written again for every seed (what is tested is which seeds are asked for)."""
    made: dict = {}

    def handle(task, inputs, params, out_dir):
        assert task == "image_to_views"
        if "set" not in made:
            made["set"] = sp.mvadapter_like_set(
                sp.make_body(), levelled["canon"], levelled["ortho"], size=64, ss=1
            )
        mv = made["set"]
        sets = []
        for seed in params["seeds"]:
            mv.cameras.update(seed=seed, worker_version="1.0")
            folder = out_dir / f"{params.get('name', 'views')}_{seed}"
            cams = mv.write(folder)
            sets.append({"seed": seed, "views": [str(folder / n) for n in mv.images], "cameras": str(cams)})
        return {"sets": sets, "frame": "opencv", "warnings": [], "offload": "model", "worker_version": "1.0"}

    return handle


@pytest.fixture
def direct(make_job, person, settings, monkeypatch, levelled):
    """``direct(mvadapter=None) -> (call, fm)``: ``call(mould_long=320, **opts) -> BackPart``
    runs ``_src_mvadapter`` alone on a prepared context (SHARP and the front made once, the
    lifting stubbed, a 512 px levelled reference), so the view cache and the worker call are
    tested without whole jobs."""
    from types import SimpleNamespace

    from vrphoto.geometry import views_lift
    from vrphoto.pipeline import run as R

    monkeypatch.setattr(views_lift, "view_based_back", _stub_lift)
    monkeypatch.setattr(S, "_LEVELLED_REF_PX", S._LEVELLED_PX)

    def make(mvadapter=None):
        photo, fm = person(mvadapter=mvadapter or quick_mv_handler(levelled))
        spec = _job(make_job, photo)
        run = R._Run(spec, context.RunContext(settings), CancelToken())
        item = R._Item(0, photo, "photo")
        scene = _scene()
        base = S.resolve_options(spec, installed=bool)

        def call(mould_long=320, **kw):
            front, canon = _front_and_frame(mould_long)
            o = replace(base, **kw)
            ctx = S.Ctx(run, item, SimpleNamespace(rgb=scene.rgb), scene.alpha, scene.sc, front, canon, o)
            return S._src_mvadapter(ctx, ProgressReporter.null())

        return call, fm

    return make


def _seeds_asked(fm) -> list[list[int]]:
    return [c["params"]["seeds"] for c in fm.calls if c["addon_id"] == "mvadapter"]


def _entries() -> list[Path]:
    return [p for p in view_cache.default_root().iterdir() if not p.name.startswith(".")]


def test_the_view_cache_key(direct, levelled):
    """The views depend on the photo, the settings, the seed and the description - not on the
    back quality's mould grid, the seam or the colour match; a third attempt only makes the
    new seed."""
    call, fm = direct()
    assert call(seed=4).ok and _seeds_asked(fm) == [[4]] and len(_entries()) == 1
    # another mould grid (another back quality) makes a slightly different levelled picture ...
    part = call(mould_long=200, seed=4, seam=55.0, color_match=False)
    assert levelled["digests"][0] != levelled["digests"][1]
    # ... and still reuses the views
    assert part.ok and _seeds_asked(fm) == [[4]] and part.diag["cache"]["hits"] == 1
    call(seed=5)
    call(seed=4, back_prompt="red jacket")
    asked = [(c["params"]["seeds"], c["params"]["prompt"]) for c in fm.calls if c["addon_id"] == "mvadapter"]
    assert asked[1:] == [([5], S.MV_DEFAULT_PROMPT), ([4], "red jacket")]
    part = call(seed=4, tries=3)
    assert _seeds_asked(fm)[-1] == [6] and [t["cached"] for t in part.diag["tries"]] == [True, True, False]
    assert len(_entries()) == 4


def test_sets_finished_before_a_worker_error_are_used(direct, levelled, monkeypatch):
    """The worker writes each set as soon as it is made: when a later seed fails (out of
    memory), the finished set makes the back of THIS run (with a note) and is cached."""
    good = quick_mv_handler(levelled)

    def first_then_oom(task, inputs, params, out_dir):
        good(task, inputs, {**params, "seeds": params["seeds"][:1]}, out_dir)
        raise AddonError("CUDA out of memory")

    call, fm = direct(mvadapter=first_then_oom)
    part = call(tries=2)
    assert part.ok and part.source == "mvadapter" and part.diag["chosen"] == 1
    assert [t.get("ok") for t in part.diag["tries"]] == [True, False]
    assert part.diag["tries"][1]["error"] == "CUDA out of memory"
    assert part.diag["warning"] == (
        "MV-Adapter could not make attempt 2 (CUDA out of memory); the views made before were used."
    )
    assert len(_entries()) == 1
    # with the cache unable to store, the worker's own files are used
    monkeypatch.setattr(view_cache.ViewCache, "put", lambda self, key, files, meta=None: None)
    part = call(seed=10, tries=2)
    assert part.ok and part.diag["chosen"] == 1 and part.diag["tries"][0]["cached"] is False
    # nothing finished at all: the add-on's error goes to the fallback chain
    monkeypatch.undo()

    def oom(task, inputs, params, out_dir):
        raise AddonError("CUDA out of memory")

    call, fm = direct(mvadapter=oom)
    with pytest.raises(AddonError, match="out of memory"):
        call(seed=20)


def test_a_damaged_cached_set_is_made_again(direct):
    """A cached picture truncated by a crash is not a dead end: the entry is deleted and the
    views are generated again in the same run."""
    call, fm = direct()
    assert call(seed=4).ok
    (entry,) = _entries()
    (entry / "view_3.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 100)
    part = call(seed=4)
    assert part.ok and _seeds_asked(fm) == [[4], [4]] and not part.diag.get("warning")
    assert part.diag["tries"][0]["cached"] is False
    (entry,) = _entries()
    from PIL import Image

    with Image.open(entry / "view_3.png") as im:
        im.load()  # whole again
    assert call(seed=4).diag["cache"]["hits"] == 1 and len(_seeds_asked(fm)) == 2


def test_the_cache_digest_ignores_the_mould_grid_but_not_the_photo():
    from types import SimpleNamespace

    scene = _scene()
    f1, c1 = _front_and_frame(320)
    f2, c2 = _front_and_frame(200)

    def ctx(front, canon, rgb=scene.rgb, alpha=scene.alpha, sc=scene.sc):
        return S.Ctx(None, None, SimpleNamespace(rgb=rgb), alpha, sc, front, canon, None)

    d = S._mv_input_digest(ctx(f1, c1))
    assert d == S._mv_input_digest(ctx(f2, c2))  # another back quality's mould grid
    # SHARP's GPU result need not be bit-identical between runs: float jitter keeps the key
    jitter = np.random.default_rng(0).normal(0.0, 1e-6, scene.sc.means.shape)
    jittered = replace(scene.sc, means=(np.asarray(scene.sc.means) + jitter).astype(np.float32))
    assert S._mv_input_digest(ctx(f1, c1, sc=jittered)) == d
    rgb = scene.rgb.copy()
    rgb[0, 0] ^= 1
    alpha = scene.alpha.copy()
    alpha[0, 0] = 0.5
    deeper = replace(scene.sc, means=(np.asarray(scene.sc.means) * 1.05).astype(np.float32))
    for other in (ctx(f1, c1, rgb=rgb), ctx(f1, c1, alpha=alpha), ctx(f1, c1, sc=deeper)):
        assert S._mv_input_digest(other) != d
    f3 = replace(f1, cam=replace(f1.cam, f=f1.cam.f * 1.1))  # another field of view
    assert S._mv_input_digest(ctx(f3, c1)) != d


# --------------------------------------------------------------------------- options
def _spec(**kw):
    from vrphoto.pipeline.jobs import ObjectJob

    return ObjectJob(inputs=["x.jpg"], method="sharp360", **kw)


@pytest.mark.parametrize(
    ("installed", "quality", "want"),
    [
        ({"mvadapter", "trellis"}, "balanced", ("mvadapter", "trellis", "shell")),
        ({"mvadapter", "trellis"}, "best", ("mvadapter", "trellis", "shell")),
        ({"mvadapter", "trellis"}, "fast", ("shell",)),
        ({"mvadapter"}, "balanced", ("mvadapter", "shell")),
        ({"trellis"}, "best", ("trellis", "shell")),
        (set(), "balanced", ("shell",)),
    ],
)
def test_auto_order_uses_the_installed_ai_backs(installed, quality, want):
    o = S.resolve_options(_spec(back_quality=quality), installed=lambda a: a in installed)
    assert o.sources == want
    # an explicit MV-Adapter is tried even when missing (the job warns and falls back)
    o = S.resolve_options(_spec(back_source="mvadapter", back_quality=quality), installed=lambda a: False)
    assert o.sources == ("mvadapter", "shell")


def test_mvadapter_settings_follow_the_quality():
    """design 1.3: steps 20 / 30 / 50, guidance 3.0, tries 1 / 1 / 2; the job's 'AI steps' /
    'AI attempts' win; the worker's 5-step minimum."""
    for quality, steps, tries in (("fast", 20, 1), ("balanced", 30, 1), ("best", 50, 2)):
        o = S.resolve_options(_spec(back_quality=quality, back_source="mvadapter"), installed=bool)
        assert S.mv_settings(o) == (steps, 3.0, S.MV_DEFAULT_PROMPT) and o.tries == tries, quality
    o = S.resolve_options(
        _spec(back_source="mvadapter", back_steps=40, back_tries=3, back_prompt="  grey   hoodie  "),
        installed=bool,
    )
    assert S.mv_settings(o) == (40, 3.0, "grey hoodie") and o.tries == 3
    o = S.resolve_options(
        _spec(back_source="mvadapter", back_steps=2, back_prompt=" grey \n hoodie "), installed=bool
    )
    assert S.mv_settings(o) == (5, 3.0, "grey hoodie")


def test_new_options_are_checked():
    o = S.resolve_options(_spec(), installed=bool)
    assert (o.back_prompt, o.polish, o.polish_steps) == ("", "auto", 5000)
    o = S.resolve_options(
        _spec(polish=" ON ", polish_steps="8000", back_prompt="a  red\tcoat"), installed=bool
    )
    assert (o.polish, o.polish_steps, o.back_prompt) == ("on", 8000, "a red coat")
    # the description is only used (and checked) by the AI view sources
    for source in ("shell", "trellis"):
        assert (
            S.resolve_options(_spec(back_source=source, back_prompt="x" * 400), installed=bool).back_prompt
            == ""
        )
    assert S.PROMPT_SOURCES == ("auto", "mvadapter")
    for kw, match in (
        ({"back_prompt": "x" * 301}, "too long"),
        ({"back_source": "mvadapter", "back_prompt": "x" * 301}, "too long"),
        ({"polish": "always"}, "Brush polish"),
        ({"polish_steps": 999}, "Brush steps"),
        ({"polish_steps": 30001}, "Brush steps"),
        ({"polish_steps": 2500.5}, "Brush steps"),
    ):
        with pytest.raises(InputError, match=match):
            S.resolve_options(_spec(**kw), installed=bool)
    assert S.resolve_options(_spec(back_prompt="x" * 300), installed=bool).back_prompt == "x" * 300


def test_bad_new_options_fail_before_any_work(make_job, person, events):
    photo, fm = person()
    for kw in ({"back_prompt": "y" * 400}, {"polish": "sometimes"}, {"polish_steps": 100}):
        with pytest.raises(InputError):
            run_job(_job(make_job, photo, **kw))
    assert not fm.calls and not events


# --------------------------------------------------------------------------- Brush polish (M5)
@pytest.fixture
def fake_brush(monkeypatch, tmp_path):
    log = tmp_path / "tools.jsonl"
    monkeypatch.setenv("VRPHOTO_BRUSH", str(FAKES / "fake_brush.py"))
    monkeypatch.setenv("FAKE_TOOL_LOG", str(log))
    for k in (
        "FAKE_BRUSH_SLEEP",
        "FAKE_BRUSH_FAIL",
        "FAKE_BRUSH_PLY",
        "FAKE_BRUSH_ECHO_INIT",
        "FAKE_BRUSH_FAIL_SH0",
        "FAKE_BRUSH_OOM_ABOVE",
        "FAKE_BRUSH_REJECT_FLAGS",
        "FAKE_BRUSH_ECHO_SCALE",
        "FAKE_BRUSH_ECHO_NAN",
    ):
        monkeypatch.delenv(k, raising=False)

    def calls() -> list[list[str]]:
        lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return [c for c in map(json.loads, lines) if c[0] == "brush" and "--total-steps" in c]

    return calls


def _spy_polish(monkeypatch) -> list[dict]:
    from vrphoto.pipeline import sharp360_polish

    seen: list[dict] = []
    real = sharp360_polish.polish_with_brush

    def spy(run, item, merged, is_front, *a, **kw):
        out, out_front, diag = real(run, item, merged, is_front, *a, **kw)
        seen.append(
            dict(merged=merged, is_front=np.asarray(is_front), out=out, out_front=out_front, diag=diag, kw=kw)
        )
        return out, out_front, diag

    monkeypatch.setattr(sharp360_polish, "polish_with_brush", spy)
    return seen


def test_polish_on_trains_with_brush_and_keeps_the_front_exact(
    make_job, person, fake_brush, monkeypatch, settings
):
    from pipeline_helpers import capture_stitch

    monkeypatch.setenv("FAKE_BRUSH_ECHO_INIT", "1")
    settings.brush_test = dict(TESTED)
    seen = _spy_polish(monkeypatch)
    stitched = capture_stitch(monkeypatch)
    progress: list[tuple[float, str]] = []
    photo, fm = person()
    res = run_job(
        _job(make_job, photo, polish="on", polish_steps=1500),
        progress=ProgressReporter(lambda f, m: progress.append((f, m))),
    )
    assert not _fallbacks(res), res.warnings
    assert len(fake_brush()) == 1 and "1500" in fake_brush()[0]
    assert len(seen) == 1
    s = seen[0]
    assert s["diag"]["ok"] and s["kw"]["steps"] == 1500 and s["kw"]["quality"] == "fast"
    # the exact kept SHARP rows come first, bit for bit, in order
    F = np.asarray(s["merged"].means)[s["is_front"]]
    out_front = np.asarray(s["out_front"], dtype=bool)
    assert out_front[: len(F)].all() and not out_front[len(F) :].any()
    for field in ("means", "scales", "quats", "colors", "opacity"):
        a = np.asarray(getattr(s["merged"], field))[s["is_front"]]
        b = np.asarray(getattr(s["out"], field))[out_front]
        assert np.array_equal(a, b), field
    # the stitched result IS the polished one (no second floater pass removed front rows)
    assert len(stitched["merged"]) == len(s["out"])
    dbg = _debug(res)
    assert dbg["polish"]["asked"] == "on" and dbg["polish"]["run"] and dbg["polish"]["ok"]
    assert dbg["floaters"] == s["diag"]["floaters"]
    # progress: Brush polishes in 0.82-0.92 (design 1.4), the export follows
    brush = [f for f, m in progress if "Brush" in m]
    assert brush and 0.819 <= min(brush) and max(brush) <= 0.921, brush
    later = [f for f, m in progress if m == "Making it life-size"]
    assert later and min(later) >= 0.92


def test_failed_polish_keeps_the_unpolished_result(make_job, person, fake_brush, monkeypatch, settings):
    monkeypatch.setenv("FAKE_BRUSH_FAIL", "1")
    settings.brush_test = dict(TESTED)
    photo, fm = person()
    res = run_job(_job(make_job, photo, polish="on", polish_steps=1000))
    fb = _fallbacks(res)
    assert len(fb) == 1 and fb[0].startswith("Brush polish did not work ("), fb
    assert "the unpolished result was kept" in fb[0]
    assert _splat_key(res) == "3D object (Gaussian splat, SHARP 360°, back by MV-Adapter)"
    dbg = _debug(res)
    assert dbg["polish"]["run"] and dbg["polish"]["ok"] is False
    assert dbg["floaters"]["gen_dropped"] >= 0  # the normal floater pass ran instead


@pytest.fixture
def brush_stub(monkeypatch) -> list[dict]:
    """``polish_with_brush`` replaced by a stub that records its calls and keeps the result."""
    from vrphoto.pipeline import sharp360_polish

    calls: list[dict] = []

    def stub(run, item, merged, is_front, *a, **kw):
        calls.append(kw)
        diag = {"ok": False, "reason": "stub", "warning": "Brush polish stub (test)."}
        return merged, np.asarray(is_front), diag

    monkeypatch.setattr(sharp360_polish, "polish_with_brush", stub)
    return calls


def test_automatic_polish_runs_with_best_quality_and_a_passed_brush_test(
    make_job, person, fake_brush, brush_stub, monkeypatch, settings
):
    # only the polish decision matters here: "best" with the "fast" mould
    monkeypatch.setitem(
        S.QUALITY, "best", replace(S.QUALITY["best"], mould_long=512, stride=2, fill_holes=False)
    )
    settings.brush_test = dict(TESTED)
    photo, fm = person()
    res = run_job(_job(make_job, photo, back_quality="best", back_tries=1, polish="auto"))
    assert len(brush_stub) == 1 and brush_stub[0]["quality"] == "best" and brush_stub[0]["steps"] == 5000
    assert _fallbacks(res) == ["Brush polish stub (test)."]
    p = _debug(res)["polish"]
    assert p["asked"] == "auto" and p["run"] is True


def test_the_polish_plan_warns_only_when_the_polish_was_asked_for(fake_brush, monkeypatch):
    """'Automatic' never warns when it skips; 'On' says why it could not polish, or that it
    polishes without a passed self-test (the rules themselves: test_sharp360_polish.py)."""
    from types import SimpleNamespace

    from vrphoto.geometry.back_mould import BackPart
    from vrphoto.geometry.views_lift import GenView

    view = GenView(
        "back",
        180.0,
        np.zeros((8, 8, 3), np.uint8),
        np.ones((8, 8), np.float32),
        kind="ortho",
        meta={"registered": True},
    )
    with_views = BackPart(sc=S._empty_cloud(), kind="mould", source="mvadapter", views=[view])
    shell = BackPart(sc=S._empty_cloud(), kind="mould", source="shell")
    base = S.resolve_options(_spec(), installed=bool)

    def plan(polish, quality, part, bt):
        warnings: list[str] = []
        run = SimpleNamespace(settings=SimpleNamespace(brush_test=bt), warn=warnings.append)
        o = replace(base, polish=polish, quality=quality)
        dbg: dict = {}
        ok, _note = S._polish_plan(S.Ctx(run, None, None, None, None, None, None, o), part, dbg)
        assert dbg["polish"]["asked"] == polish and dbg["polish"]["run"] is ok
        return ok, warnings, dbg["polish"]["note"]

    ok, warns, note = plan("auto", "best", with_views, {})
    assert not ok and not warns and "not been tested" in note  # skipped quietly
    assert plan("auto", "best", with_views, TESTED)[:2] == (True, [])
    assert plan("auto", "fast", with_views, TESTED)[:2] == (False, [])
    assert plan("off", "best", with_views, TESTED)[:2] == (False, [])
    assert plan("on", "fast", shell, TESTED)[1] == [
        "Brush polish was skipped (only a back made from views (MV-Adapter or your back photo) can be polished)."
    ]
    ok, warns, _ = plan("on", "fast", with_views, {})
    assert (
        ok
        and len(warns) == 1
        and warns[0].startswith("Brush polish was run although Brush has not been tested")
    )
    monkeypatch.delenv("VRPHOTO_BRUSH")  # the Capture add-on is missing
    assert plan("on", "fast", with_views, TESTED)[:2] == (
        False,
        ["Brush polish was skipped (the Capture add-on (Brush) is not installed)."],
    )
