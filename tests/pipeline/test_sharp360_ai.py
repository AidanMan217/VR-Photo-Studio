"""SHARP 360 with the AI photo-editor backs (M7: the Qwen photo editor add-on, your own ComfyUI
server) and the SHARP relief (M6), end to end (docs/design/SHARP360.md 3.5.6, 4.4, 5.3).

A fake SHARP add-on returns a SHARP-like cloud of the analytic synthetic person
(tests/geometry/synth_person.py).  The "AI" views are exact orthographic renders of the same
person turned round (180 / 90 / 270 deg) on light grey (#c8c8c8), each framed differently -
what an ideal photo editor would return - written by a fake ``qwenedit`` worker or served by
the fake ComfyUI server (tests/addons/fakes/fake_comfyui.py).  The bag at the body's right
hip is painted magenta in them to check it ends up on the correct side.  A "face" is a
skin-coloured head, which the monkeypatched face detector finds.  No GPU, no torch.
"""

from __future__ import annotations

import functools
import json
import math
import re
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vrphoto.core.errors import AddonError, Cancelled, InputError
from vrphoto.core.types import CancelToken
from vrphoto.pipeline import context, view_cache
from vrphoto.pipeline import run as R
from vrphoto.pipeline import sharp360 as S
from vrphoto.pipeline import sharp360_ai as A
from vrphoto.pipeline.run import run_job

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geometry"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addons" / "fakes"))
import synth_person as sp  # noqa: E402
from fake_comfyui import FakeComfyServer  # noqa: E402

W, H, HFOV, G = 240, 320, 40.0, 128  # as test_run_sharp360.py
GREY = np.array(A.CUTOUT_GREY, np.uint8)
MAGENTA = np.array([255, 0, 255], np.uint8)
SKIN = np.array([231, 190, 160], np.uint8)
#: Azimuth -> (metres per pixel, size) of the fake editor's views: each framed differently.
SHOTS = {
    180.0: (0.0055, 384),
    90.0: (0.005, 420),
    270.0: (0.0065, 330),
    135.0: (0.006, 360),
    225.0: (0.006, 360),
}


@functools.lru_cache(maxsize=1)
def _scene() -> sp.SharpScene:
    return sp.sharp_like_scene(sp.make_body(), sp.make_cam(W, H, HFOV, grid=G), G=G)


@functools.lru_cache(maxsize=1)
def _canon():
    return sp.synth_canon(sp.make_body(), sp.make_cam(W, H, HFOV, grid=G))


@functools.lru_cache(maxsize=16)
def _ai_view(phi: float, face: bool = False) -> np.ndarray:
    """What an ideal photo editor makes of the cut-out at azimuth ``phi``: the person turned
    round on #c8c8c8, the bag magenta (and a skin-coloured head for a "face")."""
    body = sp.make_body()
    m, n = SHOTS[float(phi)]
    rgb, _, _ = sp.ortho_view(body, _canon(), phi, m=m, N=n, ss=1, bg=GREY)
    parts = sp.ortho_view_parts(body, _canon(), phi, m=m, N=n)
    rgb[parts == body.part("bag")] = MAGENTA
    if face:
        rgb[parts == body.part("head")] = SKIN
    rgb.flags.writeable = False
    return rgb


def _save_png(rgb: np.ndarray, path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(rgb)).save(path)
    return path


def fake_face(rgb, alpha=None, **kw) -> float:
    """``faces.frontal_face_score`` stand-in: a skin-coloured head is a face."""
    return 0.95 if int(np.all(np.asarray(rgb)[..., :3] == SKIN, axis=-1).sum()) >= 20 else 0.0


class FakeManager:
    """``vrphoto.addons.manager.status/run``; ``handlers`` maps add-on id -> ``fn(task, inputs,
    params, out_dir) -> outputs``."""

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
            progress(0.5, "Making the back view (step 2/4)" if addon_id == "qwenedit" else "working")
        return self.handlers[addon_id](task, inputs, params, Path(out_dir))


def _sharp_handler(task, inputs, params, out_dir):
    from vrphoto.geometry.splat import write_splats

    scene = _scene()
    assert task == "image_to_splat" and Path(inputs["image"]).is_file()
    p = write_splats(scene.sc, out_dir / "model.ply")
    cam = scene.cam
    return {"ply": str(p), "frame": "opencv", "f_px": cam.f, "width": cam.width, "height": cam.height}


def qwen_handler(
    *, face_seeds=(), face_phis=(), fail_sides: str | None = None, fail_backs_after: int | None = None
):
    """A fake ``qwenedit`` worker (task ``views_from_photo``, the real worker's outputs and
    ``views.partial.jsonl``): back views get ``candidates`` tries with seeds ``seed + k``; the
    views of ``face_seeds`` and those at ``face_phis`` show a face."""

    def handle(task, inputs, params, out_dir):
        from PIL import Image

        assert task == "views_from_photo"
        with Image.open(inputs["image"]) as im:
            assert im.mode == "RGBA"  # the plain cut-out (the worker composites it on #c8c8c8)
        partial = out_dir / "views.partial.jsonl"
        partial.unlink(missing_ok=True)
        is_sides = any(abs(float(v["phi_deg"]) - 180.0) > 1 for v in params["views"])
        if is_sides and fail_sides:
            raise AddonError(fail_sides)
        entries = []
        for v in params["views"]:
            phi = float(v["phi_deg"])
            n = int(params["candidates"]) if abs(phi - 180.0) < 22.5 else 1
            for k in range(n):
                if fail_backs_after is not None and len(entries) >= fail_backs_after:
                    raise AddonError("CUDA out of memory")
                seed = int(params["seed"]) + k
                path = _save_png(
                    _ai_view(phi, seed in face_seeds or phi in face_phis),
                    out_dir / f"{params['name']}_{v['name']}_c{k}.png",
                )
                e = {"name": v["name"], "phi_deg": phi, "file": str(path), "candidate": k, "seed": seed}
                entries.append(e)
                with open(partial, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(e) + "\n")
        return {
            "views": entries,
            "frame": "nominal",
            "warnings": [],
            "comfyui_version": "0.39.2-fake",
            "steps": params.get("steps", 4),
            "lightning": True,
            "worker_version": "1.0",
        }

    return handle


def _key_segmenter(scene, events):
    class Seg:
        id = "birefnet-lite"

        def segment(self, img, *, rect=None, progress=None, cancel=None):
            events.append("segment")
            if img.shape[:2] != scene.alpha.shape:  # a view on its flat background: key the border colour
                border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.int16)
                bg = np.median(border, axis=0)
                return (np.abs(img.astype(np.int16) - bg).max(axis=2) > 15).astype(np.float32)
            return scene.alpha.copy()

    return Seg()


@pytest.fixture
def events():
    return []


@pytest.fixture
def person(tmp_path, monkeypatch, events):
    """``person(installed=("sharp", "qwenedit"), qwen=None, sharp=None) -> (photo, FakeManager)``."""
    from PIL import Image

    from vrphoto.addons import manager

    def make(installed=("sharp", "qwenedit"), qwen=None, sharp=None):
        scene = _scene()
        path = tmp_path / "photo.png"
        Image.fromarray(scene.rgb).save(path)
        seg = _key_segmenter(scene, events)
        monkeypatch.setattr(context.RunContext, "segmenter", lambda self, *a, **k: seg)
        handlers = {"sharp": sharp or _sharp_handler, "qwenedit": qwen or qwen_handler()}
        fm = FakeManager(installed, handlers, events)
        monkeypatch.setattr(manager, "status", fm.status)
        monkeypatch.setattr(manager, "run", fm.run)
        return path, fm

    return make


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    """Small previews and a quick "balanced" mould (what is tested here is the source)."""
    for name, q in list(S.QUALITY.items()):
        monkeypatch.setitem(S.QUALITY, name, replace(q, preview_px=64))
    monkeypatch.setitem(
        S.QUALITY,
        "balanced",
        replace(S.QUALITY["balanced"], mould_long=320, stride=2, fill_holes=False, relief="flat"),
    )


@pytest.fixture(scope="module")
def ai_views():
    """The fake editor's pictures rendered once for the module (about 1.5 s; shared setup, so no
    single test carries it)."""
    for phi in (180.0, 90.0, 270.0, 135.0, 225.0):
        _ai_view(phi)
    _ai_view(180.0, True)
    _ai_view(135.0, True)


@pytest.fixture
def faces_found(monkeypatch):
    from vrphoto.segment import faces

    monkeypatch.setattr(faces, "frontal_face_score", fake_face)


def _job(make_job, photo, **kw):
    kw.setdefault("common_extra", {"export_scene": False})
    kw.setdefault("back_source", "qwen")
    kw.setdefault("back_quality", "balanced")
    return make_job("object", [photo], method="sharp360", height_m=1.6, save_extras=True, **kw)


def _outputs(res) -> dict[str, tuple[str, Path]]:
    return {Path(p).name: (k, Path(p)) for k, p in res.outputs.items()}


def _debug(res) -> dict:
    path = next(
        Path(p) for p in res.outputs.values() if re.fullmatch(r"photo.*_sharp360_debug.*\.json", Path(p).name)
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _splat_key(res) -> str:
    return next(k for k in res.outputs if k.startswith("3D object (Gaussian splat"))


def _fallbacks(res) -> list[str]:
    return [w for w in res.warnings if not w.startswith("Note:")]


def _far_sides(path: Path) -> dict[str, float]:
    """Side colours of the opaque Gaussians of the far quarter of an exported splat (the bag,
    magenta in the views, left out)."""
    from vrphoto.geometry.splat import read_splats

    sc = read_splats(path)
    z = sc.means[:, 2]
    bag = np.abs(sc.colors - MAGENTA / 255.0).max(axis=1) < 0.25
    names = sp.classify_rgb(sc.colors[(z < np.percentile(z, 25)) & (sc.opacity > 0.5) & ~bag], sp.SIDES)
    return {n: float((names == n).mean()) for n in sp.SIDES}


def _bag_side(part) -> float:
    """Median body-frame x of the magenta (bag) Gaussians of a back part (< 0: the body's right)."""
    mag = np.abs(np.asarray(part.sc.colors) - MAGENTA / 255.0).max(axis=1) < 0.25
    assert mag.sum() > 20, int(mag.sum())
    return float(np.median(sp.make_body().to_B(np.asarray(part.sc.means)[mag])[:, 0]))


# --------------------------------------------------------------------------- the Qwen photo editor
def test_qwen_back_end_to_end_rejects_a_face(
    make_job, person, events, faces_found, monkeypatch, ai_views
):
    from pipeline_helpers import capture_stitch, front_render_difference

    from vrphoto.models import ort

    monkeypatch.setattr(ort, "unload_model", lambda *a, **k: events.append("unload"))
    seen = capture_stitch(monkeypatch)
    photo, fm = person(qwen=qwen_handler(face_seeds=(3,)))
    progress: list[tuple[float, str]] = []
    res = run_job(
        _job(make_job, photo, seed=3, back_tries=2, back_prompt="grey  coat"),
        progress=lambda f, m: progress.append((f, m)),
    )

    # SHARP, then the back candidates, then the side views with the chosen back as a reference
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "qwenedit", "qwenedit"]
    runs = [i for i, e in enumerate(events) if e.startswith("run:")]
    assert all(events[i - 1] == "unload" for i in runs), events
    backs, sides = fm.calls[1], fm.calls[2]
    assert backs["task"] == "views_from_photo" and Path(backs["inputs"]["image"]).name == "qwen_input.png"
    assert "references" not in backs["inputs"]
    assert backs["params"] == {
        "views": [{"name": "back", "phi_deg": 180.0}],
        "candidates": 2,
        "seed": 3,
        "prompt_extra": "grey coat",
        "long_side": 768,
        "name": "back",
    }
    # candidate 1 (seed 3) shows a face: candidate 2 is the back the side views are made from
    assert [v["phi_deg"] for v in sides["params"]["views"]] == [90.0, 270.0]
    assert sides["params"]["candidates"] == 1 and sides["params"]["seed"] == 3
    assert [Path(p).name for p in sides["inputs"]["references"]] == ["back_back_c1.png"]
    assert any(m == "Qwen photo editor: Making the back view (step 2/4)" for _, m in progress), progress

    # the back is the editor's: blue from behind, the bag on the body's right
    assert _fallbacks(res) == [
        "The Qwen photo editor's back view 1 of 2 was not used (a face was found on the back view)."
    ]
    assert _splat_key(res) == "3D object (Gaussian splat, SHARP 360°, back by the Qwen photo editor)"
    out = _outputs(res)
    sides_far = _far_sides(out["photo_sharp360_back.splat"][1])
    assert sides_far["back"] > 0.6 and sides_far["front"] < 0.05, sides_far
    part = seen["part"]
    assert part.source == "qwen" and part.ok and part.kind == "mould"
    assert _bag_side(part) < 0, "the bag is on the body's right (x_B < 0)"
    diff, n = front_render_difference(seen)
    assert n > 5000 and diff < 2.0, diff

    dbg = _debug(res)
    assert dbg["options"]["sources"] == ["qwen", "shell"] and dbg["options"]["back_prompt"] == "grey coat"
    d = dbg["back"]["diag"]
    assert d["chosen"] == 2 and d["seed"] == 4 and d["input"] == "cutout" and d["sides"] == ["left", "right"]
    c1, c2 = d["candidates"]
    assert (c1["ok"], c1["face"], c1["reason"]) == (False, 0.95, "a face was found on the back view")
    assert c2["ok"] and c2["iou"] >= 0.75 and c2["face"] == 0.0 and c2["cost"] >= 0
    assert d["face"]["max"] == 0.0 and d["qc"]["terms"]["face"] == 0.0
    assert [v["name"] for v in d["views"] if v.get("used")] == ["right", "left"], d["views"]
    assert d["generator"]["made"] == 4 and d["generator"]["cache_hits"] == 0
    # the contact sheet shows the lined-up views and the rejected candidate; the views folder
    label, sheet = out["photo_sharp360_views.png"]
    assert label == "SHARP 360° AI views as lined up (PNG)" and dbg["previews"]["views"]["views"] == 4
    label, folder = out["photo_sharp360_views"]
    assert label.startswith("SHARP 360° AI views folder") and (folder / "back.png").is_file()


def test_qwen_views_are_reused_from_the_view_cache(make_job, person, faces_found, ai_views):
    """A re-run with another seam: every view comes from the view cache (no Qwen at all).  'Fast'
    (back views only) keeps the two runs quick; the side views' keys are checked above."""
    photo, fm = person(qwen=qwen_handler(face_seeds=(3,)))
    res = run_job(_job(make_job, photo, seed=3, back_tries=2, back_quality="fast"))
    d = _debug(res)["back"]["diag"]
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "qwenedit"] and d["chosen"] == 2
    assert d["generator"]["made"] == 2 and not any(c["cached"] for c in d["candidates"])
    fm.calls.clear()
    res2 = run_job(_job(make_job, photo, seed=3, back_tries=2, back_quality="fast", seam_deg=55))
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]
    d2 = _debug(res2)["back"]["diag"]
    assert d2["generator"]["cache_hits"] == 2 and d2["generator"]["made"] == 0 and d2["chosen"] == 2
    assert [c["cached"] for c in d2["candidates"]] == [True, True]
    assert _splat_key(res2) == "3D object (Gaussian splat, SHARP 360°, back by the Qwen photo editor)"


def test_qwen_best_quality_adds_the_back_quarters_and_drops_one_with_a_face(
    make_job, person, faces_found, monkeypatch, ai_views
):
    """'Best' also asks for the 135 / 225 views (deviation 1); a quarter view showing a face is
    not lifted onto the back (design 3.5.6: the face check covers 135 / 180 / 225).  A slow run
    says how slow it was."""
    monkeypatch.setitem(
        S.QUALITY,
        "best",
        replace(S.QUALITY["best"], mould_long=320, stride=2, fill_holes=False, relief="flat", preview_px=64),
    )
    monkeypatch.setattr(A, "SLOW_VIEW_S", 0.0)  # every run counts as slow
    photo, fm = person(qwen=qwen_handler(face_phis=(135.0,)))
    res = run_job(_job(make_job, photo, back_quality="best", back_tries=1))
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "qwenedit", "qwenedit"]
    sides = fm.calls[2]["params"]["views"]
    assert [v["phi_deg"] for v in sides] == [90.0, 270.0, 135.0, 225.0]
    assert [v["name"] for v in sides] == ["right", "left", "back_right", "back_left"]
    assert _splat_key(res) == "3D object (Gaussian splat, SHARP 360°, back by the Qwen photo editor)"
    text = " ".join(_fallbacks(res))  # the source's notes arrive as one warning
    assert "The back right view was not used (it shows a face)." in text, text
    assert "The Qwen photo editor took 0 minutes for 5 views; on a GTX 10-series card" in text, text
    d = _debug(res)["back"]["diag"]
    assert d["sides"] == ["back_left", "back_right", "left", "right"]  # made ...
    used = [v.get("file") for v in d["views"] if v.get("used")]
    assert "sides_back_right_c0.png" not in used and "sides_right_c0.png" in used, d["views"]  # ... not lifted
    assert d["generator"]["made"] == 5


def test_qwen_with_a_face_on_every_back_view_falls_back(make_job, person, faces_found):
    photo, fm = person(qwen=qwen_handler(face_seeds=(0, 1)))
    res = run_job(_job(make_job, photo, back_tries=2))
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "qwenedit"]  # no side views for a rejected back
    assert _fallbacks(res) == [
        "The Qwen photo editor could not make a usable back (a face was found on all 2 back views); the "
        "built-in back was used instead."
    ]
    assert "built-in back" in _splat_key(res)
    dbg = _debug(res)
    att = dbg["sources"][0]
    assert (att["source"], att["ok"]) == ("qwen", False) and len(att["diag"]["candidates"]) == 2
    assert "photo_sharp360_views.png" in _outputs(res)  # the rejected views are shown (to see why)


def test_qwen_side_view_failure_keeps_the_back(make_job, person):
    photo, fm = person(qwen=qwen_handler(fail_sides="CUDA out of memory"))
    res = run_job(_job(make_job, photo))
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "qwenedit", "qwenedit"]
    assert _fallbacks(res) == [
        "The Qwen photo editor could not make the side views (CUDA out of memory); the back was built from "
        "the back view alone."
    ]
    assert _splat_key(res) == "3D object (Gaussian splat, SHARP 360°, back by the Qwen photo editor)"
    assert _debug(res)["back"]["diag"]["sides"] == []


def test_qwen_failure_keeps_the_finished_back_views(make_job, person):
    """The worker made candidate 1, then ran out of memory: candidate 1 is used (and cached)."""
    photo, fm = person(qwen=qwen_handler(fail_backs_after=1))
    res = run_job(_job(make_job, photo, back_tries=2, back_quality="fast"))
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "qwenedit"]  # "fast": no side views
    assert _fallbacks(res) == [
        "The Qwen photo editor could not make every back view (CUDA out of memory); the ones made before were used."
    ]
    assert "Qwen photo editor" in _splat_key(res)
    d = _debug(res)["back"]["diag"]
    assert [c["candidate"] for c in d["candidates"]] == [1] and d["chosen"] == 1
    assert len([p for p in view_cache.default_root().iterdir() if not p.name.startswith(".")]) == 1


def test_qwen_errors_and_missing_addon_fall_back(make_job, person):
    def oom(task, inputs, params, out_dir):
        raise AddonError("This PC has 8 GB of memory (RAM); the Qwen photo editor needs at least 16 GB.")

    photo, fm = person(qwen=oom)
    res = run_job(_job(make_job, photo))
    assert _fallbacks(res) == [
        "The Qwen photo editor could not make the back (This PC has 8 GB of memory (RAM); the Qwen photo editor "
        "needs at least 16 GB.); the built-in back was used instead."
    ]
    photo, fm = person(installed=("sharp",))
    res = run_job(_job(make_job, photo))
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]
    assert _fallbacks(res) == ["The Qwen photo editor is not installed; the built-in back was used instead."]


def test_a_cancel_while_qwen_runs_propagates_and_keeps_finished_views(make_job, person):
    token = CancelToken()
    good = qwen_handler()

    def cancelled(task, inputs, params, out_dir):
        good(task, inputs, {**params, "candidates": 1}, out_dir)  # candidate 1 is on disk ...
        token.cancel()
        raise Cancelled("Operation cancelled")  # ... when the user cancels during candidate 2

    photo, fm = person(qwen=cancelled)
    with pytest.raises(Cancelled):
        run_job(_job(make_job, photo, back_tries=2), cancel=token)
    assert len([p for p in view_cache.default_root().iterdir() if not p.name.startswith(".")]) == 1


def test_qwen_and_comfyui_are_never_automatic(tmp_path):
    for quality in R.BACK_QUALITIES:
        o = S.resolve_options(_spec(back_quality=quality), installed=lambda a: True)
        assert not {"qwen", "comfyui", "images"} & set(o.sources), o.sources
    assert {"qwen", "comfyui"} <= set(R.BACK_SOURCES) and "sharp" in R.BACK_RELIEFS
    o = S.resolve_options(_spec(back_source="qwen", back_prompt=" a  red coat "), installed=bool)
    assert o.sources == ("qwen", "shell") and o.back_prompt == "a red coat"  # tried even when not installed
    assert o.comfyui_url == "" and o.comfyui_workflow is None  # ignored by the other sources
    assert S.SOURCE_ADDONS["qwen"] == "qwenedit" and "comfyui" not in S.SOURCE_ADDONS
    assert (
        S.SOURCE_NAMES["qwen"] == "the Qwen photo editor"
        and S.SOURCE_NAMES["comfyui"] == "your ComfyUI server"
    )
    assert {"qwen", "comfyui"} <= S.GENERATED_VIEW_SOURCES and {"qwen", "comfyui"} <= S.NEVER_AUTO


def _spec(**kw):
    from vrphoto.pipeline.jobs import ObjectJob

    return ObjectJob(inputs=["x.jpg"], method="sharp360", **kw)


# --------------------------------------------------------------------------- your own ComfyUI server
def _workflow(tmp_path: Path, **override) -> Path:
    """An API-format workflow using every placeholder (what 'Export (API)' plus editing gives)."""
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "{{IMAGE}}"}},
        "2": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "upscale_method": "lanczos",
                                                      "width": "{{WIDTH}}", "height": "{{HEIGHT}}", "crop": "disabled"}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "edit.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": "{{PROMPT}}"}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "seed": "{{SEED}}", "steps": 2, "cfg": 1.0,
                                                    "positive": ["4", 0], "negative": ["4", 0], "latent_image": ["2", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["3", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "vrphoto/az_{{AZIMUTH}}"}},
    }  # fmt: skip
    graph.update(override)
    p = tmp_path / "my_workflow_api.json"
    p.write_text(json.dumps(graph), encoding="utf-8")
    return p


@pytest.fixture
def comfy(tmp_path):
    """A fake ComfyUI whose SaveImage pictures are the synthetic person at the azimuth in the
    file name prefix (``az_{{AZIMUTH}}``)."""
    from PIL import Image

    with FakeComfyServer(tmp_path / "comfy") as fake:
        real = fake.fake._execute

        def execute(item):
            entry = real(item)
            graph = item[2]
            for node_id, out in (entry.get("outputs") or {}).items():
                prefix = str(graph[node_id]["inputs"]["filename_prefix"])
                phi = float(prefix.rsplit("az_", 1)[1])
                for img in out.get("images") or []:
                    target = fake.fake.output_dir / img["subfolder"] / img["filename"]
                    Image.fromarray(np.ascontiguousarray(_ai_view(phi))).save(target)
            return entry

        fake.fake._execute = execute
        yield fake


def test_comfyui_back_from_your_own_server(make_job, person, comfy, tmp_path, monkeypatch, ai_views):
    from pipeline_helpers import capture_stitch

    seen = capture_stitch(monkeypatch)
    photo, fm = person(installed=("sharp",))
    wf = _workflow(tmp_path)
    res = run_job(
        _job(make_job, photo, back_source="comfyui", comfyui_url=comfy.url, comfyui_workflow=str(wf), seed=7,
             back_prompt="black backpack")
    )  # fmt: skip
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]  # no add-on: the user's own server
    assert not _fallbacks(res), res.warnings
    assert _splat_key(res) == "3D object (Gaussian splat, SHARP 360°, back by your ComfyUI workflow)"
    # one prompt per view: the back (seed 7), then the sides; placeholders filled with typed values
    prompts = comfy.fake.prompts
    assert [p["7"]["inputs"]["filename_prefix"] for p in prompts] == [
        "vrphoto/az_180",
        "vrphoto/az_90",
        "vrphoto/az_270",
    ]
    first = prompts[0]
    assert first["5"]["inputs"]["seed"] == 7 and isinstance(first["2"]["inputs"]["width"], int)
    w, h = first["2"]["inputs"]["width"], first["2"]["inputs"]["height"]
    assert max(w, h) == 768 and w % 16 == 0 and h % 16 == 0 and h > w  # the portrait cut-out's aspect
    assert first["1"]["inputs"]["image"] == "comfyui_input.png"  # the uploaded cut-out
    text = first["4"]["inputs"]["text"]
    assert "from directly behind" in text and text.endswith("black backpack")
    assert "their left side" in prompts[1]["4"]["inputs"]["text"]  # 90 = you walked to your right
    # the back is the server's views: blue, the bag on the body's right
    sides = _far_sides(_outputs(res)["photo_sharp360_back.splat"][1])
    assert sides["back"] > 0.6 and sides["front"] < 0.05, sides
    assert seen["part"].source == "comfyui" and _bag_side(seen["part"]) < 0
    d = _debug(res)["back"]["diag"]
    assert d["generator"]["url"] == comfy.url and d["generator"]["comfyui_version"] == "0.39.2-fake"
    assert d["generator"]["workflow"] == "my_workflow_api.json" and d["generator"]["made"] == 3


def test_a_cancel_during_your_comfyui_interrupts_it(
    make_job, person, comfy, tmp_path, monkeypatch, ai_views, caplog
):
    """Design 5.3: a cancel in the back stage propagates (no fallback) and the user's ComfyUI is
    told to stop the running prompt (/interrupt) and to drop it from its queue."""
    monkeypatch.setattr(A, "_COMFY_POLL_S", 0.02)
    comfy.fake.step_s = 0.5  # 2 steps: the prompt runs for a second
    token = CancelToken()

    def cancel_when_running() -> None:
        deadline = time.monotonic() + 60
        while comfy.fake.running is None and time.monotonic() < deadline:
            time.sleep(0.01)
        token.cancel()

    threading.Thread(target=cancel_when_running, daemon=True).start()
    photo, fm = person(installed=("sharp",))
    wf = _workflow(tmp_path)
    with pytest.raises(Cancelled):
        run_job(_job(make_job, photo, back_source="comfyui", comfyui_url=comfy.url, comfyui_workflow=str(wf)),
                cancel=token)
    assert not [r.getMessage() for r in caplog.records if "was used instead" in r.getMessage()]  # no fallback
    assert len(comfy.fake.prompts) == 1  # the back view; nothing was queued after the cancel
    (asked,) = comfy.fake.interrupts
    pid = asked["prompt_id"]
    assert pid and pid in comfy.fake.deleted
    deadline = time.monotonic() + 10
    while pid not in comfy.fake.history and time.monotonic() < deadline:  # the fake's executor ends it
        time.sleep(0.01)
    status = comfy.fake.history[pid]["status"]
    assert status["messages"][-1][0] == "execution_interrupted"  # it stopped, it did not finish


def test_comfyui_not_running_falls_back(make_job, person, tmp_path):
    from vrphoto.addons.workers._comfy import free_port

    photo, fm = person(installed=("sharp",))
    url = f"http://127.0.0.1:{free_port()}"
    res = run_job(
        _job(
            make_job, photo, back_source="comfyui", comfyui_url=url, comfyui_workflow=str(_workflow(tmp_path))
        )
    )
    assert _fallbacks(res) == [
        f"Your ComfyUI server could not make the back (ComfyUI is not running at {url}); the built-in back was "
        "used instead."
    ]
    assert "built-in back" in _splat_key(res)


def test_comfyui_rejected_workflow_falls_back_with_comfyuis_reason(make_job, person, comfy, tmp_path):
    photo, fm = person(installed=("sharp",))
    wf = _workflow(tmp_path, **{"8": {"class_type": "NotInstalledNode", "inputs": {}}})
    res = run_job(
        _job(make_job, photo, back_source="comfyui", comfyui_url=comfy.url, comfyui_workflow=str(wf))
    )
    fb = _fallbacks(res)
    assert len(fb) == 1 and fb[0].startswith(
        "Your ComfyUI server could not make the back (ComfyUI rejected the workflow"
    )
    assert "NotInstalledNode" in fb[0] and fb[0].endswith("the built-in back was used instead."), fb


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing", "was not found"),
        ("ui_format", "normal format, not the API format"),
        ("no_image", "no {{IMAGE}} placeholder"),
        ("unknown", "{{NEGATIVE}}"),
        ("not_json", "not a valid JSON"),
        ("empty", "needs a workflow file"),
        ("bad_url", "not an address of a ComfyUI server"),
        ("bad_ipv6", "not an address of a ComfyUI server"),
        ("bad_port", "not an address of a ComfyUI server"),
        ("big_port", "not an address of a ComfyUI server"),
    ],
)
def test_bad_comfyui_options_fail_before_any_work(make_job, person, events, tmp_path, case, match):
    photo, fm = person(installed=("sharp",))
    wf = _workflow(tmp_path)
    url = "http://127.0.0.1:8188"
    if case == "missing":
        wf = tmp_path / "gone.json"
    elif case == "ui_format":
        wf.write_text(json.dumps({"nodes": [], "links": [], "version": 0.4}), encoding="utf-8")
    elif case == "no_image":
        wf = _workflow(tmp_path, **{"1": {"class_type": "LoadImage", "inputs": {"image": "me.png"}}})
    elif case == "unknown":
        wf = _workflow(
            tmp_path, **{"4": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{NEGATIVE}}"}}}
        )
    elif case == "not_json":
        wf.write_text("{ not json", encoding="utf-8")
    elif case == "bad_url":
        url = "ftp://my-server"
    elif case in ("bad_ipv6", "bad_port", "big_port"):  # urllib raises ValueError for these
        url = {"bad_ipv6": "http://[::1", "bad_port": "http://127.0.0.1:abc", "big_port": "127.0.0.1:70000"}[case]
    value = "" if case == "empty" else str(wf)
    with pytest.raises(InputError, match=re.escape(match)):
        run_job(_job(make_job, photo, back_source="comfyui", comfyui_url=url, comfyui_workflow=value))
    assert fm.calls == [] and "segment" not in events  # before any work


def test_comfyui_options_resolve(tmp_path):
    wf = _workflow(tmp_path)
    o = S.resolve_options(
        _spec(back_source="comfyui", comfyui_url=" 127.0.0.1:8188/ ", comfyui_workflow=f'"{wf}"'),
        installed=bool,
    )
    assert o.sources == ("comfyui", "shell") and o.comfyui_url == "http://127.0.0.1:8188"
    assert o.comfyui_workflow == str(wf)
    o = S.resolve_options(
        _spec(back_source="comfyui", comfyui_url="", comfyui_workflow=str(wf)), installed=bool
    )
    assert o.comfyui_url == "http://127.0.0.1:8188"  # empty: the default address
    # a remembered (even broken) workflow is ignored by the other sources
    o = S.resolve_options(
        _spec(back_source="shell", comfyui_workflow=str(tmp_path / "gone.json")), installed=bool
    )
    assert o.comfyui_workflow is None and o.comfyui_url == ""
    from vrphoto.pipeline.jobs import ObjectJob

    assert (ObjectJob().comfyui_url, ObjectJob().comfyui_workflow) == ("http://127.0.0.1:8188", None)


# --------------------------------------------------------------------------- the SHARP relief (M6)
def _views_folder(folder: Path, names=(("back", 180.0),), face: bool = False) -> Path:
    folder = Path(folder) / "back_photos"
    for name, phi in names:
        _save_png(_ai_view(phi, face), folder / f"{name}.png")
    return folder / "back.png"


def _bulge(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return 1.0 - np.hypot((xx + 0.5 - w / 2) / w, (yy + 0.5 - h / 2) / h)


def sharp_relief_handler(calls: list):
    """The fake SHARP add-on with its list input: a SHARP-like cloud of the picture it gets
    (the subject - everything not 0.5 grey - bulging toward the camera, a wall behind)."""
    from vrphoto.core.types import SplatCloud
    from vrphoto.geometry.splat import write_splats

    def handle(task, inputs, params, out_dir):
        from PIL import Image

        if "images" not in inputs:
            return _sharp_handler(task, inputs, params, out_dir)
        calls.append({"inputs": dict(inputs), "params": dict(params)})
        img = np.asarray(Image.open(inputs["images"][0]).convert("RGB"))
        h, w = img.shape[:2]
        f = (w / 2) / math.tan(math.radians(float(params["hfov_deg"])) / 2)
        subj = np.abs(img.astype(np.int16) - 128).max(axis=2) > 3
        z = np.where(subj, 3.0 - 0.3 * _bulge(h, w), 5.0)
        yy, xx = np.mgrid[0:h:2, 0:w:2]
        zz = z[yy, xx]
        P = np.stack([(xx + 0.5 - w / 2) * zz / f, (yy + 0.5 - h / 2) * zz / f, zz], -1).reshape(-1, 3)
        s = (1.2 * zz / f).reshape(-1, 1).repeat(3, 1)
        n = len(P)
        q = np.zeros((n, 4), np.float32)
        q[:, 0] = 1.0
        cols = img[yy, xx].reshape(-1, 3) / 255.0
        sc = SplatCloud(P.astype(np.float32), s.astype(np.float32), q, cols.astype(np.float32),
                        np.full(n, 0.9, np.float32))  # fmt: skip
        p = write_splats(sc, out_dir / f"{params['name']}_0.ply")
        return {"ply": str(p), "plys": [str(p)], "f_px": f, "f_pxs": [f], "width": w, "widths": [w], "height": h,
                "heights": [h], "frame": "opencv", "warnings": []}  # fmt: skip

    return handle


def test_back_relief_sharp_runs_sharp_on_the_lined_up_back_view(make_job, person, monkeypatch):
    from vrphoto.geometry import sharp_relief as SR

    made: list[np.ndarray] = []
    real = SR.sharp_relief

    def spy(*a, **kw):
        d = real(*a, **kw)
        made.append((d, np.asarray(kw["alpha"]) >= 0.5))
        return d

    monkeypatch.setattr(SR, "sharp_relief", spy)
    calls: list = []
    photo, fm = person(installed=("sharp",), sharp=sharp_relief_handler(calls))
    back = _views_folder(photo.parent)
    res = run_job(
        _job(
            make_job,
            photo,
            back_source="images",
            back_images=str(back),
            back_quality="fast",
            back_relief="sharp",
        )
    )
    assert not _fallbacks(res), res.warnings
    # SHARP ran twice: the photo, then the lined-up back view (list input, a long lens)
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "sharp"]
    assert len(calls) == 1 and calls[0]["params"] == {"hfov_deg": A.RELIEF_HFOV_DEG, "name": "relief1"}
    assert Path(calls[0]["inputs"]["images"][0]).name == "sharp_relief_1.png"
    # its surface became the relief: the bulge, normalised inside the expected back outline
    ((d, inside),) = made
    assert d.shape == (512, 512) and inside.shape == (512, 512) and inside.mean() > 0.05
    want = 1.0 / (3.0 - 0.3 * _bulge(512, 512))
    r = np.corrcoef(d[inside], want[inside])[0, 1]
    assert r > 0.9, r
    dbg = _debug(res)
    assert dbg["back"]["diag"]["relief"] == dict(dbg["back"]["diag"]["relief"], source="depth", model="sharp")
    assert dbg["relief"]["runs"][0]["ok"] is True
    assert not [w for w in res.warnings if "Depth Anything" in w]  # no depth-model notice: SHARP's own


def test_back_relief_sharp_falls_back_to_the_depth_model(make_job, person):
    def broken(task, inputs, params, out_dir):
        if "images" in inputs:
            raise AddonError("CUDA out of memory")
        return _sharp_handler(task, inputs, params, out_dir)

    photo, fm = person(installed=("sharp",), sharp=broken)
    back = _views_folder(photo.parent)
    res = run_job(
        _job(
            make_job,
            photo,
            back_source="images",
            back_images=str(back),
            back_quality="fast",
            back_relief="sharp",
        )
    )
    fb = _fallbacks(res)
    assert fb[0] == (
        "SHARP could not shape the surface detail of the back (CUDA out of memory); the depth model was used "
        "instead."
    ), fb
    # the depth model of these tests is the model-free heuristic: a smooth back, said once
    assert fb[1:] == ["No depth model was available for the back picture, so the back is smooth."], fb
    dbg = _debug(res)
    assert dbg["relief"]["runs"][0] == dict(dbg["relief"]["runs"][0], ok=False, error="CUDA out of memory")
    assert "back from your photo" in _splat_key(res)


def test_back_detail_automatic_uses_sharp_with_best_quality(monkeypatch):
    monkeypatch.setattr(R, "_addon_installed", lambda a: a == "sharp")
    ctx = SimpleNamespace(opts=None, run=None, item=None, relief={})
    for quality, relief, mode, kind in (
        ("best", "auto", "auto", "sharp"),
        ("best", "sharp", "depth", "sharp"),
        ("balanced", "sharp", "depth", "sharp"),
        ("fast", "sharp", "depth", "sharp"),
        ("best", "depth", "depth", "depth"),
        ("fast", "auto", "flat", None),
    ):
        ctx.opts = S.resolve_options(_spec(back_quality=quality, back_relief=relief), installed=bool)
        got, fn, used = S._relief_fn(ctx)
        assert got == mode, (quality, relief)
        if kind is None:
            assert fn is None
        else:
            assert (fn.__qualname__.startswith("sharp_depth_fn")) == (kind == "sharp"), (quality, relief)
    assert S.QUALITY["best"].relief == "sharp"  # (the autouse fixture only changed "balanced")
    monkeypatch.setattr(R, "_addon_installed", lambda a: False)  # no SHARP: the depth model
    ctx.opts = S.resolve_options(_spec(back_quality="best"), installed=bool)
    got, fn, _ = S._relief_fn(ctx)
    assert got == "auto" and not fn.__qualname__.startswith("sharp_depth_fn")


# --------------------------------------------------------------------------- face checks of the other view sources
def test_a_face_on_your_own_back_photo_is_only_a_warning(make_job, person, faces_found):
    photo, fm = person(installed=("sharp",))
    back = _views_folder(photo.parent, face=True)
    res = run_job(_job(make_job, photo, back_source="images", back_images=str(back), back_quality="fast"))
    assert "back from your photo" in _splat_key(res)  # never rejected: it may be a real photo
    fb = _fallbacks(res)
    assert len(fb) == 1 and fb[0].startswith(
        "A face was found on your back photo (back.png); if it shows the front"
    ), fb
    d = _debug(res)["back"]["diag"]
    assert d["face"]["max"] == 0.95 and d["qc"]["terms"]["face"] == 0.95


# --------------------------------------------------------------------------- helpers
def test_helpers():
    assert A._clause("The graphics card ran out of memory.") == "the graphics card ran out of memory"
    assert A._clause("ComfyUI is not running at http://x.") == "ComfyUI is not running at http://x"
    assert A._clause("CUDA out of memory") == "CUDA out of memory"
    assert A._output_size(300, 400) == (576, 768) and A._output_size(800, 400) == (768, 384)
    assert A.view_prompt(180.0, " long  hair ").endswith("(back view), full body, same pose, same clothes, plain "
                                                        "light grey background. long hair")  # fmt: skip
    assert "their right side" in A.view_prompt(270.0) and "135" not in A.view_prompt(135.0)
    # rim colours: the same picture 0, another colour 1
    a = np.zeros((64, 64), np.float32)
    a[8:56, 16:48] = 1
    red = np.zeros((64, 64, 3), np.uint8)
    red[..., 0] = 200
    blue = red[..., ::-1].copy()
    assert A.rim_chi2(red, a, red, a) == pytest.approx(0.0)
    assert A.rim_chi2(blue, a, red, a) == pytest.approx(1.0)
    assert A.rim_chi2(red, np.zeros_like(a), red, a) is None


def test_with_face_check_rejects_ai_views_and_warns_for_photos():
    from vrphoto.geometry.back_mould import BackPart

    sc = S._empty_cloud()
    part = BackPart(
        sc=sc, kind="mould", source="mvadapter", diag={"qc": {"cost": 0.2, "terms": {"iou_back": 0.2}}}
    )
    out = A.with_face_check(part, {"back": 0.6}, reject=True)
    assert out.ok and out.diag["qc"]["terms"]["face"] == 0.6 and out.diag["qc"]["cost"] == pytest.approx(0.8)
    out = A.with_face_check(replace(part, diag=dict(part.diag)), {"back": 0.9}, reject=True)
    assert not out.ok and out.reason == "a face was found on the back view (score 0.90)"
    out = A.with_face_check(
        replace(part, diag=dict(part.diag)), {"back": 0.9}, reject=False, what="your back photo"
    )
    assert out.ok and out.diag["warning"].startswith("A face was found on your back photo;")
    out = A.with_face_check(replace(part, diag=dict(part.diag)), {"back": 0.3}, reject=True)
    assert out.ok and out.diag["qc"]["terms"]["face"] == 0.0 and "warning" not in out.diag


def test_sharp_rows_that_are_not_finite_are_counted_for_the_grid(make_job, person):
    """M6 provenance: read_splats(return_rows=True) - SHARP's grid is detected from the file's
    row count even when some rows (NaN) were dropped."""
    from vrphoto.core.types import SplatCloud
    from vrphoto.geometry.splat import write_splats

    def nan_rows(task, inputs, params, out_dir):
        sc = _scene().sc
        means = np.array(sc.means, copy=True)
        means[[5, 17, len(means) - 1]] = np.nan  # the last row too
        cloud = SplatCloud(means, sc.scales, sc.quats, sc.colors, sc.opacity, sc.sh_rest)
        p = write_splats(cloud, out_dir / "model.ply")
        cam = _scene().cam
        return {"ply": str(p), "frame": "opencv", "f_px": cam.f, "width": cam.width, "height": cam.height}

    photo, fm = person(installed=("sharp",), sharp=nan_rows)
    res = run_job(_job(make_job, photo, back_source="shell", back_quality="fast"))
    dbg = _debug(res)
    n = len(_scene().sc)
    assert dbg["sharp"] == {"gaussians": n - 3, "file_rows": n, "dropped_rows": 3}  # the last row counted too
    assert dbg["camera"]["n_raw"] == n
    assert "built-in back" in _splat_key(res)
