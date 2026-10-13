"""SHARP 360 with the TRELLIS back (docs/design/SHARP360.md 3.5.5, 1.3 and the TRELLIS items
of 5.3).

A fake SHARP add-on returns a SHARP-like cloud of the analytic synthetic person
(tests/geometry/synth_person.py) and a fake TRELLIS add-on a TRELLIS-like cloud of the same
body: Gaussians over its whole surface in TRELLIS's own unit-cube frame, with any of the six
axes as "up", any yaw and any scale (TRELLIS's convention is never trusted), so the
registration, colour fit and stitching run for real on consistent data.  Tests that only
check how attempts are chosen replace the registration with a stub (a real one takes ~3 s).
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vrphoto.core.errors import AddonError, Cancelled, InputError
from vrphoto.core.types import CancelToken, ProgressReporter, SplatCloud
from vrphoto.geometry import register as rg
from vrphoto.geometry.splat import write_splats
from vrphoto.pipeline import context
from vrphoto.pipeline import run as R
from vrphoto.pipeline import sharp360 as S
from vrphoto.pipeline.run import run_job

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geometry"))
import synth_person as sp  # noqa: E402

#: The scene of test_run_sharp360.py: a 240 x 320 portrait photo of the person 2.5 m away.
W, H, HFOV, G = 240, 320, 40.0, 128
NOTE = "Contains output of TRELLIS"  # start of TRELLIS's non-commercial notice


# --------------------------------------------------------------------------- synthetic data
@functools.lru_cache(maxsize=1)
def _world() -> tuple[sp.Body, sp.SharpScene, object]:
    body = sp.make_body()
    cam = sp.make_cam(W, H, HFOV, grid=G)
    return body, sp.sharp_like_scene(body, cam, G=G), sp.synth_canon(body, cam)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def body_surface(body: sp.Body, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(P, rgb)``: about ``n`` points on the outer surface of the body (frame C), roughly
    area-uniform, with their side colours (uint8)."""
    rng = np.random.default_rng(seed)
    r = body.radii
    p = 1.6075  # Knud Thomsen's ellipsoid area approximation
    area = (
        4
        * np.pi
        * (((r[:, 0] * r[:, 1]) ** p + (r[:, 0] * r[:, 2]) ** p + (r[:, 1] * r[:, 2]) ** p) / 3) ** (1 / p)
    )
    cnt = np.maximum(1, np.round(1.4 * n * area / area.sum()).astype(int))
    pts, part = [], []
    for k in range(len(r)):
        d = rng.normal(size=(cnt[k], 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts.append(body.centers[k] + (d * r[k]) @ body.rot[k].T)
        part.append(np.full(cnt[k], k))
    P, part = np.concatenate(pts), np.concatenate(part)
    inside = np.zeros(len(P), bool)
    for k in range(len(r)):  # drop points buried inside another part
        q = (P - body.centers[k]) @ body.rot[k] / r[k]
        inside |= ((q * q).sum(1) < 1.0 - 1e-9) & (part != k)
    P, part = P[~inside], part[~inside]
    if len(P) > n:
        sel = np.sort(rng.choice(len(P), n, replace=False))
        P, part = P[sel], part[sel]
    return P, body.colour_of(body.normal_at(P, part))


def _cloud(P: np.ndarray, rgb01: np.ndarray, size: float, opacity: float = 0.9) -> SplatCloud:
    m = len(P)
    return SplatCloud(
        means=np.asarray(P, np.float32),
        scales=np.full((m, 3), size, np.float32),
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (m, 1)),
        colors=np.asarray(rgb01, np.float32),
        opacity=np.full(m, opacity, np.float32),
    )


def trellis_cloud(
    up: str, yaw: float, s: float, *, n: int = 20_000, seed: int = 0, gain: float = 1.0
) -> SplatCloud:
    """The whole body as TRELLIS would write it: its frame turned so the body's up is the
    cloud axis ``up``, turned ``yaw`` deg about it, shrunk by ``s`` and centred in the unit
    cube; colours times ``gain`` in linear RGB (TRELLIS's own exposure)."""
    body, _, canon = _world()
    P, rgb = body_surface(body, n, seed)
    M = rg.up_rotation(up).T @ rg.yaw_rotation(yaw).T @ canon.R.T  # C -> TRELLIS rotation
    PT = s * (P - canon.c) @ M.T
    PT -= 0.5 * (PT.min(0) + PT.max(0))
    col = _linear_to_srgb(gain * _srgb_to_linear(rgb / 255.0))
    return _cloud(PT, col, 0.004 * s)


def true_cloud(n: int = 12_000, seed: int = 5) -> SplatCloud:
    """The body's surface already in frame C (what a perfect registration returns)."""
    body, _, _ = _world()
    P, rgb = body_surface(body, n, seed)
    return _cloud(P, rgb / 255.0, 0.004)


def box_cloud(n: int = 12_000) -> SplatCloud:
    """Garbage: the faces of a box (no person in it)."""
    rng = np.random.default_rng(7)
    face = rng.integers(0, 6, n)
    P = rng.uniform(-0.5, 0.5, (n, 3)) * np.array([0.8, 1.0, 0.6])
    ext = np.array([0.4, 0.5, 0.3])
    P[np.arange(n), face // 2] = np.where(face % 2 == 0, -1.0, 1.0) * ext[face // 2]
    return _cloud(P, np.full((n, 3), 0.5), 0.01)


# --------------------------------------------------------------------------- fake add-ons
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
            progress(0.5, "Generating the shape")
        return self.handlers[addon_id](task, inputs, params, Path(out_dir))

    def params(self, addon_id):
        return [c["params"] for c in self.calls if c["addon_id"] == addon_id]


def _sharp_handler(task, inputs, params, out_dir):
    _, scene, _ = _world()
    assert task == "image_to_splat"
    p = write_splats(scene.sc, out_dir / "model.ply")
    cam = scene.cam
    return {"ply": str(p), "frame": "opencv", "f_px": cam.f, "width": cam.width, "height": cam.height}


def trellis_handler(make_cloud):
    """A fake TRELLIS writing ``make_cloud(seed)`` (a SplatCloud) as its Gaussian PLY."""

    def handle(task, inputs, params, out_dir):
        from PIL import Image

        assert task == "image_to_3d"
        with Image.open(inputs["image"]) as im:  # an RGBA picture with a real cut-out
            assert im.mode == "RGBA" and max(im.size) <= 1024
            a = np.asarray(im)[..., 3]
        assert (a == 0).any() and (a == 255).any()
        name = params.get("name") or "model"
        p = write_splats(make_cloud(int(params["seed"])), out_dir / f"{name}_gaussians.ply")
        return {"ply": str(p), "preview": str(inputs["image"])}

    return handle


@pytest.fixture
def events():
    return []


@pytest.fixture(autouse=True)
def small_previews(monkeypatch):
    """Tiny turntable previews: they are tested in test_run_sharp360.py, and these jobs already
    spend seconds in the real registration."""
    from dataclasses import replace

    for name, q in list(S.QUALITY.items()):
        monkeypatch.setitem(S.QUALITY, name, replace(q, preview_px=64))


@pytest.fixture
def person(tmp_path, monkeypatch, events):
    """``person(trellis=None, installed=("sharp", "trellis")) -> (photo path, FakeManager)``:
    the synthetic photo, a segmenter returning its matte, a fake SHARP and (``trellis``) a
    fake TRELLIS handler."""
    from PIL import Image

    from vrphoto.addons import manager

    def make(trellis=None, installed=("sharp", "trellis")):
        _, scene, _ = _world()
        path = tmp_path / "photo.png"
        Image.fromarray(scene.rgb).save(path)

        class Seg:
            id = "birefnet-lite"

            def segment(self, img, *, rect=None, progress=None, cancel=None):
                events.append("segment")
                return scene.alpha.copy()

        seg = Seg()
        monkeypatch.setattr(context.RunContext, "segmenter", lambda self, *a, **k: seg)
        handlers = {"sharp": _sharp_handler}
        if trellis is not None:
            handlers["trellis"] = trellis
        fm = FakeManager(installed, handlers, events)
        monkeypatch.setattr(manager, "status", fm.status)
        monkeypatch.setattr(manager, "run", fm.run)
        return path, fm

    return make


def stub_registration(monkeypatch, results):
    """Replace the (3 s) registration: ``results`` maps a cloud's Gaussian count to
    ``(ok, iou, resid_m)``; a passing one returns :func:`true_cloud`."""
    calls = []

    def fake(sc, front, canon, *, progress=None, cancel=None):
        ok, iou, resid = results[len(sc)]
        calls.append(len(sc))
        diag = {"ok": ok, "reason": "" if ok else "its outline matches the photo only 40 %, 70 % is needed"}
        diag.update(iou=iou, resid_m=resid, scale=2.0, up="+y", yaw_deg=10.0, n_in=len(sc))
        return (true_cloud() if ok else None), diag

    monkeypatch.setattr(S, "register_cloud_to_front", fake)
    return calls


def _job(make_job, photo, **kw):
    kw.setdefault("common_extra", {"export_scene": False})
    return make_job("object", [photo], method="sharp360", height_m=1.6, save_extras=True, **kw)


def _outputs(res) -> dict[str, Path]:
    return {Path(p).name: Path(p) for p in res.outputs.values()}


def _debug(res) -> dict:
    return json.loads(_outputs(res)["photo_sharp360_debug.json"].read_text(encoding="utf-8"))


def _splat_key(res) -> str:
    return next(k for k in res.outputs if k.startswith("3D object (Gaussian splat"))


def _far_sides(path: Path) -> dict[str, float]:
    """Share of each side colour among the opaque Gaussians of the far quarter (lowest z:
    the subject faces +z) of an exported splat.  The true body there is mostly its blue
    back (plus the green / yellow sides of the arms and legs); the built-in back's colours
    come from the outline band of the photo, i.e. mostly the red front."""
    from vrphoto.geometry.splat import read_splats

    sc = read_splats(path)
    z = sc.means[:, 2]
    names = sp.classify_rgb(sc.colors[(z < np.percentile(z, 25)) & (sc.opacity > 0.5)], sp.SIDES)
    return {n: float((names == n).mean()) for n in sp.SIDES}


def _fallbacks(res) -> list[str]:
    return [w for w in res.warnings if not w.startswith("Note:")]


# --------------------------------------------------------------------------- the TRELLIS back, end to end
def test_trellis_back_is_lined_up_behind_the_front(make_job, person, events, check_scene):
    rng = np.random.default_rng(2024)  # a "random" convention, fixed for the test
    up = str(rng.choice(list(rg.UP_AXES)))
    yaw = float(rng.uniform(0.0, 360.0))
    s = float(rng.uniform(0.3, 1.5))
    photo, fm = person(trellis=trellis_handler(lambda seed: trellis_cloud(up, yaw, s, gain=0.8)))
    res = run_job(_job(make_job, photo, seed=3, common_extra={}))
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "trellis"]
    assert events.index("run:sharp") < events.index("run:trellis")
    call = fm.calls[1]
    assert call["task"] == "image_to_3d"
    assert call["params"] == {
        "seed": 3,
        "ss_steps": 12,
        "slat_steps": 12,
        "glb": False,
        "ply": True,
        "name": "back1",
    }
    assert Path(call["inputs"]["image"]).name == "trellis_input.png"
    assert not _fallbacks(res), res.warnings
    assert "back by TRELLIS" in _splat_key(res)

    dbg = _debug(res)
    assert dbg["options"]["sources"] == ["trellis", "shell"] and dbg["options"]["seed"] == 3
    assert dbg["sources"] == [dict(dbg["sources"][0], source="trellis", ok=True)]
    back = dbg["back"]
    assert back["source"] == "trellis" and back["kind"] == "cloud"
    d = back["diag"]
    assert d["ok"] and d["iou"] >= 0.7 and d["resid_m"] <= 0.05, d
    assert d["up"] == up and abs((d["yaw_deg"] - yaw + 180.0) % 360.0 - 180.0) < 3.0
    assert abs(1.0 / d["scale"] - s) / s < 0.03
    assert d["chosen"] == 1 and d["seed"] == 3 and d["steps"] == [12, 12]
    assert d["input"] == "levelled" and d["back_cells"] > 1000
    assert [t["try"] for t in d["tries"]] == [1]
    # TRELLIS's exposure (x 0.8) was matched to the photo
    c = d["color"]
    assert c["ok"] and c["pairs"] >= 64 and c["median_error_after"] < 0.5 * c["median_error_before"], c

    # TRELLIS's research-only licence travels with the result, next to SHARP's
    notices = check_scene(res, "object")["notices"]
    assert any(NOTE in n and "non-commercial" in n for n in notices), notices
    assert any(w.startswith(f"Note: {NOTE}") for w in res.warnings)

    # the back is behind the photographed front and is TRELLIS's (blue), not the shell's
    from vrphoto.geometry.splat import read_splats

    out = _outputs(res)
    front = read_splats(out["photo_sharp360_front.splat"])
    rear = read_splats(out["photo_sharp360_back.splat"])
    assert np.median(front.means[:, 2]) > np.median(rear.means[:, 2]) + 0.05
    sides = _far_sides(out["photo_sharp360_back.splat"])
    assert sides["back"] > 0.7 and sides["front"] < 0.05, sides


def test_trellis_garbage_fails_the_gates_and_the_built_in_back_is_used(make_job, person):
    photo, fm = person(trellis=trellis_handler(lambda seed: box_cloud()))
    res = run_job(_job(make_job, photo))
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "trellis"]
    fb = _fallbacks(res)
    assert len(fb) == 1 and fb[0].startswith(
        "TRELLIS could not make a usable back (its shape did not line up with the photo: its outline matches"
    ), res.warnings
    assert fb[0].endswith("; the built-in back was used instead."), fb
    assert "built-in back" in _splat_key(res)
    assert not any(NOTE in w for w in res.warnings)  # nothing of TRELLIS's is in the result
    dbg = _debug(res)
    att = dbg["sources"]
    assert [(a["source"], a["ok"]) for a in att] == [("trellis", False), ("shell", True)]
    assert att[0]["diag"]["best"]["iou"] < 0.7 and att[0]["diag"]["tries"][0]["ok"] is False
    assert dbg["back"]["source"] == "shell"


def test_trellis_out_of_memory_falls_back(make_job, person):
    def oom(task, inputs, params, out_dir):
        raise AddonError("CUDA out of memory")

    photo, fm = person(trellis=oom)
    res = run_job(_job(make_job, photo, back_tries=2))
    # an add-on failure ends the attempts: a second one would run out of memory too
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "trellis"]
    assert _fallbacks(res) == [
        "TRELLIS could not make the back (CUDA out of memory); the built-in back was used instead."
    ]
    assert "built-in back" in _splat_key(res)
    assert not any(NOTE in w for w in res.warnings)


def test_trellis_returning_no_gaussians_falls_back(make_job, person):
    photo, _ = person(trellis=lambda task, inputs, params, out_dir: {"preview": inputs["image"]})
    res = run_job(_job(make_job, photo))
    fb = _fallbacks(res)
    assert len(fb) == 1 and "did not return its 3D Gaussians" in fb[0], res.warnings
    assert "built-in back" in _splat_key(res)


def test_trellis_not_installed(make_job, person):
    # auto: TRELLIS is simply not part of the chain (no call, no warning)
    photo, fm = person(installed=("sharp",))
    res = run_job(_job(make_job, photo))
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]
    assert not _fallbacks(res) and "built-in back" in _splat_key(res)
    assert _debug(res)["options"]["sources"] == ["shell"]
    assert _far_sides(_outputs(res)["photo_sharp360_back.splat"])["front"] > 0.9  # cf. the TRELLIS back
    # chosen explicitly: one warning, then the built-in back
    photo, fm = person(installed=("sharp",))
    res = run_job(_job(make_job, photo, back_source="trellis"))
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]
    assert _fallbacks(res) == ["TRELLIS is not installed; the built-in back was used instead."]
    assert "built-in back" in _splat_key(res)


def test_fast_quality_never_runs_trellis_by_itself(make_job, person):
    photo, fm = person(trellis=trellis_handler(lambda seed: box_cloud()))
    res = run_job(_job(make_job, photo, back_quality="fast"))
    assert [c["addon_id"] for c in fm.calls] == ["sharp"]
    assert not _fallbacks(res)


def test_cancel_during_trellis_propagates(make_job, person, monkeypatch):
    warned: list[str] = []
    real_warn = R._Run.warn
    monkeypatch.setattr(R._Run, "warn", lambda self, m: (warned.append(m), real_warn(self, m)))
    token = CancelToken()

    def cancelled(task, inputs, params, out_dir):
        token.cancel()  # the user presses Cancel while TRELLIS runs
        raise Cancelled("Operation cancelled")

    photo, fm = person(trellis=cancelled)
    with pytest.raises(Cancelled):
        run_job(_job(make_job, photo), cancel=token)
    assert [c["addon_id"] for c in fm.calls] == ["sharp", "trellis"]
    assert not [w for w in warned if "instead" in w], warned


def test_front_is_untouched_with_the_trellis_back(make_job, person, monkeypatch):
    """Design 5.2 "front untouched": seen from the photo's camera, the fused result (front +
    TRELLIS cloud, stitched, closed by the under-layer, feathered, floaters removed) is SHARP's
    cleaned front, although TRELLIS's own front half lies on the photographed surface - and
    that half is gone: a sparse cloud coloured like the photo hardly shows in a render, so it
    is also counted geometrically (generated Gaussians on the photographed surface)."""
    from pipeline_helpers import capture_stitch, front_render_difference, generated_on_the_front

    seen = capture_stitch(monkeypatch)
    stub_registration(monkeypatch, {500: (True, 0.9, 0.01)})  # a perfectly lined-up body surface
    photo, _ = person(trellis=trellis_handler(lambda seed: true_cloud(n=500)))
    res = run_job(_job(make_job, photo))
    assert "back by TRELLIS" in _splat_key(res)
    assert seen["part"].source == "trellis" and int((~seen["is_front"]).sum()) > 1000
    diff, n = front_render_difference(seen)
    assert n > 5000 and diff < 2.0, diff
    share, n_gen = generated_on_the_front(seen)  # with TRELLIS's front half kept: ~14 %
    assert n_gen > 1000 and share < 0.02, share


# --------------------------------------------------------------------------- attempts, seeds and steps
@pytest.mark.parametrize("better", [1, 2])
def test_tries_use_consecutive_seeds_and_keep_the_better_fit(make_job, person, monkeypatch, better):
    # TRELLIS's cloud size tells the stub registration which seed made it
    reg = stub_registration(
        monkeypatch,
        {
            500 + 10: (True, 0.92 if better == 1 else 0.80, 0.01),
            500 + 11: (True, 0.92 if better == 2 else 0.80, 0.01),
        },
    )
    photo, fm = person(trellis=trellis_handler(lambda seed: true_cloud(n=500 + seed)))
    res = run_job(_job(make_job, photo, seed=10, back_tries=2, back_steps=30))
    assert [p["seed"] for p in fm.params("trellis")] == [10, 11]
    assert [(p["ss_steps"], p["slat_steps"], p["name"]) for p in fm.params("trellis")] == [
        (30, 30, "back1"),
        (30, 30, "back2"),
    ]
    assert reg == [510, 511]
    assert not _fallbacks(res), res.warnings
    d = _debug(res)["back"]["diag"]
    assert d["chosen"] == better and d["seed"] == 9 + better and d["iou"] == 0.92
    assert [(t["try"], t["seed"], t["ok"]) for t in d["tries"]] == [(1, 10, True), (2, 11, True)]


def test_best_quality_makes_two_slower_attempts(make_job, person, monkeypatch):
    stub_registration(monkeypatch, {510: (False, 0.40, 0.20), 511: (False, 0.45, 0.20)})
    photo, fm = person(trellis=trellis_handler(lambda seed: true_cloud(n=500 + seed)))
    res = run_job(_job(make_job, photo, seed=10, back_quality="best"))
    assert [(p["seed"], p["ss_steps"], p["slat_steps"]) for p in fm.params("trellis")] == [
        (10, 25, 25),
        (11, 25, 25),
    ]
    assert _fallbacks(res) == [
        "TRELLIS could not make a usable back (none of its 2 shapes lined up with the photo: its outline "
        "matches the photo only 40 %, 70 % is needed); the built-in back was used instead."
    ]
    att = _debug(res)["sources"][0]
    assert att["diag"]["best"]["iou"] == 0.45 and len(att["diag"]["tries"]) == 2


def test_a_failed_second_attempt_keeps_the_first(make_job, person, monkeypatch):
    stub_registration(monkeypatch, {510: (True, 0.9, 0.01)})

    def handler(task, inputs, params, out_dir):
        if params["seed"] == 11:
            raise AddonError("CUDA out of memory")
        return trellis_handler(lambda seed: true_cloud(n=500 + seed))(task, inputs, params, out_dir)

    photo, fm = person(trellis=handler)
    res = run_job(_job(make_job, photo, seed=10, back_tries=2))
    assert [p["seed"] for p in fm.params("trellis")] == [10, 11]
    assert _fallbacks(res) == ["TRELLIS attempt 2 of 2 failed (CUDA out of memory); attempt 1 was used."]
    assert "back by TRELLIS" in _splat_key(res)
    d = _debug(res)["back"]["diag"]
    assert d["chosen"] == 1 and d["tries"][1] == dict(d["tries"][1], ok=False, error="CUDA out of memory")


def test_a_crashed_attempt_after_a_bad_one_is_named(make_job, person, monkeypatch):
    """One shape was made (and did not line up), the other attempt crashed: the warning counts
    the shapes made and names the crash, so the advice to try another Variation is not
    followed for an out-of-memory error."""
    stub_registration(monkeypatch, {510: (False, 0.40, 0.20)})

    def handler(task, inputs, params, out_dir):
        if params["seed"] == 11:
            raise AddonError("CUDA out of memory")
        return trellis_handler(lambda seed: true_cloud(n=500 + seed))(task, inputs, params, out_dir)

    photo, fm = person(trellis=handler)
    res = run_job(_job(make_job, photo, seed=10, back_tries=2))
    assert [p["seed"] for p in fm.params("trellis")] == [10, 11]
    assert _fallbacks(res) == [
        "TRELLIS could not make a usable back (its shape did not line up with the photo: its outline "
        "matches the photo only 40 %, 70 % is needed; attempt 2 failed (CUDA out of memory)); the built-in "
        "back was used instead."
    ]
    assert "built-in back" in _splat_key(res)


def test_colour_match_off_and_worker_warnings(make_job, person, monkeypatch):
    stub_registration(monkeypatch, {500: (True, 0.9, 0.01)})
    base = trellis_handler(lambda seed: true_cloud(n=500))

    def handler(task, inputs, params, out_dir):
        return {
            **base(task, inputs, params, out_dir),
            "warnings": ["Ran on 6 GB of graphics memory (slower)."],
        }

    photo, _ = person(trellis=handler)
    res = run_job(_job(make_job, photo, color_match=False))
    d = _debug(res)["back"]["diag"]
    assert d["color"]["ok"] is False and "off" in d["color"]["reason"]
    # the worker's own warnings reach the job, named after the add-on
    assert _fallbacks(res) == ["TRELLIS (photo to 3D object): Ran on 6 GB of graphics memory (slower)."]


# --------------------------------------------------------------------------- options
def _spec(**kw):
    from vrphoto.pipeline.jobs import ObjectJob

    return ObjectJob(inputs=["x.jpg"], method="sharp360", **kw)


def test_auto_tries_trellis_first_except_with_fast_quality():
    """Without MV-Adapter, "auto" tries TRELLIS first (MV-Adapter comes before it when it is
    installed: test_sharp360_mvadapter.py)."""
    asked = []

    def installed(addon_id):
        asked.append(addon_id)
        return addon_id == "trellis"

    for quality, chain in (
        ("fast", ("shell",)),
        ("balanced", ("trellis", "shell")),
        ("best", ("trellis", "shell")),
    ):
        assert S.resolve_options(_spec(back_quality=quality), installed=installed).sources == chain
    assert asked == ["mvadapter", "trellis", "mvadapter", "trellis"]
    assert S.resolve_options(_spec(), installed=lambda a: False).sources == ("shell",)
    # an explicit TRELLIS runs whatever the quality (and warns when it is missing)
    assert S.resolve_options(_spec(back_source="trellis", back_quality="fast"), installed=bool).sources == (
        "trellis",
        "shell",
    )
    assert S.SOURCE_ADDONS["trellis"] == "trellis" and S.SOURCE_NAMES["trellis"] == "TRELLIS"


def test_resolve_options_tries_steps_seed_and_colours():
    o = S.resolve_options(_spec(), installed=bool)
    assert (o.tries, o.steps, o.seed, o.color_match) == (1, 0, 0, True)
    o = S.resolve_options(_spec(back_quality="best"), installed=bool)
    assert (o.tries, S.QUALITY["best"].trellis_steps, S.QUALITY["balanced"].trellis_steps) == (
        2,
        (25, 25),
        (12, 12),
    )
    o = S.resolve_options(
        _spec(back_quality="best", back_tries=1, back_steps=40, seed=99, color_match=False), installed=bool
    )
    assert (o.tries, o.steps, o.seed, o.color_match) == (1, 40, 99, False)
    # whole numbers from a hand-written job file
    o = S.resolve_options(_spec(back_tries="3", back_steps=20.0, seed="7"), installed=bool)
    assert (o.tries, o.steps, o.seed) == (3, 20, 7)
    for kw, what in (
        ({"back_tries": 5}, "AI attempts"),
        ({"back_tries": -1}, "AI attempts"),
        ({"back_tries": 1.5}, "AI attempts"),
        ({"back_steps": 101}, "AI steps"),
        ({"back_steps": "many"}, "AI steps"),
        ({"seed": -3}, "Variation"),
        ({"seed": 2**31}, "Variation"),
        ({"seed": float("nan")}, "Variation"),
    ):
        with pytest.raises(InputError, match=what):
            S.resolve_options(_spec(**kw), installed=bool)


@pytest.mark.parametrize(("field", "value"), [("back_tries", 9), ("back_steps", 500), ("seed", -1)])
def test_bad_ai_options_fail_before_any_work(make_job, person, events, field, value):
    photo, fm = person(trellis=trellis_handler(lambda seed: box_cloud()))
    with pytest.raises(InputError):
        run_job(make_job("object", [photo], method="sharp360", **{field: value}))
    assert not fm.calls and not events


# --------------------------------------------------------------------------- helpers on exact data
@pytest.fixture(scope="module")
def exact():
    """The true front model / frame of the photo (mould grid = the photo's pixels)."""
    body, scene, canon = _world()
    return body, sp.synth_front_model(body, scene.cam, grid_long=320), canon


def test_back_depth_is_the_far_surface(exact):
    body, front, _ = exact
    bz = S._cloud_back_z(front, true_cloud(n=60_000))
    truth = sp.true_back_depth(body, front.cam, grid_long=320)
    assert bz.shape == front.matte.shape
    assert not np.isfinite(bz[~front.matte]).any()
    both = np.isfinite(bz) & np.isfinite(truth) & (front.dt >= 3)
    assert both.sum() > 0.5 * front.matte.sum()
    err = np.abs(bz[both] - truth[both])
    assert np.median(err) < 0.01  # the max over a cell's points sits on the far surface
    # faint Gaussians do not define the back
    faint = true_cloud(n=2000)
    faint.opacity[:] = 0.1
    assert not np.isfinite(S._cloud_back_z(front, faint)).any()


def test_colours_are_matched_as_a_gain_not_a_tint(exact):
    """TRELLIS 25 % darker (linear): the photographed red front comes back to the photo's red,
    and the unseen back is brightened by the same gain, not tinted by an offset learned on a
    one-colour front (the black anchor)."""
    _, front, _ = exact
    P, rgb = body_surface(exact[0], 40_000, seed=9)
    lin = _srgb_to_linear(rgb / 255.0)
    sc = _cloud(P, _linear_to_srgb(0.75 * lin), 0.004)
    out, info = S._trellis_colours(front, sc, 60.0)
    assert info["ok"] and info["pairs"] >= 64, info
    assert info["median_error_after"] < 0.25 * info["median_error_before"], info
    got = _srgb_to_linear(out.colors.astype(np.float64))
    for side in ("front", "back"):
        sel = np.all(rgb == sp.PALETTE[side], axis=1)
        ratio = got[sel].mean(0) / lin[sel].mean(0)  # per channel: a gain of 1 = the true colour
        dominant = int(np.argmax(lin[sel].mean(0)))
        assert abs(ratio[dominant] - 1.0) < 0.08, (side, ratio)
    blue = np.all(rgb == sp.BACK_RGB, axis=1)
    assert got[blue, 0].mean() < 0.03  # no red tint on the blue back
    # matching colours never touches the geometry
    assert np.array_equal(out.means, sc.means) and np.array_equal(out.opacity, sc.opacity)


def test_colour_match_skips_with_too_few_pairs(exact):
    _, front, _ = exact
    sc = true_cloud(n=40)
    out, info = S._trellis_colours(front, sc, 60.0)
    assert out is sc and not info["ok"] and "too few" in info["reason"]


def test_try_score_and_summary():
    good = S._Try(1, 4, diag={"ok": True, "iou": 0.9, "resid_m": 0.01, "transform": {"R": [[1.0]]}})
    bad = S._Try(2, 5, diag={"ok": False, "iou": 0.95, "resid_m": 0.3, "reason": "far off"})
    empty = S._Try(3, 6, diag={"ok": False, "reason": "the generated shape is empty"})
    crashed = S._Try(4, 7, error="CUDA out of memory")
    assert good.score == pytest.approx(0.85) and bad.score == pytest.approx(0.45)
    assert empty.score == crashed.score == -np.inf
    s = crashed.summary()
    assert s == {"try": 4, "seed": 7, "seconds": 0.0, "ok": False, "error": "CUDA out of memory"}
    assert "transform" not in good.summary() and good.summary()["iou"] == 0.9
    json.dumps([t.summary() for t in (good, bad, empty, crashed)])


def test_trellis_input_is_the_levelled_picture_or_the_cutout(exact, monkeypatch, caplog):
    from types import SimpleNamespace

    from vrphoto.geometry import views_lift

    _, scene, canon = _world()
    ctx = S.Ctx(
        SimpleNamespace(cancel=CancelToken()),
        None,
        SimpleNamespace(rgb=scene.rgb),
        scene.alpha,
        scene.sc,
        exact[1],
        canon,
        None,
    )

    def check(rgba):
        assert (
            rgba.dtype == np.uint8 and rgba.ndim == 3 and rgba.shape[2] == 4 and max(rgba.shape[:2]) <= 1024
        )
        a = rgba[..., 3]
        assert (a == 0).any() and (a >= 250).mean() > 0.05
        # the subject's colours (red front) dominate the opaque part
        assert (sp.classify_rgb(rgba[..., :3][a >= 250], sp.SIDES) == "front").mean() > 0.5

    rgba, kind = S._trellis_input(ctx)
    assert kind == "levelled" and rgba.shape == (1024, 1024, 4)  # design 3.4's 1024 px reference
    check(rgba)

    # a levelled picture that cannot be made falls back to the photo's cut-out (logged)
    def broken(*a, **k):
        raise RuntimeError("levelling bug (test)")

    monkeypatch.setattr(views_lift, "levelled_ortho_input", broken)
    rgba, kind = S._trellis_input(ctx)
    assert kind == "cutout" and "levelled TRELLIS input failed" in caplog.text
    check(rgba)
    # ... but a cancel is never a fallback
    ctx.run.cancel.cancel()
    monkeypatch.setattr(views_lift, "levelled_ortho_input", lambda *a, cancel=None, **k: cancel.check())
    with pytest.raises(Cancelled):
        S._trellis_input(ctx)


def test_progress_and_reporter_wrapping(make_job, person, monkeypatch):
    """The worker's messages are shown as "TRELLIS (try i/n): ..." inside the back's range."""
    stub_registration(monkeypatch, {510: (True, 0.9, 0.01), 511: (True, 0.8, 0.01)})
    photo, _ = person(trellis=trellis_handler(lambda seed: true_cloud(n=500 + seed)))
    seen: list[tuple[float, str]] = []
    job = _job(make_job, photo, seed=10, back_tries=2)
    run_job(job, progress=ProgressReporter(lambda f, m: seen.append((f, m))))
    msgs = [m for _, m in seen]
    assert (
        "TRELLIS (try 1/2): Generating the shape" in msgs
        and "TRELLIS (try 2/2): Generating the shape" in msgs
    )
    fr = [f for f, m in seen if m.startswith("TRELLIS")]
    assert fr == sorted(fr)
