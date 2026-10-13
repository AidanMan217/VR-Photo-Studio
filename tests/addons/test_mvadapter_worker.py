"""MV-Adapter add-on: the worker's pure helpers (cameras, input compositing, parameters, the
out-of-memory / NaN ladder, output packaging), its torch-free request handling and the
manifest <-> worker consistency.  No torch, no GPU: the diffusers parts are replaced by fakes."""

from __future__ import annotations

import fnmatch
import importlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vrphoto.addons.manifest import load_manifest, manifests_dir, workers_dir

PINNED_COMMIT = "4277e0018232bac82bb2c103caf0893cedb711be"


@pytest.fixture
def mva(monkeypatch):
    """A fresh ``mvadapter_worker`` whose protocol messages are recorded, not written."""
    monkeypatch.syspath_prepend(str(workers_dir()))
    common = importlib.import_module("_common")
    sent: list[tuple[str, dict]] = []
    monkeypatch.setattr(common, "emit", lambda t, **kw: sent.append((t, kw)))
    sys.modules.pop("mvadapter_worker", None)
    mod = importlib.import_module("mvadapter_worker")
    mod.sent = sent  # type: ignore[attr-defined]
    return mod


def _c2w(mod, az, el=0.0):
    return np.asarray(mod.c2w_opencv(az, el))


# ------------------------------------------------------------------ cameras
def test_get_c2w_port_matches_mvadapter_by_hand(mva):
    """get_c2w: camera at d (cos a, sin a, 0) in the z-up world, columns [right, up, -lookat]."""
    v0 = mva.mv_c2w_gl(-90.0)  # MV-Adapter's view 0 (azimuth 0 passed as 0 - 90)
    want0 = np.array([[1, 0, 0, 0], [0, 0, -1, -1.8], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    np.testing.assert_allclose(v0, want0, atol=1e-12)
    v90 = mva.mv_c2w_gl(0.0)  # MV-Adapter's azimuth 90
    want90 = np.array([[0, 0, 1, 1.8], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    np.testing.assert_allclose(v90, want90, atol=1e-12)
    batch = mva.mv_c2w_gl([-90.0, 0.0])
    assert batch.shape == (2, 4, 4)
    np.testing.assert_allclose(batch[1], want90, atol=1e-12)
    up = mva.mv_c2w_gl(-90.0, 30.0)  # elevation: camera above (world +z), still looking at the origin
    assert up[2, 3] == pytest.approx(1.8 * math.sin(math.radians(30)))
    np.testing.assert_allclose(-up[:3, 2], -up[:3, 3] / 1.8, atol=1e-12)
    with pytest.raises(ValueError):
        mva.mv_c2w_gl(0.0, 90.0)


def test_view0_is_the_input_camera(mva):
    c = _c2w(mva, 0)
    np.testing.assert_allclose(c[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(c[:3, 3], [0, 0, -1.8], atol=1e-12)


@pytest.mark.parametrize("az", [0, 45, 90, 180, 270, 315])
def test_phi_follows_the_app_azimuth_contract(mva, az):
    """CONTRACTS 13: forward(phi) = cos phi z - sin phi x, right(phi) = cos phi x + sin phi z,
    down = y; phi is derived from MV-Adapter's own camera and equals its azimuth."""
    c = _c2w(mva, az)
    phi, elev = mva.phi_elevation(c)
    assert phi == pytest.approx(az % 360) and elev == pytest.approx(0.0)
    p = math.radians(phi)
    x, y, z = np.eye(3)
    np.testing.assert_allclose(c[:3, 2], math.cos(p) * z - math.sin(p) * x, atol=1e-12)  # forward
    np.testing.assert_allclose(c[:3, 0], math.cos(p) * x + math.sin(p) * z, atol=1e-12)  # right
    np.testing.assert_allclose(c[:3, 1], y, atol=1e-12)  # down
    np.testing.assert_allclose(c[:3, 3], -1.8 * c[:3, 2], atol=1e-12)  # looks at the orbit centre
    assert np.linalg.det(c[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(c[:3, :3].T @ c[:3, :3], np.eye(3), atol=1e-12)


def _project(mod, c2w, P):
    m = 2 * mod.ORTHO_HALF / mod.SIZE
    d = np.asarray(P, float) - c2w[:3, 3]
    return d @ c2w[:3, 0] / m + mod.SIZE / 2, d @ c2w[:3, 1] / m + mod.SIZE / 2, d @ c2w[:3, 2]


def test_phi_90_sits_on_the_inputs_plus_x_side_and_180_looks_at_the_back(mva):
    v0, v90, v180 = _c2w(mva, 0), _c2w(mva, 90), _c2w(mva, 180)
    assert v90[0, 3] > 1.7 and abs(v90[2, 3]) < 1e-9  # on the +x (right-hand) side of the input picture
    right_hand = [0.3, 0.0, 0.0]  # a point on the right of the input picture
    u0, _, _ = _project(mva, v0, right_hand)
    assert u0 > mva.SIZE / 2
    _, _, depth90 = _project(mva, v90, right_hand)
    _, _, depth_centre = _project(mva, v90, [0, 0, 0])
    assert depth90 < depth_centre  # the phi-90 camera sees that side first
    front = [0.0, 0.0, -0.2]  # the subject's front (towards the input camera)
    assert _project(mva, v90, front)[0] < mva.SIZE / 2  # walking to the right: the front is on the left
    # phi 180: behind the subject, looking back at the input camera, left/right mirrored
    assert v180[2, 3] > 1.7
    np.testing.assert_allclose(v180[:3, 2], [0, 0, -1], atol=1e-12)
    assert _project(mva, v180, right_hand)[0] < mva.SIZE / 2
    # rows are shared by every view (orthographic, elevation 0): the invariant rows of the design
    for az in mva.AZIMUTHS:
        assert _project(mva, _c2w(mva, az), [0.1, 0.25, -0.05])[1] == pytest.approx(_project(mva, v0, [0, 0.25, 0])[1])


def test_elevation_round_trip(mva):
    c = _c2w(mva, 45, 20.0)
    phi, elev = mva.phi_elevation(c)
    assert phi == pytest.approx(45.0) and elev == pytest.approx(20.0)
    assert c[1, 3] < 0  # y is down: a camera above the subject has negative y
    assert np.linalg.det(c[:3, :3]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mva.phi_elevation(np.eye(4))


def test_cameras_doc_schema(mva):
    doc = mva.cameras_doc(7, input_fill_measured=0.8912345)
    assert doc["projection"] == "orthographic" and doc["convention"] == "c2w_opencv"
    assert (doc["ortho_half_extent"], doc["distance"], doc["width"], doc["height"]) == (0.55, 1.8, 512, 512)
    assert doc["input_fill"] == 0.9 and doc["pixel_center"] == 0.5 and doc["seed"] == 7
    assert doc["input_fill_measured"] == pytest.approx(0.8912)
    views = doc["views"]
    assert [v["file"] for v in views] == [f"view_{k}.png" for k in range(6)]
    assert [v["azimuth_deg"] for v in views] == [0, 45, 90, 180, 270, 315]
    assert [v["phi_deg"] for v in views] == [0, 45, 90, 180, 270, 315]
    assert all(v["elevation_deg"] == 0 for v in views)
    c2w = np.array(views[2]["c2w"])
    assert c2w.shape == (4, 4) and c2w[0, 3] == pytest.approx(1.8)
    text = json.dumps(doc)
    assert "-0.0" not in text and "e-17" not in text  # no float noise in the file
    assert json.loads(text) == doc


# ------------------------------------------------------------------ input
def _framed_rgba(size=512, box=(100, 300, 180, 330), color=(200, 40, 10)):
    rgba = np.zeros((size, size, 4), np.uint8)
    r0, r1, c0, c1 = box
    rgba[r0:r1, c0:c1, :3] = color
    rgba[r0:r1, c0:c1, 3] = 255
    return rgba


def test_composite_on_grey_keeps_the_framing(mva):
    rgba = _framed_rgba()
    rgba[150, 200, 3] = 128  # a half-transparent edge pixel
    rgb, info = mva.composite_on_grey(rgba)
    assert rgb.shape == (512, 512, 3) and rgb.dtype == np.uint8
    assert (rgb[0, 0] == 127).all()  # 0.5 grey, truncated like preprocess_image
    assert (rgb[100, 180] == (200, 40, 10)).all() and (rgb[299, 329] == (200, 40, 10)).all()
    assert (rgb[99, 180] == 127).all() and (rgb[300, 329] == 127).all()  # not moved or scaled
    f = rgba[150, 200].astype(np.float32) / 255
    want = ((f[:3] * f[3] + (1 - f[3]) * 0.5) * 255).clip(0, 255).astype(np.uint8)
    assert (rgb[150, 200] == want).all()
    assert info["fill"] == pytest.approx(200 / 512) and info["transparent"]
    assert any("fills 39 %" in w for w in info["warnings"])  # far from the expected 90 %
    big = _framed_rgba(box=(26, 486, 150, 360))  # 460 px = 90 %
    assert mva.composite_on_grey(big)[1]["warnings"] == []


def test_composite_on_grey_rgb_and_errors(mva):
    rgb_in = np.full((512, 512, 3), 77, np.uint8)
    rgb, info = mva.composite_on_grey(rgb_in)
    assert (rgb == 77).all() and not info["transparent"] and "no transparent background" in info["warnings"][0]
    gray, _ = mva.composite_on_grey(np.full((512, 512), 9, np.uint8))
    assert gray.shape == (512, 512, 3)
    with pytest.raises(mva.WorkerError, match="empty"):
        mva.composite_on_grey(np.zeros((512, 512, 4), np.uint8))
    with pytest.raises(mva.WorkerError, match="512 x 512"):
        mva.composite_on_grey(_framed_rgba(size=256, box=(10, 200, 50, 100)))
    with pytest.raises(mva.WorkerError, match="8-bit"):
        mva.composite_on_grey(np.zeros((512, 512, 4), np.float32))


def test_load_input_resizes_square_pictures_and_refuses_others(mva, tmp_path):
    from PIL import Image

    Image.fromarray(_framed_rgba(size=256, box=(13, 243, 75, 180))).save(tmp_path / "small.png")
    rgb, info = mva.load_input(tmp_path / "small.png")
    assert rgb.shape == (512, 512, 3) and info["fill"] == pytest.approx(460 / 512, abs=0.01)
    assert "256 px" in info["warnings"][0]
    assert (rgb[0, 0] == 127).all() and (rgb[256, 250] == (200, 40, 10)).all()
    Image.fromarray(np.zeros((300, 200, 4), np.uint8)).save(tmp_path / "tall.png")
    with pytest.raises(mva.WorkerError, match="square"):
        mva.load_input(tmp_path / "tall.png")
    (tmp_path / "junk.png").write_bytes(b"not a png")
    with pytest.raises(mva.WorkerError, match="Cannot read"):
        mva.load_input(tmp_path / "junk.png")


# ------------------------------------------------------------------ parameters
def test_parse_params_defaults(mva):
    s = mva.parse_params({})
    assert (s.steps, s.guidance, s.seeds, s.offload, s.name) == (30, 3.0, (0,), "auto", "views")
    assert s.prompt == "a person, full body, high quality, realistic photo"
    assert s.negative == mva.DEFAULT_NEGATIVE and s.warnings == []


def test_parse_params_values_clamps_and_seeds(mva):
    s = mva.parse_params({"steps": "50", "guidance": 4.5, "seed": 11, "prompt": "  a man,\n grey hoodie ",
                          "negative_prompt": "", "offload": "Sequential", "name": "back try/1"})
    assert (s.steps, s.guidance, s.seeds, s.offload) == (50, 4.5, (11,), "sequential")
    assert s.prompt == "a man, grey hoodie" and s.negative == "" and s.name == "back_try_1"
    assert mva.parse_params({"steps": 1, "guidance": 99}).steps == 5
    assert mva.parse_params({"steps": 1000}).steps == 100
    assert mva.parse_params({"guidance": 0.1}).guidance == 1.0 and mva.parse_params({"guidance": 99}).guidance == 10
    assert mva.parse_params({"steps": 40.0}).steps == 40
    s = mva.parse_params({"seeds": [3, "4", 5.0, 3], "seed": 99})  # seeds wins over seed
    assert s.seeds == (3, 4, 5) and "Repeated seeds" in s.warnings[0]
    assert mva.parse_params({"seeds": [], "seed": 8}).seeds == (8,)
    assert mva.parse_params({"seeds": 12}).seeds == (12,)
    assert mva.parse_params({"seed": None}).seeds == (0,)
    assert mva.parse_params({"prompt": "   "}).prompt == mva.DEFAULT_PROMPT
    assert mva.parse_params({"name": "///"}).name == "views"


@pytest.mark.parametrize(
    "params, match",
    [
        ({"steps": "many"}, "Invalid value for 'steps'"),
        ({"steps": 12.5}, "Invalid value for 'steps'"),
        ({"steps": True}, "Invalid value for 'steps'"),
        ({"guidance": "strong"}, "Invalid value for 'guidance'"),
        ({"guidance": float("nan")}, "Invalid value for 'guidance'"),
        ({"seed": -1}, "Invalid seed"),
        ({"seeds": [1, "x"]}, "Invalid seed"),
        ({"seeds": [1.5]}, "Invalid seed"),
        ({"seeds": [False]}, "Invalid seed"),
        ({"seeds": list(range(9))}, "Too many seeds"),
        ({"offload": "fast"}, "Invalid value for 'offload'"),
    ],
)
def test_parse_params_rejects_malformed_values(mva, params, match):
    with pytest.raises(mva.WorkerError, match=match):
        mva.parse_params(params)


def test_choose_offload(mva):
    assert mva.choose_offload("auto", 6.5) == "model"
    assert mva.choose_offload("auto", 7.2) == "none"
    assert mva.choose_offload("auto", None) == "model"
    # fp32 weights (the NaN / black retry): model offload does not fit an 8 GB card
    assert mva.choose_offload("auto", 7.5, fp32=True) == "sequential"
    assert mva.choose_offload("auto", None, fp32=True) == "sequential"
    assert mva.choose_offload("auto", 10.5, fp32=True) == "model"
    assert mva.choose_offload("auto", 13.0, fp32=True) == "none"
    assert mva.stricter_placement("model", "none") == "model"
    assert mva.stricter_placement("model", "sequential") == "sequential"
    for mode in ("model", "sequential", "none"):
        assert mva.choose_offload(mode, 1.0) == mode


def test_is_oom_bad_views_and_to_uint8(mva):
    class OutOfMemoryError(RuntimeError):
        pass

    assert mva.is_oom(OutOfMemoryError("x"))
    assert mva.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert mva.is_oom(RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert not mva.is_oom(ValueError("shapes do not match"))
    good = np.full((8, 8, 3), 0.5, np.float32)
    nan = good.copy()
    nan[2, 2, 1] = np.nan
    black = np.zeros_like(good)
    assert mva.bad_views([good, nan, black, good]) == [(1, "nan"), (2, "black")]
    assert mva.bad_views([good] * 6) == []
    u8 = mva.to_uint8(np.array([[[0.0, 0.5, 1.2]]], np.float32))
    assert u8.dtype == np.uint8 and u8.tolist() == [[[0, 128, 255]]]


def test_step_callback_reports_progress(mva):
    seen = []
    cb = mva.step_callback(lambda f, m: seen.append((f, m)), 30, attempt=2, attempts=3)
    pipe = types.SimpleNamespace(_num_timesteps=30)
    kwargs = {"latents": "L"}
    assert cb(pipe, 11, 500, kwargs) is kwargs
    assert seen[-1] == (pytest.approx(12 / 30), "Generating the views (try 2/3, step 12/30)")
    cb(types.SimpleNamespace(), 99, 0, {})  # unknown total: the requested steps, never above 1
    assert seen[-1][0] == 1.0
    single = mva.step_callback(lambda f, m: seen.append((f, m)), 20)
    single(types.SimpleNamespace(_num_timesteps=20), 0, 999, {})
    assert seen[-1][1] == "Generating the views (step 1/20)"


def test_adapter_fit(mva):
    unet = {"down.attn1.processor.to_q_mv.weight", "down.attn1.processor.to_q_ref.weight", "conv_in.weight"}
    cond = {"adapter.body.0.weight"}
    ok = mva.adapter_fit(["down.attn1.processor.to_q_mv.weight", "down.attn1.processor.to_q_ref.weight",
                          "adapter.body.0.weight"], unet, cond)
    assert ok["ok"] and ok["matched"] == 3 and ok["mv"] == 1 and ok["cond"] == 1
    renamed = mva.adapter_fit(["blocks.0.attn1.to_q_mv.weight", "adapter.body.0.weight"], unet, cond)
    assert not renamed["ok"] and renamed["fraction"] == 0.5
    assert not mva.adapter_fit([], unet, cond)["ok"]
    assert not mva.adapter_fit(["down.attn1.processor.to_q_mv.weight"], unet, cond)["ok"]  # no encoder


class _T:
    """A stand-in tensor: only ``dtype`` and ``to``."""

    def __init__(self, dtype):
        self.dtype = dtype

    def to(self, dtype):
        return _T(dtype)


def test_cast_vae_io_runs_an_fp32_vae_in_an_fp16_pipeline(mva):
    seen = {}

    class Dist:
        mean = "mean"

        def sample(self, generator=None):
            seen["generator"] = generator
            return _T("fp32")

        def mode(self):
            return _T("fp32")

    class Vae:
        dtype = "fp32"

        def encode(self, x, return_dict=True):
            seen["encode_in"] = x.dtype
            d = Dist()
            return types.SimpleNamespace(latent_dist=d) if return_dict else (d,)

        def decode(self, z, return_dict=True, generator=None):
            seen["decode_in"] = z.dtype
            return (_T("fp32"),)

    vae = mva.cast_vae_io(Vae())
    out = vae.encode(_T("fp16"))
    assert seen["encode_in"] == "fp32" and hasattr(out, "latent_dist")  # diffusers' retrieve_latents
    assert out.latent_dist.sample("gen").dtype == "fp16" and seen["generator"] == "gen"
    assert out.latent_dist.mode().dtype == "fp16" and out.latent_dist.mean == "mean"
    assert vae.encode(_T("fp16"), return_dict=False)[0].sample().dtype == "fp16"
    img = vae.decode(_T("fp16"), return_dict=False, generator=None)[0]
    assert seen["decode_in"] == "fp32" and img.dtype == "fp32"


# ------------------------------------------------------------------ generation ladder
def _views(value=0.5):
    return [np.full((4, 4, 3), value, np.float32) for _ in range(6)]


class _Ladder:
    """Fake make / generate recording what the ladder asks for."""

    def __init__(self, oom_modes=(), oom_on_load=(), bad_until_fp32=False, bad_always=False):
        self.made: list[tuple[str, bool]] = []
        self.calls: list[tuple[str, int]] = []
        self.frees = 0
        self.oom_modes = set(oom_modes)
        self.oom_on_load = set(oom_on_load)
        self.bad_until_fp32 = bad_until_fp32
        self.bad_always = bad_always

    def make(self, mode, fp32):
        self.made.append((mode, fp32))
        if mode in self.oom_on_load:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return {"mode": mode, "fp32": fp32}

    def generate(self, pipe, seed, index):
        self.calls.append((pipe["mode"], seed))
        if pipe["mode"] in self.oom_modes:
            raise RuntimeError("CUDA out of memory. Tried to allocate 3.00 GiB")
        if self.bad_always or (self.bad_until_fp32 and not pipe["fp32"]):
            return _views(np.nan)
        return _views(0.4 + 0.01 * index)

    def free(self):
        self.frees += 1


def test_generate_all_loads_once_for_all_seeds(mva):
    f = _Ladder()
    results, state = mva.generate_all([5, 6], make=f.make, generate=f.generate, mode="none", free=f.free)
    assert f.made == [("none", False)] and f.calls == [("none", 5), ("none", 6)]
    assert len(results) == 2 and results[1][0][0, 0, 0] == pytest.approx(0.41)
    assert state == {"offload": "none", "precision": "fp16", "reloads": 0}


def test_generate_all_out_of_memory_falls_back_to_sequential_offload(mva):
    f = _Ladder(oom_modes={"model"})
    warnings: list[str] = []
    results, state = mva.generate_all([1, 2], make=f.make, generate=f.generate, mode="model", free=f.free,
                                      warnings=warnings)
    assert f.made == [("model", False), ("sequential", False)]
    assert f.calls == [("model", 1), ("sequential", 1), ("sequential", 2)]
    assert state["offload"] == "sequential" and len(results) == 2 and f.frees >= 2
    assert "sequential CPU offload" in warnings[0]
    f = _Ladder(oom_on_load={"none"})  # out of memory while moving the weights to the GPU
    _, state = mva.generate_all([1], make=f.make, generate=f.generate, mode="none", free=f.free)
    assert f.made == [("none", False), ("sequential", False)] and state["offload"] == "sequential"


def test_generate_all_out_of_memory_with_sequential_offload_is_a_plain_error(mva):
    f = _Ladder(oom_modes={"none", "sequential"})
    with pytest.raises(mva.WorkerError, match="ran out of memory") as ei:
        mva.generate_all([1], make=f.make, generate=f.generate, mode="none", free=f.free)
    assert "graphics card" in ei.value.hint and f.made == [("none", False), ("sequential", False)]


def test_generate_all_retries_broken_fp16_output_in_fp32_once(mva):
    f = _Ladder(bad_until_fp32=True)
    warnings: list[str] = []
    results, state = mva.generate_all([3, 4], make=f.make, generate=f.generate, mode="none", free=f.free,
                                      free_gb=lambda: 7.5, warnings=warnings)
    # fp32 weights on an 8 GB card: straight to sequential offload (model offload cannot fit)
    assert f.made == [("none", False), ("sequential", True)]
    assert f.calls == [("none", 3), ("sequential", 3), ("sequential", 4)]
    assert state["precision"] == "fp32" and state["reloads"] == 1
    assert not mva.bad_views(results[0]) and "full precision" in warnings[0]
    f = _Ladder(bad_until_fp32=True)  # model offload in fp16 (< 7 GB free) also escalates
    _, state = mva.generate_all([3], make=f.make, generate=f.generate, mode="model", free=f.free,
                                free_gb=lambda: 6.9)
    assert f.made == [("model", False), ("sequential", True)] and state["offload"] == "sequential"
    f = _Ladder(bad_until_fp32=True)  # a 16 GB card: model offload, never less saving than before
    _, state = mva.generate_all([3], make=f.make, generate=f.generate, mode="none", free=f.free,
                                free_gb=lambda: 11.0)
    assert f.made == [("none", False), ("model", True)]
    f = _Ladder(bad_always=True)
    with pytest.raises(mva.WorkerError, match="even in full precision"):
        mva.generate_all([3], make=f.make, generate=f.generate, mode="sequential", free=f.free)
    assert f.made == [("sequential", False), ("sequential", True)]


def test_generate_all_hands_each_good_set_over_at_once(mva):
    """A set is handed to ``on_set`` before the next seed runs, so a later failure keeps it."""
    f = _Ladder()
    got: list[tuple[int, int]] = []

    def generate(pipe, seed, index):
        if seed == 2:
            raise ValueError("boom")
        return f.generate(pipe, seed, index)

    with pytest.raises(ValueError, match="boom"):
        mva.generate_all([1, 2], make=f.make, generate=generate, mode="none", free=f.free,
                         on_set=lambda i, seed, views: got.append((i, seed)))
    assert got == [(0, 1)]


def test_generate_all_propagates_other_errors(mva):
    def boom(pipe, seed, index):
        raise ValueError("shapes do not match")

    with pytest.raises(ValueError, match="shapes"):
        mva.generate_all([1], make=lambda m, f: object(), generate=boom, mode="none", free=lambda: None)


# ------------------------------------------------------------------ files and outputs
def test_save_set_writes_views_and_cameras(mva, tmp_path):
    from PIL import Image

    views = [np.full((512, 512, 3), k / 10, np.float32) for k in range(6)]
    entry = mva.save_set(tmp_path, "views", 42, views, input_fill=0.9)
    assert entry["seed"] == 42 and len(entry["views"]) == 6
    folder = tmp_path / "views_42"
    assert entry["views"][3] == str(folder / "view_3.png") and entry["cameras"] == str(folder / "cameras.json")
    img = np.asarray(Image.open(entry["views"][3]))
    assert img.shape == (512, 512, 3) and img[0, 0, 0] == round(0.3 * 255)
    doc = json.loads(Path(entry["cameras"]).read_text(encoding="utf-8"))
    assert doc["seed"] == 42 and [v["file"] for v in doc["views"]] == [Path(p).name for p in entry["views"]]
    assert doc["views"][3]["phi_deg"] == 180
    with pytest.raises(mva.WorkerError, match="6"):
        mva.save_set(tmp_path, "views", 1, views[:5])
    with pytest.raises(mva.WorkerError, match="512 x 512"):
        mva.save_set(tmp_path, "views", 1, [np.zeros((64, 64, 3), np.float32)] * 6)


def _fake_install(mod, root: Path, monkeypatch) -> None:
    """Every file the worker needs, one byte each, where the manifest's [env] points."""
    monkeypatch.setenv("VRPHOTO_ADDON_DIR", str(root))
    monkeypatch.setenv("VRPHOTO_ADDON_SRC", str(root / "src"))
    monkeypatch.setenv("MVADAPTER_BASE", str(root / "MODELS" / "sd21-base"))
    monkeypatch.setenv("MVADAPTER_WEIGHTS", str(root / "MODELS" / "mv-adapter"))
    paths = [root / "src" / "mvadapter" / f for f in mod.SOURCE_FILES]
    paths += [root / "MODELS" / "sd21-base" / f for f in mod.BASE_FILES]
    paths.append(root / "MODELS" / "mv-adapter" / mod.ADAPTER_FILE)
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        # mvadapter/__init__.py is an EMPTY file at the pinned commit (models/__init__.py too)
        p.write_bytes(b"" if p.name == "__init__.py" and p.parent.name == "mvadapter" else b"x")


def test_missing_files_and_require_models(mva, tmp_path, monkeypatch):
    monkeypatch.setenv("VRPHOTO_ADDON_DIR", str(tmp_path))
    monkeypatch.delenv("VRPHOTO_ADDON_SRC", raising=False)
    monkeypatch.delenv("MVADAPTER_BASE", raising=False)
    monkeypatch.delenv("MVADAPTER_WEIGHTS", raising=False)
    paths = mva.model_paths()
    assert paths.base == tmp_path / "MODELS" / "sd21-base" and paths.src == tmp_path / "src" / "mvadapter"
    missing = mva.missing_files(paths)
    assert len(missing) == len(mva.SOURCE_FILES) + len(mva.BASE_FILES) + 1
    with pytest.raises(mva.WorkerError, match="files are missing") as ei:
        mva.require_models(paths)
    assert "more" in str(ei.value) and "Reinstall" in ei.value.hint
    _fake_install(mva, tmp_path, monkeypatch)
    paths = mva.model_paths()
    assert mva.missing_files(paths) == []
    assert (paths.src / "mvadapter" / "__init__.py").stat().st_size == 0  # an empty source file is fine
    (paths.base / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"")  # empty = missing
    assert mva.missing_files(paths) == ["sd21-base/unet/diffusion_pytorch_model.fp16.safetensors"]
    for f in ("mvadapter/schedulers/scheduler_utils.py", "mvadapter/loaders/__init__.py"):  # imported too
        assert f in mva.SOURCE_FILES


def test_handle_validates_everything_before_torch(mva, tmp_path, monkeypatch):
    from PIL import Image

    def no_torch(*a, **k):
        raise AssertionError("torch must not be needed for this")

    monkeypatch.setattr(mva, "cuda_info", no_torch)
    monkeypatch.setenv("VRPHOTO_ADDON_DIR", str(tmp_path / "empty"))
    Image.fromarray(_framed_rgba(box=(26, 486, 150, 360))).save(tmp_path / "in.png")
    progress = mva.Progress()
    with pytest.raises(mva.WorkerError, match="Unknown task"):
        mva.handle("image_to_3d", {}, {}, tmp_path, progress)
    with pytest.raises(mva.WorkerError, match="Missing input 'image'"):
        mva.handle("image_to_views", {}, {}, tmp_path, progress)
    with pytest.raises(mva.WorkerError, match="offload"):
        mva.handle("image_to_views", {"image": str(tmp_path / "in.png")}, {"offload": "fast"}, tmp_path, progress)
    with pytest.raises(mva.WorkerError, match="files are missing"):
        mva.handle("image_to_views", {"image": str(tmp_path / "in.png")}, {}, tmp_path, progress)


def test_handle_end_to_end_with_a_fake_pipeline(mva, tmp_path, monkeypatch):
    """The request -> load -> generate -> save wiring, with the torch parts replaced."""
    from PIL import Image

    _fake_install(mva, tmp_path / "addon", monkeypatch)
    Image.fromarray(_framed_rgba(box=(26, 486, 150, 360))).save(tmp_path / "in.png")
    calls: dict[str, list] = {"load": [], "place": [], "generate": []}
    monkeypatch.setattr(mva, "cuda_info", lambda require=True: {"gpu": "NVIDIA GeForce GTX 1080", "vram_gb": 8.0})
    monkeypatch.setattr(mva, "_free_vram_gb", lambda: 6.8)
    monkeypatch.setattr(mva, "_import_mvadapter", lambda src: (None, None, "plucker"))
    monkeypatch.setattr(mva, "_control_images", lambda fn: f"control from {fn}")
    monkeypatch.setattr(mva, "_load_pipeline", lambda paths, fp32: calls["load"].append(fp32) or {"fp32": fp32})
    monkeypatch.setattr(mva, "_place", lambda pipe, mode: calls["place"].append(mode) or pipe)

    def fake_generate(pipe, reference, control, s, seed, progress, attempt, attempts):
        calls["generate"].append((seed, attempt, attempts, control, reference.size))
        mva.step_callback(progress, s.steps, attempt, attempts)(types.SimpleNamespace(_num_timesteps=s.steps),
                                                                 s.steps - 1, 0, {})
        return [np.full((512, 512, 3), 0.25 + 0.1 * k, np.float32) for k in range(6)]

    monkeypatch.setattr(mva, "_generate", fake_generate)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = mva.handle("image_to_views", {"image": str(tmp_path / "in.png")},
                     {"seeds": [7, 8], "steps": 20, "name": "back"}, out_dir, mva.Progress())
    assert calls["load"] == [False] and calls["place"] == ["model"]  # 6.8 GB free < 7 GB: model offload
    assert [c[:3] for c in calls["generate"]] == [(7, 1, 2), (8, 2, 2)]
    assert calls["generate"][0][3] == "control from plucker" and calls["generate"][0][4] == (512, 512)
    assert res["frame"] == "opencv" and res["offload"] == "model" and res["precision"] == "fp16"
    assert [s["seed"] for s in res["sets"]] == [7, 8] and res["warnings"] == []
    first = res["sets"][0]
    assert len(first["views"]) == 6 and all(Path(p).is_file() for p in first["views"])
    doc = json.loads(Path(first["cameras"]).read_text(encoding="utf-8"))
    assert doc["input_fill_measured"] == pytest.approx(460 / 512, abs=1e-3)
    ref = np.asarray(Image.open(res["input"]))
    assert ref.shape == (512, 512, 3) and (ref[0, 0] == 127).all()
    fracs = [kw["frac"] for t, kw in mva.sent if t == "progress"]
    assert fracs == sorted(fracs) and fracs[-1] == 1.0
    assert any("try 2/2, step 20/20" in kw["msg"] for t, kw in mva.sent if t == "progress")


def test_handle_reloads_with_sequential_offload_after_out_of_memory(mva, tmp_path, monkeypatch):
    from PIL import Image

    _fake_install(mva, tmp_path / "addon", monkeypatch)
    Image.fromarray(_framed_rgba(box=(26, 486, 150, 360))).save(tmp_path / "in.png")
    placed: list[str] = []
    monkeypatch.setattr(mva, "cuda_info", lambda require=True: {"gpu": "GPU", "vram_gb": 8.0})
    monkeypatch.setattr(mva, "_free_vram_gb", lambda: 7.5)
    monkeypatch.setattr(mva, "_import_mvadapter", lambda src: (None, None, None))
    monkeypatch.setattr(mva, "_control_images", lambda fn: None)
    monkeypatch.setattr(mva, "_load_pipeline", lambda paths, fp32: {})
    monkeypatch.setattr(mva, "_place", lambda pipe, mode: placed.append(mode) or {"mode": mode})
    monkeypatch.setattr(mva, "free_cuda", lambda: None)

    def fake_generate(pipe, reference, control, s, seed, progress, attempt, attempts):
        progress(0.5, "half way")
        if pipe["mode"] == "none":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.50 GiB")
        return [np.full((512, 512, 3), 0.5, np.float32)] * 6

    monkeypatch.setattr(mva, "_generate", fake_generate)
    res = mva.handle("image_to_views", {"image": str(tmp_path / "in.png")}, {"seeds": [1, 2]}, tmp_path,
                     mva.Progress())
    assert placed == ["none", "sequential"] and res["offload"] == "sequential"
    assert any("sequential CPU offload" in w for w in res["warnings"]) and len(res["sets"]) == 2
    fracs = [kw["frac"] for t, kw in mva.sent if t == "progress"]
    reload_at = next(kw["frac"] for t, kw in mva.sent if t == "progress" and kw["msg"].startswith("Reloading"))
    assert reload_at == pytest.approx(0.15) and fracs[-1] == 1.0
    assert any(t == "log" and kw["level"] == "warning" and "out of memory" in kw["msg"] for t, kw in mva.sent)


def test_handle_writes_each_set_before_the_next_seed_runs(mva, tmp_path, monkeypatch):
    """Seed 2 fails (out of memory even with sequential offload): seed 1's set is on disk, whole,
    for the app's salvage into the view cache."""
    from PIL import Image

    _fake_install(mva, tmp_path / "addon", monkeypatch)
    Image.fromarray(_framed_rgba(box=(26, 486, 150, 360))).save(tmp_path / "in.png")
    monkeypatch.setattr(mva, "cuda_info", lambda require=True: {"gpu": "GPU", "vram_gb": 8.0})
    monkeypatch.setattr(mva, "_free_vram_gb", lambda: 6.0)
    monkeypatch.setattr(mva, "_import_mvadapter", lambda src: (None, None, None))
    monkeypatch.setattr(mva, "_control_images", lambda fn: None)
    monkeypatch.setattr(mva, "_load_pipeline", lambda paths, fp32: {})
    monkeypatch.setattr(mva, "_place", lambda pipe, mode: {"mode": mode})
    monkeypatch.setattr(mva, "free_cuda", lambda: None)

    def fake_generate(pipe, reference, control, s, seed, progress, attempt, attempts):
        if seed == 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.50 GiB")
        return [np.full((512, 512, 3), 0.5, np.float32)] * 6

    monkeypatch.setattr(mva, "_generate", fake_generate)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(mva.WorkerError, match="ran out of memory"):
        mva.handle("image_to_views", {"image": str(tmp_path / "in.png")}, {"seeds": [1, 2]}, out, mva.Progress())
    doc = json.loads((out / "views_1" / "cameras.json").read_text(encoding="utf-8"))
    assert doc["seed"] == 1 and all((out / "views_1" / v["file"]).is_file() for v in doc["views"])
    assert not (out / "views_2").exists() and not list(out.rglob("*.tmp"))


def test_handle_turns_a_broken_code_install_into_a_reinstall_hint(mva, tmp_path, monkeypatch):
    from PIL import Image

    _fake_install(mva, tmp_path / "addon", monkeypatch)
    Image.fromarray(_framed_rgba(box=(26, 486, 150, 360))).save(tmp_path / "in.png")
    monkeypatch.setattr(mva, "cuda_info", lambda require=True: {"gpu": "GPU", "vram_gb": 8.0})
    monkeypatch.setattr(mva, "_free_vram_gb", lambda: 6.0)

    def broken(src):
        raise ModuleNotFoundError("No module named 'mvadapter.schedulers.scheduler_utils'")

    monkeypatch.setattr(mva, "_import_mvadapter", broken)
    with pytest.raises(mva.WorkerError, match="could not be loaded") as ei:
        mva.handle("image_to_views", {"image": str(tmp_path / "in.png")}, {}, tmp_path, mva.Progress())
    assert "Reinstall" in ei.value.hint


def test_check_without_the_packages_is_a_readable_error(mva, tmp_path, monkeypatch):
    monkeypatch.setenv("VRPHOTO_ADDON_DIR", str(tmp_path))
    monkeypatch.setattr(mva, "cuda_info", lambda require: {"cuda": False})
    monkeypatch.setitem(sys.modules, "diffusers", None)  # import fails like a broken install
    with pytest.raises(mva.WorkerError, match="Broken MV-Adapter environment") as ei:
        mva.check(require_gpu=False)
    assert "diffusers" in str(ei.value) and "model files" in str(ei.value) and "Reinstall" in ei.value.hint


# ------------------------------------------------------------------ manifest <-> worker
def test_manifest_matches_the_worker(mva):
    m = load_manifest(manifests_dir() / "mvadapter.toml")
    assert m.tasks == [mva.TASK] and m.worker == "mvadapter_worker.py"
    assert m.experimental and m.kind == "venv" and set(m.platforms) == {"win32", "linux"}
    assert not m.noncommercial  # LICENSE at the pinned commit: Apache-2.0
    assert "Apache-2.0" in m.license and "Open RAIL++-M" in m.license
    src = [s for s in m.steps if s["type"] == "github_archive"]
    assert len(src) == 1 and src[0]["repo"] == "huanngzh/MV-Adapter" and src[0]["ref"] == PINNED_COMMIT
    assert src[0]["dest"] == "src/mvadapter" and src[0].get("pip_install") is False
    hf = {s["local_dir"]: s for s in m.steps if s["type"] == "hf_download"}
    assert m.env["MVADAPTER_BASE"] == "{addon_dir}/MODELS/sd21-base"
    assert m.env["MVADAPTER_WEIGHTS"] == "{addon_dir}/MODELS/mv-adapter"
    base, weights = hf["MODELS/sd21-base"], hf["MODELS/mv-adapter"]
    for f in mva.BASE_FILES:  # everything the worker loads is downloaded
        assert any(fnmatch.fnmatch(f, pat) for pat in base["allow_patterns"]), f
    assert weights["allow_patterns"] == [mva.ADAPTER_FILE] and weights["repo"] == "huanngzh/mv-adapter"
    for step in (base, weights):
        assert len(step["revision"]) == 40 and int(step["revision"], 16) >= 0
    # the disk check reserves the torch 2.8 + diffusers venv (about 9.5 GB, as the sharp / sdxl360
    # manifests do) plus the weights; 10 GB let an install start that then ran out of disk
    assert m.disk_gb >= 9.5 + base["size_gb"] + weights["size_gb"]
    pkgs = {p.split("==")[0]: p for s in m.steps if s["type"] == "pip" for p in s["packages"]}
    for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "einops", "setuptools"):
        assert name in pkgs and "==" in pkgs[name]
    assert "peft" not in pkgs and "nvdiffrast" not in " ".join(pkgs)
