"""Qwen photo editor add-on: the worker's pure helpers (camera-LoRA prompts, RAM tiers, the ComfyUI
graph, tqdm parsing, parameters, the out-of-memory ladder), its request handling end to end with a
fake ComfyUI process standing in for ComfyUI's ``main.py`` (``fakes/fake_comfyui.py``), the install
check and the manifest <-> worker consistency.  No torch, no GPU, no network."""

from __future__ import annotations

import fnmatch
import importlib
import json
import math
import socket
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vrphoto.addons.manifest import load_manifest, manifests_dir, workers_dir

FAKES = Path(__file__).parent / "fakes"
COMFYUI_COMMIT = "3c1b7a17fdf239d35ce98cc777756f572d089795"  # v0.39.2
GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"


@pytest.fixture
def qw(monkeypatch):
    """A fresh ``qwenedit_worker`` whose protocol messages are recorded, not written."""
    monkeypatch.syspath_prepend(str(workers_dir()))
    common = importlib.import_module("_common")
    sent: list[tuple[str, dict]] = []
    monkeypatch.setattr(common, "emit", lambda t, **kw: sent.append((t, kw)))
    for name in ("qwenedit_worker", "_comfy"):
        sys.modules.pop(name, None)
    mod = importlib.import_module("qwenedit_worker")
    mod.sent = sent  # type: ignore[attr-defined]
    monkeypatch.setattr(mod, "POLL_S", 0.02)
    monkeypatch.setattr(mod, "READY_POLL_S", 0.05)
    monkeypatch.setattr(mod, "total_ram_gb", lambda: 31.9)
    return mod


# ------------------------------------------------------------------ camera prompts
@pytest.mark.parametrize("phi, words", [
    (0, "front view"), (45, "front-right quarter view"), (90, "right side view"), (135, "back-right quarter view"),
    (180, "back view"), (225, "back-left quarter view"), (270, "left side view"), (315, "front-left quarter view"),
    (359, "front view"), (-90, "left side view"), (200, "back view"), (203, "back-left quarter view"),
])
def test_angle_prompt_uses_the_lora_vocabulary(qw, phi, words):
    assert qw.angle_prompt(phi) == f"<sks> {words} eye-level shot medium shot"


def test_angle_prompt_sign_and_extra(qw):
    """phi 90 = "you walked to your right" (the subject's LEFT side): "right side view" with the
    default sign; az_sign -1 swaps the sides and keeps front / back."""
    assert qw.AZ_SIGN == 1
    assert "left side view" in qw.angle_prompt(90, az_sign=-1)
    assert "back-left quarter view" in qw.angle_prompt(135, az_sign=-1)
    assert "back view" in qw.angle_prompt(180, az_sign=-1) and "front view" in qw.angle_prompt(0, az_sign=-1)
    assert qw.angle_prompt(180, extra="  long brown   ponytail,\nblack backpack ") == (
        "<sks> back view eye-level shot medium shot. long brown ponytail, black backpack")
    assert len(qw.angle_prompt(180, extra="x" * 1000)) < 400
    assert qw.angle_prompt(180, elevation="elevated shot", distance="wide shot").endswith("elevated shot wide shot")


def test_back_views_and_jobs(qw):
    assert qw.is_back_view(180) and qw.is_back_view(160) and not qw.is_back_view(135) and not qw.is_back_view(0)
    views = [{"name": "back", "phi_deg": 180.0}, {"name": "right", "phi_deg": 90.0}]
    jobs = qw.plan_jobs(views, candidates=2, seed=10)
    assert [(j.view["name"], j.candidate, j.candidates, j.seed, j.index) for j in jobs] == [
        ("back", 0, 2, 10, 0), ("back", 1, 2, 11, 1), ("right", 0, 1, 10, 2)]


# ------------------------------------------------------------------ RAM tiers
@pytest.mark.parametrize("gb, tier", [
    (8, "refuse"), (14.4, "refuse"), (14.6, "low"), (15.7, "low"), (16, "low"), (22.4, "low"), (22.6, "normal"),
    (31.9, "normal"), (64, "normal"), (None, "low"), (float("nan"), "low"), (0, "low"),
])
def test_ram_tier(qw, gb, tier):
    assert qw.ram_tier(gb) == tier


def test_total_ram_is_measured(monkeypatch):
    monkeypatch.syspath_prepend(str(workers_dir()))
    sys.modules.pop("qwenedit_worker", None)
    mod = importlib.import_module("qwenedit_worker")
    gb = mod.total_ram_gb()
    if sys.platform == "win32" or sys.platform.startswith("linux"):  # the CI platforms: it must be measured
        assert gb is not None and 0.5 < gb < 1e5, gb
    else:
        assert gb is None or 0.5 < gb < 1e5


# ------------------------------------------------------------------ tqdm, ports, sizes, names
@pytest.mark.parametrize("line, want", [
    (" 38%|███▊      | 3/8 [00:42<01:10, 14.1s/it]", (3, 8)),
    ("100%|██████████| 8/8 [01:53<00:00, 14.16s/it]", (8, 8)),
    ("  0%|          | 0/40 [00:00<?, ?it/s]", (0, 40)),
    ("\r 12%|#2   | 1/8 [00:10<01:10]\r 25%|##5  | 2/8 [00:20<01:00]", (2, 8)),
    (b" 50%|#####     | 4/8 [00:04<00:04,  1.00s/it]", (4, 8)),
    ("Requested to load QwenImage 1/2", None),
    ("loaded completely; 7000.00 MB usable", None),
    ("| 9/8 [", None),
    ("", None),
])
def test_parse_tqdm(qw, line, want):
    assert qw.parse_tqdm(line) == want


def test_free_port_is_bindable(qw):
    port = qw.free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


@pytest.mark.parametrize("w, h, long_side, want", [
    (300, 600, 768, (384, 768)), (1000, 500, 512, (512, 256)), (512, 512, 1024, (1024, 1024)),
    (333, 1000, 768, (256, 768)), (100, 4000, 512, (128, 512)),
])
def test_output_size(qw, w, h, long_side, want):
    out = qw.output_size(w, h, long_side)
    assert out == want and all(v % 16 == 0 for v in out)


def test_scaled_size(qw):
    assert qw.scaled_size(384, 768, 1.0) == (384, 768)
    assert qw.scaled_size(384, 768, 0.8) == (304, 608)


def test_mmproj_matches_comfyui_gguf_rule(qw):
    assert qw.mmproj_matches(qw.CLIP_FILE, qw.MMPROJ_FILE)
    assert qw.mmproj_matches("Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf", "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf")
    assert not qw.mmproj_matches("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "mmproj-F16.gguf")  # unsloth's name
    assert not qw.mmproj_matches(qw.CLIP_FILE, "mmproj-Qwen2.5-VL-7B-Instruct-f16.safetensors")


def test_substitute_wrapper(qw):
    g = {"1": {"class_type": "KSampler", "inputs": {"seed": "{{SEED}}", "text": "{{PROMPT}} please"}}}
    assert qw.substitute(g, {"SEED": 5, "PROMPT": "a cat"}) == {
        "1": {"class_type": "KSampler", "inputs": {"seed": 5, "text": "a cat please"}}}


# ------------------------------------------------------------------ the graph
FILES = {"unet": "qwen-image-edit-2511-Q4_K_M.gguf", "clip": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
         "vae": "qwen_image_vae.safetensors", "angles_lora": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
         "lightning_lora": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "mmproj": "m.gguf"}


def _links_ok(graph: dict) -> None:
    for nid, node in graph.items():
        for key, v in node["inputs"].items():
            if isinstance(v, list):
                assert len(v) == 2 and v[0] in graph and v[1] == 0, (nid, key, v)


def test_build_graph(qw):
    g = qw.build_graph({"name": "back", "phi_deg": 180}, seed=7, steps=8, cfg=1.0, prompt="grey hoodie",
                       files=FILES, image="views_input.png", width=384, height=768)
    _links_ok(g)
    assert {n["class_type"] for n in g.values()} == set(qw.REQUIRED_NODES)
    assert g["1"]["inputs"] == {"unet_name": FILES["unet"]}
    assert g["2"]["inputs"] == {"clip_name": FILES["clip"], "type": "qwen_image"}
    assert g["4"]["inputs"]["lora_name"] == FILES["angles_lora"] and g["4"]["inputs"]["model"] == ["1", 0]
    assert g["5"]["inputs"]["lora_name"] == FILES["lightning_lora"] and g["6"]["inputs"]["model"] == ["5", 0]
    assert g["6"]["inputs"]["shift"] == 3.1 and g["7"]["inputs"]["strength"] == 1.0
    assert g["11"]["inputs"]["width"] == 384 and g["11"]["inputs"]["height"] == 768
    pos, neg = g["20"]["inputs"], g["21"]["inputs"]
    assert pos["prompt"] == "<sks> back view eye-level shot medium shot. grey hoodie" and neg["prompt"] == ""
    assert pos["image1"] == ["11", 0] and "image2" not in pos and neg["image1"] == ["11", 0]
    assert "vae" not in pos and "vae" not in neg  # else ComfyUI encodes every picture at ~1 megapixel
    # picture 1's latent (the input at the output size) on both conditionings, then the method
    assert g["24"] == {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["20", 0], "latent": ["30", 0]}}
    assert g["27"] == {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["21", 0], "latent": ["30", 0]}}
    assert g["22"]["inputs"] == {"conditioning": ["24", 0], "reference_latents_method": "index_timestep_zero"}
    assert g["23"]["inputs"]["conditioning"] == ["27", 0]
    ks = g["31"]["inputs"]
    assert (ks["seed"], ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"], ks["denoise"]) == (
        7, 8, 1.0, "euler", "simple", 1.0)
    assert ks["model"] == ["7", 0] and ks["positive"] == ["22", 0] and ks["negative"] == ["23", 0]
    assert ks["latent_image"] == ["30", 0] and g["30"]["inputs"]["pixels"] == ["11", 0]
    assert g["33"]["inputs"]["filename_prefix"] == "qwenedit_back"
    json.dumps(g)  # JSON-ready


def test_build_graph_without_lightning_and_with_references(qw):
    files = {**FILES, "lightning_lora": None}
    g = qw.build_graph({"name": "back left", "phi_deg": 225}, seed=1, steps=40, cfg=4.0, files=files,
                       image="in.png", width=512, height=512, references=["back.png", "side.png"], az_sign=-1,
                       prefix="views")
    _links_ok(g)
    assert "5" not in g and g["6"]["inputs"]["model"] == ["4", 0]
    assert g["12"]["inputs"]["image"] == "back.png" and g["13"]["inputs"]["image"] == "side.png"
    for nid in ("20", "21"):
        assert g[nid]["inputs"]["image2"] == ["14", 0] and g[nid]["inputs"]["image3"] == ["15", 0]
    # the references are at the output size by default, their latents chained after picture 1's
    assert [(g[n]["inputs"]["width"], g[n]["inputs"]["height"]) for n in ("11", "14", "15")] == [(512, 512)] * 3
    assert [g[n]["inputs"]["pixels"] for n in ("30", "16", "17")] == [["11", 0], ["14", 0], ["15", 0]]
    for chain, first in ((("24", "25", "26"), "20"), (("27", "28", "29"), "21")):
        assert [g[n]["inputs"]["conditioning"] for n in chain] == [[first, 0], [chain[0], 0], [chain[1], 0]]
        assert [g[n]["inputs"]["latent"] for n in chain] == [["30", 0], ["16", 0], ["17", 0]]
    assert g["22"]["inputs"]["conditioning"] == ["26", 0] and g["23"]["inputs"]["conditioning"] == ["29", 0]
    sized = qw.build_graph({"name": "left", "phi_deg": 270}, seed=1, steps=4, cfg=1.0, files=FILES, image="in.png",
                           width=384, height=768, references=["back.png"], reference_sizes=[(512, 256)])
    assert (sized["14"]["inputs"]["width"], sized["14"]["inputs"]["height"]) == (512, 256)
    with pytest.raises(ValueError, match="one reference size"):
        qw.build_graph({"name": "b", "phi_deg": 90}, seed=1, steps=4, cfg=1, files=FILES, image="a", width=64,
                       height=64, references=["r"], reference_sizes=[])
    assert "back-right quarter view" in g["20"]["inputs"]["prompt"]  # az_sign -1 mirrors 225
    assert g["33"]["inputs"]["filename_prefix"] == "views_back_left"
    with pytest.raises(ValueError, match="at most 2"):
        qw.build_graph({"name": "b", "phi_deg": 180}, seed=1, steps=8, cfg=1, files=FILES, image="a",
                       width=64, height=64, references=["1", "2", "3"])
    with pytest.raises(ValueError, match="unet"):
        qw.build_graph({"name": "b", "phi_deg": 180}, seed=1, steps=8, cfg=1, files={**FILES, "unet": None},
                       image="a", width=64, height=64)


def test_comfy_args(qw, tmp_path):
    folders = {k: tmp_path / k for k in ("input", "output", "temp", "user")}
    a = qw.comfy_args(8190, folders=folders, yaml_path=tmp_path / "x.yaml")
    joined = " ".join(a)
    assert a[:5] == ["--listen", "127.0.0.1", "--port", "8190", "--lowvram"]
    assert "--reserve-vram 1.0" in joined and "--disable-auto-launch" in a and "--dont-print-server" in a
    assert "--fp32-vae" in a and "--cache-none" not in a and "--force-fp32" not in a
    assert a[a.index("--extra-model-paths-config") + 1] == str(tmp_path / "x.yaml")
    assert a[a.index("--output-directory") + 1] == str(folders["output"])
    low = qw.comfy_args(1, folders=folders, yaml_path="y", tier="low", vram_mode="novram", precision="fp32")
    assert "--novram" in low and "--lowvram" not in low and "--cache-none" in low and "--force-fp32" in low
    cpu = qw.comfy_args(1, folders=folders, yaml_path="y", vram_mode="cpu")
    assert "--cpu" in cpu and "--reserve-vram" not in cpu
    with pytest.raises(ValueError):
        qw.comfy_args(1, folders=folders, yaml_path="y", vram_mode="gpu-only")


def test_extra_model_paths_yaml(qw, tmp_path):
    text = qw.extra_model_paths_yaml({"diffusion_models": [tmp_path / "d"], "loras": [tmp_path / "a", tmp_path / "b",
                                                                                         tmp_path / "a"],
                                      "vae": [r"C:\Users\Me\My Models\vae"]})
    lines = text.splitlines()
    assert lines[1] == "vrphoto_qwenedit:" and "  diffusion_models: |" in lines
    assert lines[lines.index("  loras: |") + 1:lines.index("  loras: |") + 3] == [f"    {tmp_path / 'a'}",
                                                                                 f"    {tmp_path / 'b'}"]
    assert r"    C:\Users\Me\My Models\vae" in lines and "text_encoders" not in text
    try:  # when PyYAML is around, ComfyUI's own loader logic gives the same folders
        import yaml
    except ImportError:
        return
    conf = yaml.safe_load(text)["vrphoto_qwenedit"]
    assert [y for y in conf["loras"].split("\n") if y] == [str(tmp_path / "a"), str(tmp_path / "b")]


# ------------------------------------------------------------------ parameters
def test_parse_params_defaults(qw):
    s = qw.parse_params({})
    assert [v["name"] for v in s.views] == ["back", "right", "left", "back_right", "back_left"]
    assert [v["phi_deg"] for v in s.views] == [180, 90, 270, 135, 225]
    assert (s.candidates, s.seed, s.steps, s.cfg, s.long_side, s.ram_tier, s.lightning, s.az_sign, s.name) == (
        2, 0, None, None, 768, "auto", "auto", 1, "views")
    assert s.view_timeout_s is None  # by the steps (default_view_timeout)


def test_default_view_timeout_follows_the_model_passes(qw):
    assert qw.default_view_timeout(4, 1.0) == qw.default_view_timeout(8, 1.0) == 3600.0  # Lightning
    assert qw.default_view_timeout(16, 1.0) == 7200.0
    assert qw.default_view_timeout(40, 4.0) == 14400.0  # without Lightning: 80 passes -> the maximum
    assert qw.default_view_timeout(qw.DEFAULT_STEPS_BASE, qw.DEFAULT_CFG_BASE) == qw.VIEW_TIMEOUT_RANGE[1]


def test_parse_params_values_and_clamps(qw):
    s = qw.parse_params({"views": [{"name": "back!", "phi_deg": 540}, {"phi_deg": -90}, {"azimuth_deg": "45"}],
                         "candidates": 9, "seed": "12", "steps": 100, "cfg": 0.2, "prompt_extra": " a  b ",
                         "long_side": 2000, "ram_tier": "LOW", "lightning": False, "az_sign": -1,
                         "view_timeout_s": 5, "name": "my views"})
    assert s.views == [{"name": "back", "phi_deg": 180.0}, {"name": "az270", "phi_deg": 270.0},
                       {"name": "az045", "phi_deg": 45.0}]
    assert (s.candidates, s.seed, s.steps, s.cfg, s.prompt_extra, s.long_side) == (4, 12, 40, 1.0, "a b", 1024)
    assert (s.ram_tier, s.lightning, s.az_sign, s.view_timeout_s, s.name) == ("low", "off", -1, 60.0, "my_views")
    assert qw.parse_params({"steps": 1, "long_side": 700}).steps == 4
    assert qw.parse_params({"long_side": 700}).long_side == 704
    assert qw.parse_params({"steps": 0, "lightning": "true"}).steps is None
    assert qw.parse_params({"lightning": "true"}).lightning == "on"


@pytest.mark.parametrize("params, match", [
    ({"views": "back"}, "views"),
    ({"views": [{"name": "a"}]}, "phi_deg"),
    ({"views": [{"name": "a", "phi_deg": "x"}]}, "phi_deg"),
    ({"views": [{"name": "a", "phi_deg": 1}, {"name": "a", "phi_deg": 2}]}, "both called 'a'"),
    ({"views": [{"phi_deg": k} for k in range(9)]}, "Too many views"),
    ({"views": ["back"]}, "Invalid view"),
    ({"seed": -1}, "seed"),
    ({"seed": 1.5}, "seed"),
    ({"cfg": float("nan")}, "cfg"),
    ({"cfg": "x"}, "cfg"),
    ({"steps": "many"}, "steps"),
    ({"ram_tier": "huge"}, "ram_tier"),
    ({"lightning": "maybe"}, "lightning"),
    ({"az_sign": 2}, "az_sign"),
])
def test_parse_params_errors(qw, params, match):
    with pytest.raises(qw.WorkerError, match=match):
        qw.parse_params(params)


# ------------------------------------------------------------------ the fallback ladder
def test_ladder_step(qw):
    s = {"vram_mode": "lowvram", "scale": 1.0, "precision": "auto"}
    assert qw.ladder_step(s, "oom") == "novram" and qw.ladder_step(s, "crash") == "novram"
    s["vram_mode"] = "novram"
    assert qw.ladder_step(s, "oom") == "smaller"
    s["scale"] = 0.8
    assert qw.ladder_step(s, "oom") is None and qw.ladder_step(s, "timeout") is None
    assert qw.ladder_step(s, "black") == "fp32"
    s["precision"] = "fp32"
    assert qw.ladder_step(s, "black") is None and qw.ladder_step(s, "error") is None


def test_generate_runs_the_ladder(qw, tmp_path):
    jobs = qw.plan_jobs([{"name": "back", "phi_deg": 180.0}, {"name": "right", "phi_deg": 90.0}], candidates=1,
                        seed=0)
    calls: list[tuple] = []
    restarts: list[dict] = []
    outcomes = {"back": ["oom", "oom", "ok"], "right": ["oom"]}

    def run_one(job, st):
        calls.append((job.view["name"], st["vram_mode"], st["scale"]))
        kind = outcomes[job.view["name"]].pop(0)
        if kind != "ok":
            raise qw.JobFailed(kind, "out of memory")
        return tmp_path / f"{job.view['name']}.png"

    done: list[str] = []
    warnings: list[str] = []
    state = {"vram_mode": "lowvram", "scale": 1.0, "precision": "auto"}
    skipped = qw.generate(jobs, run_one=run_one, restart=lambda st: restarts.append(dict(st)), state=state,
                          warnings=warnings, on_done=lambda job, path, st: done.append(job.view["name"]))
    assert calls == [("back", "lowvram", 1.0), ("back", "novram", 1.0), ("back", "novram", 0.8),
                     ("right", "novram", 0.8)]
    assert len(restarts) == 1 and restarts[0]["vram_mode"] == "novram"
    assert done == ["back"] and skipped == [{"name": "right", "candidate": 0, "reason": "out of memory"}]
    assert any("--novram" in w for w in warnings) and any("80 %" in w for w in warnings)
    assert any("right view was skipped" in w for w in warnings)


def test_generate_raises_when_nothing_was_made(qw):
    jobs = qw.plan_jobs([{"name": "back", "phi_deg": 180.0}, {"name": "left", "phi_deg": 270.0}], candidates=1,
                        seed=0)
    calls: list[str] = []

    def run_one(job, st):
        calls.append(job.view["name"])
        raise qw.JobFailed("timeout", "it took longer than 60 s")

    # the first view failing for another reason than memory stops at once (the others would fail too)
    with pytest.raises(qw.WorkerError, match="could not make the back view .*longer than 60 s") as ei:
        qw.generate(jobs, run_one=run_one, restart=lambda st: None, state={"vram_mode": "lowvram", "scale": 1.0},
                    warnings=[], on_done=lambda *a: None)
    assert "built-in back" in ei.value.hint and calls == ["back"]
    # after a view was made, a failing view is skipped and the rest go on
    calls.clear()
    outcomes = iter(["ok", "error"])

    def run_two(job, st):
        calls.append(job.view["name"])
        if next(outcomes) == "error":
            raise qw.JobFailed("error", "KSampler failed")
        return Path("x.png")

    warnings: list[str] = []
    skipped = qw.generate(jobs, run_one=run_two, restart=lambda st: None,
                          state={"vram_mode": "lowvram", "scale": 1.0}, warnings=warnings, on_done=lambda *a: None)
    assert calls == ["back", "left"] and skipped[0]["name"] == "left" and "KSampler failed" in warnings[0]
    # memory everywhere: the ladder runs out and the error names it
    state = {"vram_mode": "novram", "scale": 0.8}
    with pytest.raises(qw.WorkerError, match="could not make any view") as ei:
        qw.generate(jobs, run_one=lambda job, st: (_ for _ in ()).throw(qw.JobFailed("oom", "out of memory")),
                    restart=lambda st: None, state=state, warnings=[], on_done=lambda *a: None)
    assert "Close other programs" in ei.value.hint


def test_server_log_reader_splits_tqdm_from_log(qw, tmp_path):
    logs: list[tuple[str, str]] = []
    srv = qw.ComfyServer(tmp_path, tmp_path, tmp_path / "x.yaml", log_fn=lambda m, lvl="info": logs.append((lvl, m)))
    steps: list[tuple[int, int]] = []
    srv.on_step = lambda n, t: steps.append((n, t))

    class Stream:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        def read(self, n):
            return self.chunks.pop(0) if self.chunks else b""

    srv._read(Stream([b"got prompt\n  0%|   | 0/8 [00:00<?, ?it/s]\r 12%|#  ", b" | 1/8 [00:10<01:10]\r",
                      b"\x1b[1m\x1b[31m[ERROR]\x1b[0m !!! Exception during processing !!! CUDA out of memory\n"
                      b"\x1b[32m[INFO]\x1b[0m Prompt executed"]))
    assert steps == [(0, 8), (1, 8)]
    assert [m for _l, m in logs] == ["[comfyui] got prompt",
                                     "[comfyui] [ERROR] !!! Exception during processing !!! CUDA out of memory",
                                     "[comfyui] [INFO] Prompt executed"]  # ComfyUI's colour codes removed
    assert logs[1][0] == "warning" and "Prompt executed" in srv.tail_text() and "\x1b" not in srv.tail_text()


def test_strip_ansi_and_port_taken(qw):
    assert qw.strip_ansi("\x1b[1m\x1b[31m[ERROR]\x1b[0m Port 5 is\x1b[K in use") == "[ERROR] Port 5 is in use"
    assert qw.port_taken(["loading", "\x1b[1m\x1b[31m[ERROR]\x1b[0m Port 51234 is already in use on address "
                                     "127.0.0.1. Please close the other application or use a different port with --port."])
    assert qw.port_taken(["OSError: [Errno 98] Address already in use"])
    assert qw.port_taken(["OSError: [WinError 10048] Only one usage of each socket address (protocol/network "
                          "address/port) is normally permitted"])
    assert not qw.port_taken(["The file is already in use by another process", "Requested to load QwenImage"])


def test_start_tries_another_port_when_comfyui_finds_its_port_taken(qw, tmp_path, monkeypatch):
    """ComfyUI binds its port a minute or two after the worker picked it; when another program took
    it meanwhile, ComfyUI v0.39.2 logs "Port N is already in use ..." (coloured) and exits with 1."""
    inst = _install(qw, tmp_path, monkeypatch)
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text(qw.extra_model_paths_yaml({}), encoding="utf-8")
    taken = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    taken.bind(("127.0.0.1", 0))  # bound, not listening: the readiness probe is refused at once
    port = taken.getsockname()[1]
    real, first = qw.free_port, iter([port])
    monkeypatch.setattr(qw, "free_port", lambda host="127.0.0.1": next(first, None) or real(host))
    logs: list[str] = []
    srv = qw.ComfyServer(inst["comfy"], tmp_path / "work", yaml_path, log_fn=lambda m, lvl="info": logs.append(m))
    try:
        srv.start()
        assert srv.starts == 2 and srv.port != port and srv.alive and srv.version == "0.39.2-fake"
    finally:
        srv.stop()
        taken.close()
    assert any("[ERROR] Port " in m and "already in use" in m for m in logs)
    assert any("trying another one" in m for m in logs) and not any("\x1b" in m for m in logs)


# ------------------------------------------------------------------ pictures
def _cutout(path: Path, w: int = 300, h: int = 600) -> Path:
    from PIL import Image

    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((xx - w / 2) / (w * 0.3)) ** 2 + ((yy - h / 2) / (h * 0.45)) ** 2 <= 1.0
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 180, 60, 40
    rgba[..., 3] = np.where(inside, 255, 0)
    Image.fromarray(rgba).save(path)
    return path


def test_load_picture_composites_on_grey(qw, tmp_path):
    rgb, info = qw.load_picture(_cutout(tmp_path / "c.png"))
    assert rgb.shape == (600, 300, 3) and info["transparent"] and not info["warnings"]
    assert tuple(rgb[0, 0]) == (200, 200, 200) and tuple(rgb[300, 150]) == (180, 60, 40)
    small, _ = qw.load_picture(tmp_path / "c.png", size=(64, 128))
    assert small.shape == (128, 64, 3) and tuple(small[0, 0]) == (200, 200, 200)
    capped, _ = qw.load_picture(tmp_path / "c.png", max_side=100)
    assert capped.shape == (100, 50, 3)
    from PIL import Image

    Image.new("RGB", (10, 10), (1, 2, 3)).save(tmp_path / "rgb.png")
    rgb2, info2 = qw.load_picture(tmp_path / "rgb.png")
    assert tuple(rgb2[0, 0]) == (1, 2, 3) and "no transparent background" in info2["warnings"][0]
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(tmp_path / "empty.png")
    with pytest.raises(qw.WorkerError, match="empty"):
        qw.load_picture(tmp_path / "empty.png")
    (tmp_path / "bad.png").write_bytes(b"nope")
    with pytest.raises(qw.WorkerError, match="Cannot read"):
        qw.load_picture(tmp_path / "bad.png")


def test_is_black(qw, tmp_path):
    from PIL import Image

    Image.new("RGB", (8, 8), (0, 0, 0)).save(tmp_path / "b.png")
    Image.new("RGB", (8, 8), (90, 90, 90)).save(tmp_path / "g.png")
    assert qw.is_black(tmp_path / "b.png") and not qw.is_black(tmp_path / "g.png")
    assert not qw.is_black(tmp_path / "missing.png")


# ------------------------------------------------------------------ end to end with a fake ComfyUI
def _install(qw, tmp_path: Path, monkeypatch, *, lightning: bool = True) -> dict[str, Path]:
    """An add-on folder whose ComfyUI ``main.py`` runs the fake server, and one-byte model files."""
    addon = tmp_path / "addon"
    comfy = addon / "src" / "ComfyUI"
    for rel in qw.COMFY_FILES:
        (comfy / rel).parent.mkdir(parents=True, exist_ok=True)
        (comfy / rel).write_text("", encoding="utf-8")
    (comfy / "main.py").write_text(
        f"import sys\nsys.path.insert(0, {str(FAKES)!r})\nimport fake_comfyui\nfake_comfyui.main()\n", encoding="utf-8")
    models = addon / "MODELS"
    layout = {"unet": "diffusion_models", "clip": "text_encoders", "mmproj": "text_encoders",
              "vae": "vae/split_files/vae", "angles_lora": "loras/angles", "lightning_lora": "loras/lightning"}
    paths = {}
    for kind, (_folder, name) in qw.MODEL_FILES.items():
        if kind == "lightning_lora" and not lightning:
            continue
        p = models / layout[kind] / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        paths[kind] = p
    (models / ".cache" / "huggingface").mkdir(parents=True)
    (models / ".cache" / "huggingface" / qw.UNET_FILE).write_bytes(b"metadata, not the model")
    monkeypatch.setenv("VRPHOTO_ADDON_DIR", str(addon))
    monkeypatch.setenv("COMFYUI_DIR", str(comfy))
    monkeypatch.setenv("QWENEDIT_MODELS", str(models))
    record = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_COMFY_RECORD", str(record))
    for var in ("FAKE_COMFY_OOM", "FAKE_COMFY_FAIL", "FAKE_COMFY_BLACK", "FAKE_COMFY_REJECT",
                "FAKE_COMFY_MISSING_NODES", "FAKE_COMFY_CRASH_AFTER", "FAKE_COMFY_STEP_S"):
        monkeypatch.delenv(var, raising=False)
    return {"addon": addon, "comfy": comfy, "models": models, "record": record, **paths}


def _records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _run(qw, tmp_path, params=None, inputs=None):
    img = _cutout(tmp_path / "cutout.png")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    res = qw.handle(qw.TASK, inputs or {"image": str(img)}, params or {}, out, qw.Progress())
    return res, out


def _port_closed(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def test_handle_end_to_end(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_COMFY_STEP_S", "0.03")
    views = [{"name": "back", "phi_deg": 180}, {"name": "right", "phi_deg": 90}]
    res, out = _run(qw, tmp_path, {"views": views, "candidates": 2, "seed": 7, "prompt_extra": "grey hoodie"})
    assert [(v["name"], v["candidate"], v["seed"], v["phi_deg"]) for v in res["views"]] == [
        ("back", 0, 7, 180.0), ("back", 1, 8, 180.0), ("right", 0, 7, 90.0)]
    assert res["frame"] == "nominal" and res["comfyui_version"] == "0.39.2-fake"
    assert (res["width"], res["height"], res["steps"], res["cfg"], res["lightning"]) == (384, 768, 4, 1.0, True)
    assert (res["ram_tier"], res["vram_mode"], res["precision"], res["skipped"]) == ("normal", "lowvram", "auto", [])
    assert res["view_timeout_s"] == 3600.0 and not any("Lightning" in w for w in res["warnings"])
    from PIL import Image

    for v in res["views"]:
        assert Path(v["file"]).parent == out
        with Image.open(v["file"]) as im:
            assert im.size == (384, 768)
    assert Path(res["views"][0]["file"]).name == "views_back_c0.png"
    lines = [json.loads(x) for x in (out / "views.partial.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["file"] for x in lines] == [v["file"] for v in res["views"]]
    assert lines[0]["prompt"] == "<sks> back view eye-level shot medium shot. grey hoodie" and lines[0]["steps"] == 4
    with Image.open(res["input"]) as im:  # the cut-out on #c8c8c8 at the output size
        assert im.size == (384, 768) and im.getpixel((0, 0)) == (200, 200, 200)
    rec = _records(inst["record"])
    starts = [r for r in rec if r["event"] == "start"]
    prompts = [r["prompt"] for r in rec if r["event"] == "prompt"]
    assert len(starts) == 1 and len(prompts) == 3
    argv = starts[0]["argv"]
    assert "--lowvram" in argv and argv[argv.index("--reserve-vram") + 1] == "1.0" and "--cache-none" not in argv
    yaml_text = Path(argv[argv.index("--extra-model-paths-config") + 1]).read_text(encoding="utf-8")
    assert str(inst["vae"].parent) in yaml_text and ".cache" not in yaml_text
    assert [p["31"]["inputs"]["seed"] for p in prompts] == [7, 8, 7]
    assert "right side view" in prompts[2]["20"]["inputs"]["prompt"]
    assert prompts[0]["1"]["inputs"]["unet_name"] == qw.UNET_FILE and prompts[0]["5"]["inputs"]["lora_name"]
    assert prompts[0]["10"]["inputs"]["image"] == "views_input.png"
    msgs = [kw.get("msg", "") for t, kw in qw.sent if t == "progress"]
    assert any("(step " in m for m in msgs) and msgs[-1] == "Done"
    assert any("candidate 2/2" in m for m in msgs)
    assert _port_closed(int(argv[argv.index("--port") + 1]))  # ComfyUI was stopped


def test_handle_with_references_and_without_lightning(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch, lightning=False)
    ref = _cutout(tmp_path / "ref.png", 200, 400)
    img = _cutout(tmp_path / "cutout.png")
    res, _ = _run(qw, tmp_path, {"views": [{"name": "left", "phi_deg": 270}], "long_side": 512},
                  {"image": str(img), "references": [str(ref)]})
    assert (res["steps"], res["cfg"], res["lightning"]) == (40, 4.0, False)
    # the optional LoRA is missing: said plainly, and a 40-step view gets the longest time limit
    assert any("Lightning LoRA is not installed" in w and "20 times slower" in w for w in res["warnings"])
    assert res["view_timeout_s"] == qw.VIEW_TIMEOUT_RANGE[1]
    prompt = [r["prompt"] for r in _records(inst["record"]) if r["event"] == "prompt"][0]
    assert "5" not in prompt and prompt["12"]["inputs"]["image"] == "views_reference1.png"
    assert prompt["20"]["inputs"]["image2"] == ["14", 0] and "left side view" in prompt["20"]["inputs"]["prompt"]
    assert prompt["31"]["inputs"]["steps"] == 40 and prompt["11"]["inputs"]["width"] == 256
    # the reference latent is made at the long side asked for (512), not at ComfyUI's fixed ~1 megapixel
    assert (prompt["14"]["inputs"]["width"], prompt["14"]["inputs"]["height"]) == (256, 512)
    assert prompt["25"]["inputs"]["latent"] == ["16", 0] and prompt["16"]["inputs"]["pixels"] == ["14", 0]


def test_handle_oom_restarts_with_novram(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_COMFY_OOM", "unless_novram")
    res, _ = _run(qw, tmp_path, {"views": [{"name": "back", "phi_deg": 180}], "candidates": 1})
    assert res["vram_mode"] == "novram" and len(res["views"]) == 1
    assert any("--novram" in w for w in res["warnings"])
    starts = [r["argv"] for r in _records(inst["record"]) if r["event"] == "start"]
    assert len(starts) == 2 and "--lowvram" in starts[0] and "--novram" in starts[1]


def test_handle_oom_then_smaller_then_skip(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_COMFY_OOM", "above:200000")  # 384 x 768 fails, 304 x 608 fits
    ref = _cutout(tmp_path / "ref.png", 300, 600)
    res, _ = _run(qw, tmp_path, {"views": [{"name": "right", "phi_deg": 90}], "candidates": 1},
                  {"image": str(_cutout(tmp_path / "cutout.png")), "references": [str(ref)]})
    assert res["scale"] == 0.8 and res["vram_mode"] == "novram"
    from PIL import Image

    with Image.open(res["views"][0]["file"]) as im:
        assert im.size == (304, 608)
    # the smaller rung shrinks EVERY latent (the output and the reference), not only the output
    prompts = [r["prompt"] for r in _records(inst["record"]) if r["event"] == "prompt"]
    sizes = [{n: (g[n]["inputs"]["width"], g[n]["inputs"]["height"]) for n in ("11", "14")} for g in prompts]
    assert sizes[0] == {"11": (384, 768), "14": (384, 768)} and sizes[-1] == {"11": (304, 608), "14": (304, 608)}
    for g in prompts:
        assert all("vae" not in n["inputs"] for n in g.values() if n["class_type"] == "TextEncodeQwenImageEditPlus")


def test_handle_out_of_memory_everywhere(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_COMFY_OOM", "always")
    with pytest.raises(qw.WorkerError, match="could not make any view") as ei:
        _run(qw, tmp_path, {"views": [{"name": "back", "phi_deg": 180}], "candidates": 1})
    assert "Close other programs" in ei.value.hint
    argv = [r["argv"] for r in _records(inst["record"]) if r["event"] == "start"][-1]
    assert _port_closed(int(argv[argv.index("--port") + 1]))


def test_handle_survives_a_comfyui_crash(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_COMFY_CRASH_AFTER", "1")  # dies while running its first prompt (unless --novram)
    res, _ = _run(qw, tmp_path, {"views": [{"name": "back", "phi_deg": 180}, {"name": "right", "phi_deg": 90}],
                                 "candidates": 1})
    assert res["vram_mode"] == "novram" and [v["name"] for v in res["views"]] == ["back", "right"]
    assert len([r for r in _records(inst["record"]) if r["event"] == "start"]) == 2


def test_handle_black_picture_retries_in_fp32(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_COMFY_BLACK", "1")
    res, _ = _run(qw, tmp_path, {"views": [{"name": "back", "phi_deg": 180}], "candidates": 1})
    assert res["precision"] == "fp32" and len(res["views"]) == 1 and not qw.is_black(Path(res["views"][0]["file"]))
    starts = [r["argv"] for r in _records(inst["record"]) if r["event"] == "start"]
    assert "--force-fp32" not in starts[0] and "--force-fp32" in starts[1]


def test_handle_low_ram_tier_and_refusal(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    monkeypatch.setattr(qw, "total_ram_gb", lambda: 15.8)
    res, _ = _run(qw, tmp_path, {"views": [{"name": "back", "phi_deg": 180}], "candidates": 1})
    assert res["ram_tier"] == "low" and any("without its cache" in w for w in res["warnings"])
    assert "--cache-none" in [r["argv"] for r in _records(inst["record"]) if r["event"] == "start"][0]
    inst["record"].unlink()
    monkeypatch.setattr(qw, "total_ram_gb", lambda: 8.0)
    with pytest.raises(qw.WorkerError, match="needs at least 16 GB") as ei:
        _run(qw, tmp_path, {})
    assert ei.value.hint == "Use 'Your own back photo' or the built-in back."
    assert not inst["record"].exists()  # refused before ComfyUI was started
    res, _ = _run(qw, tmp_path, {"views": [{"name": "back", "phi_deg": 180}], "candidates": 1, "ram_tier": "low"})
    assert res["ram_tier"] == "low"  # the expert override


def test_handle_rejected_graph_and_missing_nodes(qw, tmp_path, monkeypatch):
    _install(qw, tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_COMFY_REJECT", "1")
    with pytest.raises(qw.WorkerError, match="could not run the Qwen workflow.*rejected"):
        _run(qw, tmp_path, {"views": [{"name": "back", "phi_deg": 180}], "candidates": 1})
    monkeypatch.delenv("FAKE_COMFY_REJECT")
    monkeypatch.setenv("FAKE_COMFY_MISSING_NODES", "UnetLoaderGGUF,CLIPLoaderGGUF")
    with pytest.raises(qw.WorkerError, match="ComfyUI-GGUF nodes did not load") as ei:
        _run(qw, tmp_path, {"views": [{"name": "back", "phi_deg": 180}], "candidates": 1})
    assert "Reinstall" in ei.value.hint


def test_handle_missing_files_and_inputs(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    inst["unet"].unlink()
    with pytest.raises(qw.WorkerError, match="files are missing: qwen-image-edit-2511-Q4_K_M.gguf") as ei:
        _run(qw, tmp_path, {})
    assert "Reinstall" in ei.value.hint
    with pytest.raises(qw.WorkerError, match="Missing input 'image'"):
        qw.handle(qw.TASK, {}, {}, tmp_path, qw.Progress())
    with pytest.raises(qw.WorkerError, match="reference picture 1"):
        qw.handle(qw.TASK, {"image": str(_cutout(tmp_path / "c.png")), "references": [str(tmp_path / "nope.png")]},
                  {}, tmp_path, qw.Progress())
    with pytest.raises(qw.WorkerError, match="Unknown task"):
        qw.handle("image_to_views", {}, {}, tmp_path, qw.Progress())


def test_missing_files_checks_the_mmproj_pairing(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    models = qw.find_models(inst["models"])
    assert models["unet"] == inst["unet"]  # the .cache copy is ignored
    assert qw.missing_files(inst["comfy"], models) == []
    moved = inst["models"] / "elsewhere" / qw.MMPROJ_FILE
    moved.parent.mkdir()
    inst["mmproj"].rename(moved)
    assert qw.missing_files(inst["comfy"], qw.find_models(inst["models"])) == [
        f"{qw.MMPROJ_FILE} next to {qw.CLIP_FILE}"]
    models = {**qw.find_models(inst["models"]), "lightning_lora": None}
    assert f"{qw.MMPROJ_FILE} next to {qw.CLIP_FILE}" in qw.missing_files(inst["comfy"], models)
    assert qw.LIGHTNING_LORA_FILE not in qw.missing_files(inst["comfy"], models)  # optional


# ------------------------------------------------------------------ install check
def _fake_torch(cuda: bool = True):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.8.0+cu128"
    torch.version = types.SimpleNamespace(cuda="12.8")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_properties=lambda i=0: types.SimpleNamespace(name="NVIDIA GeForce GTX 1080",
                                                                total_memory=8 * 1024**3, major=6, minor=1),
        get_device_capability=lambda device=0: (6, 1),
        get_arch_list=lambda: ["sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
        get_device_name=lambda device=0: "NVIDIA GeForce GTX 1080",
    )
    return torch


def test_check_starts_comfyui_and_lists_everything(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    info = qw.check(require_gpu=True)
    assert info["gpu"] == "NVIDIA GeForce GTX 1080" and info["comfyui_version"] == "0.39.2-fake"
    assert info["ram_tier"] == "normal" and info["lightning"] is True and "warnings" not in info
    argv = [r["argv"] for r in _records(inst["record"]) if r["event"] == "start"][0]
    assert "--lowvram" in argv and _port_closed(int(argv[argv.index("--port") + 1]))
    # no GPU (CI with --no-gpu): ComfyUI is started on the CPU
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    info = qw.check(require_gpu=False)
    assert "--cpu" in [r["argv"] for r in _records(inst["record"]) if r["event"] == "start"][-1]
    monkeypatch.setattr(qw, "total_ram_gb", lambda: 8.0)
    assert "will refuse to run" in qw.check(require_gpu=False)["warnings"][0]


def test_check_reports_missing_nodes_and_files(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setenv("FAKE_COMFY_MISSING_NODES", "TextEncodeQwenImageEditPlus")
    with pytest.raises(qw.WorkerError, match="too old or damaged"):
        qw.check(require_gpu=True)
    (inst["comfy"] / "custom_nodes" / "ComfyUI-GGUF" / "nodes.py").unlink()
    with pytest.raises(qw.WorkerError, match="ComfyUI/custom_nodes/ComfyUI-GGUF/nodes.py"):
        qw.check(require_gpu=True)


def test_check_reports_a_comfyui_that_does_not_start(qw, tmp_path, monkeypatch):
    inst = _install(qw, tmp_path, monkeypatch)
    (inst["comfy"] / "main.py").write_text(
        "import sys\nprint('\\x1b[1m\\x1b[31m[ERROR]\\x1b[0m ModuleNotFoundError: No module named \"comfy_aimdo\"')\n"
        "sys.exit(1)\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    with pytest.raises(qw.WorkerError, match="stopped while starting: .*comfy_aimdo") as ei:
        qw.check(require_gpu=True)
    assert "Reinstall" in ei.value.hint and "\x1b" not in str(ei.value)  # plain text, no colour codes


# ------------------------------------------------------------------ manifest <-> worker
def test_manifest_matches_the_worker(qw):
    m = load_manifest(manifests_dir() / "qwenedit.toml")
    assert m.tasks == [qw.TASK] and m.worker == "qwenedit_worker.py"
    assert m.experimental and m.kind == "venv" and set(m.platforms) == {"win32", "linux"}
    assert m.title == "Qwen photo editor (AI back views, experimental, slow)"
    assert m.min_ram_gb == qw.MIN_RAM_GB == 16 and m.min_vram_gb == 6
    assert not m.noncommercial and "Apache-2.0" in m.license and "GPL-3.0" in m.license
    assert m.env["COMFYUI_DIR"] == "{addon_dir}/src/ComfyUI" and m.env["QWENEDIT_MODELS"] == "{addon_dir}/MODELS"
    src = {s["repo"]: s for s in m.steps if s["type"] == "github_archive"}
    assert src["comfyanonymous/ComfyUI"]["ref"] == COMFYUI_COMMIT
    assert src["comfyanonymous/ComfyUI"]["dest"] == "src/ComfyUI"
    assert src["city96/ComfyUI-GGUF"]["ref"] == GGUF_COMMIT
    assert src["city96/ComfyUI-GGUF"]["dest"] == "src/ComfyUI/custom_nodes/ComfyUI-GGUF"
    order = [s.get("repo") for s in m.steps]
    assert order.index("comfyanonymous/ComfyUI") < order.index("city96/ComfyUI-GGUF")  # ComfyUI replaces its folder
    assert all(not s.get("pip_install") for s in src.values())
    hf = [s for s in m.steps if s["type"] == "hf_download"]
    for kind, (_folder, name) in qw.MODEL_FILES.items():  # everything the worker looks for is downloaded
        steps = [s for s in hf if any(fnmatch.fnmatch(name, p.rsplit("/", 1)[-1]) for p in s["allow_patterns"])]
        assert len(steps) == 1, name
        assert steps[0]["local_dir"].startswith("MODELS/") and steps[0]["revision"]
        assert bool(steps[0].get("optional")) == (kind in qw.OPTIONAL_MODELS), name
    unet = [s for s in hf if s["repo"] == "unsloth/Qwen-Image-Edit-2511-GGUF"]
    assert len(unet) == 1 and unet[0]["allow_patterns"] == [qw.UNET_FILE]  # one quant, pinned in the manifest
    te = [s for s in hf if qw.CLIP_FILE in s["allow_patterns"]][0]
    assert qw.MMPROJ_FILE in te["allow_patterns"] and qw.mmproj_matches(qw.CLIP_FILE, qw.MMPROJ_FILE)
    weights = sum(s.get("size_gb", 0) for s in hf)
    assert 20 < weights < 22 and m.disk_gb >= weights + 7
    pkgs = {p.split("==")[0].lower(): p for s in m.steps if s["type"] == "pip" for p in s["packages"]}
    assert pkgs["torch"] == "torch==2.8.0" and pkgs["torchvision"] == "torchvision==0.23.0"
    assert "torchaudio" not in pkgs and "comfyui-workflow-templates" not in pkgs
    for name, pin in {"comfyui-frontend-package": "1.53.10", "comfy-kitchen": "0.2.37", "comfy-aimdo": "0.5.5"}.items():
        assert pkgs[name] == f"{name}=={pin}"  # exactly as in ComfyUI v0.39.2's requirements.txt
    for name in ("av", "transformers", "safetensors", "aiohttp", "sqlalchemy", "alembic", "gguf", "sentencepiece",
                 "protobuf", "huggingface-hub", "spandrel", "kornia", "pydantic", "scipy", "psutil", "blake3",
                 "simpleeval", "torchsde", "pyyaml", "pillow", "numpy", "setuptools"):
        assert name in pkgs, name
    assert int(pkgs["av"].split("==")[1].split(".")[0]) < 19  # av 19 needs Python 3.12
    assert m.steps[-1]["type"] == "check"


def test_worker_imports_comfy_only_inside_functions():
    import ast

    tree = ast.parse((workers_dir() / "qwenedit_worker.py").read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name for n in top if isinstance(n, ast.Import) for a in n.names} | {
        n.module for n in top if isinstance(n, ast.ImportFrom)}
    assert "_comfy" not in names and "PIL" not in names and "torch" not in names


def test_math_sanity_of_defaults(qw):
    assert qw.DEFAULT_LONG_SIDE % qw.SIZE_MULTIPLE == 0 and qw.STEPS_RANGE[0] <= qw.DEFAULT_STEPS_LIGHTNING
    assert "4steps" in qw.LIGHTNING_LORA_FILE and qw.DEFAULT_STEPS_LIGHTNING == 4  # the LoRA's own step count
    assert math.isclose(qw.OOM_SCALE, 0.8) and qw.GREY == (200, 200, 200)


def test_worker_process_through_the_app_runner(qw, tmp_path, monkeypatch):
    """The real protocol path: the app's runner starts the worker script, which starts (fake) ComfyUI;
    ComfyUI's output must not leak into the protocol stream and the result arrives intact."""
    from vrphoto.addons.env import clean_env
    from vrphoto.addons.runner import run_worker

    inst = _install(qw, tmp_path, monkeypatch)
    img = _cutout(tmp_path / "cutout.png")
    env = clean_env({"VRPHOTO_ADDON_DIR": str(inst["addon"]), "COMFYUI_DIR": str(inst["comfy"]),
                     "QWENEDIT_MODELS": str(inst["models"]), "FAKE_COMFY_RECORD": str(inst["record"]),
                     "FAKE_COMFY_STEP_S": "0.02"})
    progress: list[tuple[float, str]] = []
    out = run_worker(sys.executable, workers_dir() / "qwenedit_worker.py", qw.TASK, {"image": str(img)},
                     {"views": [{"name": "back", "phi_deg": 180}], "candidates": 1, "ram_tier": "normal"},
                     tmp_path / "out", env=env, timeout=120, title="Qwen photo editor",
                     progress=lambda f, m="": progress.append((f, m)))
    assert [v["name"] for v in out["views"]] == ["back"] and Path(out["views"][0]["file"]).is_file()
    assert out["comfyui_version"] == "0.39.2-fake" and out["frame"] == "nominal"
    assert any("Starting ComfyUI" in m for _f, m in progress)
    argv = [r["argv"] for r in _records(inst["record"]) if r["event"] == "start"][0]
    assert _port_closed(int(argv[argv.index("--port") + 1]))
