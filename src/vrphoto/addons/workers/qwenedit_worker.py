"""Qwen photo editor worker (experimental, slow): a cut-out person -> nominal back / side views.

Runs inside the ``qwenedit`` add-on venv (Python 3.11, torch 2.8.0 + CUDA 12.8 and ComfyUI's
requirements).  The worker does not run a model itself: it starts a private, **headless ComfyUI**
(pinned v0.39.2, extracted to ``COMFYUI_DIR`` = ``<addon_dir>/src/ComfyUI``, with ComfyUI-GGUF in
its ``custom_nodes``) as a child process on ``127.0.0.1:<free port>``, queues one API graph per
view and candidate and polls ``/history`` (``_comfy.ComfyClient``).  ComfyUI is GPL-3.0 and runs as
a separate program; the worker only talks to it over HTTP.

Models (``QWENEDIT_MODELS`` = ``<addon_dir>/MODELS``, found by file name anywhere below it):
Qwen-Image-Edit-2511 as a Q4_K_M GGUF (Unsloth), the Qwen2.5-VL-7B-Instruct text encoder as a
Q4_K_M GGUF plus its ``mmproj`` vision file (ggml-org; ComfyUI-GGUF pairs them by name), the
Qwen-Image VAE, fal's Multiple-Angles camera LoRA and lightx2v's 4-step Lightning LoRA (an optional
install step: without it every view takes 40 steps at CFG 4, about 20 times the work, with a
warning and a longer time limit per view).

Task ``views_from_photo``:
    inputs:  ``image`` - the cut-out (RGBA; RGB is used as it is), composited on #c8c8c8 (design
             3.5.6: hair edges stay readable); ``references`` - optional list of up to 2 more
             pictures handed to the edit model as "picture 2 / 3" (e.g. the chosen back for the side
             views, design 3.5.6 phase 3).
    params:  ``views`` (``[{"name", "phi_deg"}]``, default :data:`DEFAULT_VIEWS`: back, both sides,
             both back quarters), ``candidates`` (2: tries for each back view, 1-4), ``seed``,
             ``steps`` (4 with the Lightning LoRA, else 40; 4-40), ``cfg`` (1.0 with Lightning, else
             4.0; 1-10), ``prompt_extra`` ('Describe the hidden side'), ``long_side`` (768,
             512-1024; the output AND every reference latent), ``ram_tier`` (auto | low | normal),
             ``lightning`` (auto | true | false), ``az_sign`` (+1 / -1, see :func:`angle_prompt`),
             ``view_timeout_s`` (60-14400; default :func:`default_view_timeout`), ``name``.
    outputs: ``views`` = ``[{"name", "phi_deg", "file", "candidate", "seed"}]`` (only the views that
             were made), ``frame`` = "nominal", ``warnings``, ``comfyui_version``, plus ``input``,
             ``width`` / ``height``, ``steps``, ``cfg``, ``lightning``, ``ram_tier``, ``vram_mode``,
             ``precision``, ``skipped`` and ``worker_version``.  Every finished view is written at
             once (``<name>_<view>_c<k>.png``) and appended to ``views.partial.jsonl``, so a failure
             or a cancel (the app kills the process tree) keeps the views made before.

Runtime (design 4.4):

1. RAM tier (:func:`ram_tier`): below 16 GB of system RAM the job is refused ("use 'Your own back
   photo' or the built-in back"); below 24 GB ComfyUI runs with ``--cache-none``.
2. ComfyUI: ``python main.py --listen 127.0.0.1 --port <free> --lowvram --reserve-vram 1.0
   --disable-auto-launch --dont-print-server --fp32-vae ...`` with ``HF_HUB_OFFLINE=1``, the model
   folders given by an ``extra_model_paths.yaml`` the worker writes, its input / output / temp /
   user folders inside the job's work folder and an in-memory database.
   Precision on GTX 10-series (ComfyUI v0.39.2 ``model_management``): Qwen-Image only supports
   bf16 / fp32 inference and Pascal has no bf16 compute, so ComfyUI computes the diffusion model in
   fp32 by itself (GGUF weights are dequantised on the fly); the text encoder runs in fp16 (the GTX
   1080 is on ComfyUI's fp16 list); the VAE is forced to fp32.  A black / NaN picture -> ComfyUI is
   restarted once with ``--force-fp32``.  (``--lowvram`` does nothing while ComfyUI's dynamic VRAM
   is active, which it is on NVIDIA with torch 2.8; ``--novram`` turns it off.)
3. ``/system_stats`` must answer; ``/object_info`` must hold every node of :func:`build_graph`
   and the loaders must list the model files.
4. Out of memory (or ComfyUI crashing, e.g. out of system RAM): restart with ``--novram``, then
   the long side x 0.8, then the view is skipped with a warning (:func:`ladder_step`).
5. ComfyUI is ALWAYS stopped in ``finally``.  It is a child of the worker, so the app's
   kill-on-close Job object (Windows) / process group (Linux, plus a parent-death signal) ends it
   together with the worker.

The pure helpers (:func:`angle_prompt`, :func:`ram_tier`, :func:`build_graph`, :func:`parse_tqdm`,
:func:`substitute`, :func:`free_port`, ...) are unit-tested in ``tests/addons/test_qwenedit_worker.py``
(no torch, no GPU; a fake ComfyUI server plays ComfyUI).
"""

from __future__ import annotations

import collections
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from _common import (
    Progress,
    WorkerError,
    addon_dir,
    cuda_info,
    log,
    param,
    require_input,
    run,
    src_dir,
)

VERSION = "1.0"
TASK = "views_from_photo"

# --------------------------------------------------------------------------- pinned files
#: Model files (``kind -> (ComfyUI folder kind, file name)``); the manifest downloads exactly these.
UNET_FILE = "qwen-image-edit-2511-Q4_K_M.gguf"
CLIP_FILE = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
VAE_FILE = "qwen_image_vae.safetensors"
ANGLES_LORA_FILE = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
LIGHTNING_LORA_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
MODEL_FILES: dict[str, tuple[str, str]] = {
    "unet": ("diffusion_models", UNET_FILE),
    "clip": ("text_encoders", CLIP_FILE),
    "mmproj": ("text_encoders", MMPROJ_FILE),
    "vae": ("vae", VAE_FILE),
    "angles_lora": ("loras", ANGLES_LORA_FILE),
    "lightning_lora": ("loras", LIGHTNING_LORA_FILE),
}
OPTIONAL_MODELS = frozenset({"lightning_lora"})
#: ComfyUI files the worker needs (relative to ``COMFYUI_DIR``).
COMFY_FILES = ("main.py", "server.py", "nodes.py", "comfy_extras/nodes_qwen.py",
               "custom_nodes/ComfyUI-GGUF/__init__.py", "custom_nodes/ComfyUI-GGUF/nodes.py",
               "custom_nodes/ComfyUI-GGUF/loader.py")

#: Every node type :func:`build_graph` uses (``/object_info`` must have them all).
REQUIRED_NODES = (
    "UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoader", "LoraLoaderModelOnly", "ModelSamplingAuraFlow", "CFGNorm",
    "LoadImage", "ImageScale", "TextEncodeQwenImageEditPlus", "ReferenceLatent",
    "FluxKontextMultiReferenceLatentMethod", "VAEEncode", "KSampler", "VAEDecode", "SaveImage",
)

# --------------------------------------------------------------------------- generation settings
#: ComfyUI's Qwen-Image-Edit-2511 template (Comfy-Org/workflow_templates): ModelSamplingAuraFlow
#: shift 3.1, CFGNorm 1.0, the "index_timestep_zero" reference method (needed for re-packaged 2511
#: files such as GGUF), euler / simple, 4 steps CFG 1 with Lightning, 40 steps CFG 4 without.
SHIFT = 3.1
CFG_NORM = 1.0
REFERENCE_METHOD = "index_timestep_zero"
SAMPLER = "euler"
SCHEDULER = "simple"
DEFAULT_STEPS_LIGHTNING = 4  # lightx2v's LoRA is distilled for 4 steps at CFG 1 (Comfy-Org's templates)
DEFAULT_STEPS_BASE = 40
STEPS_RANGE = (4, 40)
DEFAULT_CFG_LIGHTNING = 1.0
DEFAULT_CFG_BASE = 4.0
CFG_RANGE = (1.0, 10.0)
DEFAULT_LONG_SIDE = 768
LONG_SIDE_RANGE = (512, 1024)
SIZE_MULTIPLE = 16  # 8x VAE x 2x2 patches
MIN_SIDE = 256
OOM_SCALE = 0.8
GREY = (200, 200, 200)  # #c8c8c8
MAX_REFERENCES = 2
MAX_PROMPT_CHARS = 300
MAX_VIEWS = 8
CANDIDATES_RANGE = (1, 4)
DEFAULT_CANDIDATES = 2
MAX_SEED = 2**63 - 1
DEFAULT_VIEW_TIMEOUT_S = 3600.0
VIEW_TIMEOUT_RANGE = (60.0, 14400.0)
#: Model passes (steps x 2 with CFG > 1) :data:`DEFAULT_VIEW_TIMEOUT_S` is meant for; more passes get a
#: proportionally longer default time limit (:func:`default_view_timeout`).
TIMEOUT_REF_PASSES = 8
DEFAULT_VIEWS: tuple[dict[str, Any], ...] = (
    {"name": "back", "phi_deg": 180.0},
    {"name": "right", "phi_deg": 90.0},
    {"name": "left", "phi_deg": 270.0},
    {"name": "back_right", "phi_deg": 135.0},
    {"name": "back_left", "phi_deg": 225.0},
)

# --------------------------------------------------------------------------- camera LoRA vocabulary
#: fal's Qwen-Image-Edit-2511-Multiple-Angles-LoRA: ``<sks> <azimuth> <elevation> <distance>``,
#: 8 azimuths x 4 elevations (low-angle -30, eye-level 0, elevated 30, high-angle 60) x 3 distances
#: (close-up, medium shot, wide shot); azimuth words in 45-degree steps from 0 = front view.
TRIGGER = "<sks>"
AZIMUTH_WORDS = ("front view", "front-right quarter view", "right side view", "back-right quarter view",
                 "back view", "back-left quarter view", "left side view", "front-left quarter view")
ELEVATION_WORDS = "eye-level shot"
DISTANCE_WORDS = "medium shot"
#: Direction of the LoRA's azimuth relative to the app's ``phi`` (CONTRACTS 13: phi 90 = "you walked
#: to your right", the subject's LEFT side is visible).  +1: the LoRA's 90 = "right side view" = phi
#: 90, i.e. "right" names the side the camera moved to, as in the ComfyUI-qwenmultiangle camera
#: widget made for this LoRA (camera at +x of a picture facing +z for 90 degrees).  NOT verified on
#: real outputs: whether the training captions meant the camera's or the subject's right is measured
#: in the M7 spike; the app tests both hypotheses (design 3.5.6 "Side sign"), and ``az_sign = -1``
#: flips the mapping without a code change.
AZ_SIGN = 1

# --------------------------------------------------------------------------- RAM tiers
MIN_RAM_GB = 16.0  # manifest min_ram_gb
LOW_RAM_GB = 24.0  # below: --cache-none
RAM_SLACK_GB = 1.5  # Windows reports a "16 GB" PC as 15.4-15.9 GB
RAM_TIERS = ("auto", "low", "normal")

# --------------------------------------------------------------------------- ComfyUI process
START_TIMEOUT_S = 900.0  # first start on Windows: torch import, custom nodes, file scans
STOP_WAIT_S = 15.0
POLL_S = 0.5  # /history polling while a view is made
READY_POLL_S = 0.5  # /system_stats polling while ComfyUI starts
VRAM_MODES = ("lowvram", "novram", "cpu")
PRECISIONS = ("auto", "fp32")
REINSTALL_HINT = "Reinstall the Qwen photo editor add-on from the Add-ons dialog."
FALLBACK_HINT = "Use 'Your own back photo' or the built-in back."
OOM_HINT = ("Close other programs that use the graphics card or a lot of memory and try again, or "
            "use 'Your own back photo' or the built-in back.")

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


# =========================================================================== pure helpers
def wrap_phi(phi_deg: float) -> float:
    """``phi`` in [0, 360)."""
    v = float(phi_deg) % 360.0
    return 0.0 if math.isclose(v, 360.0) else v + 0.0


def azimuth_words(phi_deg: float, *, az_sign: int = AZ_SIGN) -> str:
    """The LoRA's azimuth words for the app's ``phi`` (nearest 45-degree step)."""
    lora_az = wrap_phi((1 if az_sign >= 0 else -1) * float(phi_deg))
    return AZIMUTH_WORDS[int(math.floor((lora_az + 22.5) / 45.0)) % 8]


def angle_prompt(phi_deg: float, *, extra: str = "", az_sign: int = AZ_SIGN,
                 elevation: str = ELEVATION_WORDS, distance: str = DISTANCE_WORDS) -> str:
    """The camera LoRA prompt for a view at the app's azimuth ``phi_deg``.

    ``"<sks> back view eye-level shot medium shot"`` for phi 180.  phi 90 ("you walked to your
    right", the subject's LEFT side visible) -> "right side view" with ``az_sign = +1``
    (:data:`AZ_SIGN`), "left side view" with -1.  Eye level and medium distance keep the input's
    framing.  ``extra`` ('Describe the hidden side') follows as its own sentence."""
    base = f"{TRIGGER} {azimuth_words(phi_deg, az_sign=az_sign)} {elevation} {distance}"
    tidy = " ".join(str(extra or "").split())[:MAX_PROMPT_CHARS].strip()
    return f"{base}. {tidy}" if tidy else base


def ram_tier(total_gb: float | None, *, min_gb: float = MIN_RAM_GB, low_gb: float = LOW_RAM_GB,
             slack_gb: float = RAM_SLACK_GB) -> str:
    """``"refuse"`` below ``min_gb`` of system RAM, ``"low"`` below ``low_gb`` (ComfyUI runs with
    ``--cache-none``), else ``"normal"``.  ``slack_gb`` absorbs what the OS / integrated graphics
    keep (a "16 GB" PC reports 15.4-15.9 GB).  Unknown RAM -> ``"low"`` (careful, not refused)."""
    if total_gb is None or not math.isfinite(float(total_gb)) or total_gb <= 0:
        return "low"
    if total_gb < min_gb - slack_gb:
        return "refuse"
    if total_gb < low_gb - slack_gb:
        return "low"
    return "normal"


_TQDM_RE = re.compile(r"\|\s*(\d+)\s*/\s*(\d+)\s*\[")
#: ANSI escape sequences (ComfyUI's logger colours every console line: ``\x1b[1m\x1b[31m[ERROR]\x1b[0m``).
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
#: What a server start says when its port was taken (ComfyUI v0.39.2's own line "Port N is already in
#: use on address ...", Python's "[Errno 98] Address already in use", Windows' "Only one usage of each
#: socket address ...").
_PORT_TAKEN = ("already in use", "only one usage of each socket")


def strip_ansi(text: str) -> str:
    """``text`` without terminal colour / cursor sequences."""
    return _ANSI_RE.sub("", str(text))


def port_taken(lines: Iterable[str]) -> bool:
    """True when a server's log says its network port was already taken."""
    for line in lines:
        low = strip_ansi(line).lower()
        if any(k in low for k in _PORT_TAKEN) and ("port" in low or "address" in low or "socket" in low):
            return True
    return False


def parse_tqdm(line: str | bytes) -> tuple[int, int] | None:
    """``(n, total)`` from a tqdm bar line (``" 38%|###   | 3/8 [00:42<01:10, 14.1s/it]"``), else
    None.  Only the bar form counts, so ``"1/2"`` in ordinary log text is ignored."""
    s = line.decode("utf-8", "replace") if isinstance(line, bytes) else str(line)
    matches = _TQDM_RE.findall(s)  # the last bar of a line holding several "\r" updates
    if not matches:
        return None
    n, total = int(matches[-1][0]), int(matches[-1][1])
    if total <= 0 or n < 0 or n > total:
        return None
    return n, total


def free_port(host: str = "127.0.0.1") -> int:
    """A TCP port that is free on ``host`` right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def substitute(graph: Any, mapping: Mapping[str, Any], *, strict: bool = False) -> Any:
    """``{{IMAGE}} {{PROMPT}} {{SEED}} {{AZIMUTH}} {{WIDTH}} {{HEIGHT}}`` placeholders of an expert
    API workflow (:func:`_comfy.substitute`; a value standing alone keeps its type)."""
    import _comfy

    return _comfy.substitute(graph, mapping, strict=strict)


def output_size(width: int, height: int, long_side: int, *, multiple: int = SIZE_MULTIPLE) -> tuple[int, int]:
    """``(w, h)`` with the input's aspect, the long side ``long_side`` and both multiples of
    ``multiple`` (at least :data:`MIN_SIDE` // 2)."""
    w, h = max(1, int(width)), max(1, int(height))
    s = float(long_side) / max(w, h)
    lo = max(multiple, MIN_SIDE // 2)
    out_w = max(lo, int(round(w * s / multiple)) * multiple)
    out_h = max(lo, int(round(h * s / multiple)) * multiple)
    return out_w, out_h


def scaled_size(width: int, height: int, scale: float, *, multiple: int = SIZE_MULTIPLE) -> tuple[int, int]:
    """``(w, h)`` times ``scale``, rounded to multiples of ``multiple``."""
    if scale >= 0.999:
        return int(width), int(height)
    return output_size(width, height, int(round(max(width, height) * scale)), multiple=multiple)


_QUANT_SUFFIX_RE = re.compile(r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$", re.IGNORECASE)


def mmproj_matches(clip_name: str, mmproj_name: str) -> bool:
    """ComfyUI-GGUF's pairing rule (``loader.gguf_mmproj_loader`` at the pinned commit): the vision
    file in the text encoder's folder must contain "mmproj" and the text encoder's file name without
    extension and quant suffix (``Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`` -> ``qwen2.5-vl-7b-instruct``)."""
    tenc = _QUANT_SUFFIX_RE.sub("", os.path.splitext(os.path.basename(clip_name))[0].lower())
    name, ext = os.path.splitext(os.path.basename(mmproj_name))
    return ext.lower() == ".gguf" and "mmproj" in name.lower() and tenc in name.lower()


def is_back_view(phi_deg: float) -> bool:
    """Views that get ``candidates`` tries: those the LoRA calls "back view" (180 +- 22.5)."""
    return abs(wrap_phi(phi_deg) - 180.0) < 22.5


def ladder_step(state: Mapping[str, Any], kind: str) -> str | None:
    """Next fallback after a failed view: ``kind`` "oom" / "crash" -> ``"novram"`` (restart ComfyUI
    with ``--novram``), then ``"smaller"`` (long side x 0.8), then None (skip the view); "black" ->
    ``"fp32"`` (restart with ``--force-fp32``) once, then None."""
    if kind in ("oom", "crash"):
        if state.get("vram_mode") == "lowvram":
            return "novram"
        if float(state.get("scale", 1.0)) > OOM_SCALE + 1e-6:
            return "smaller"
        return None
    if kind == "black":
        return "fp32" if state.get("precision") != "fp32" else None
    return None


def comfy_args(port: int, *, folders: Mapping[str, Path | str], yaml_path: Path | str, tier: str = "normal",
               vram_mode: str = "lowvram", precision: str = "auto") -> list[str]:
    """ComfyUI's command line after ``main.py`` (design 4.4 step 2).  ``folders``: ``input`` /
    ``output`` / ``temp`` / ``user`` directories."""
    if vram_mode not in VRAM_MODES:
        raise ValueError(f"unknown vram mode {vram_mode!r}")
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}")
    args = ["--listen", "127.0.0.1", "--port", str(int(port)), f"--{vram_mode}"]
    if vram_mode != "cpu":
        args += ["--reserve-vram", "1.0"]
    args += ["--disable-auto-launch", "--dont-print-server", "--fp32-vae", "--offline", "--disable-metadata",
             "--disable-comfy-compiler",
             "--preview-method", "none", "--database-url", "sqlite:///:memory:",
             "--extra-model-paths-config", str(yaml_path),
             "--input-directory", str(folders["input"]), "--output-directory", str(folders["output"]),
             "--temp-directory", str(folders["temp"]), "--user-directory", str(folders["user"])]
    if tier == "low":
        args.append("--cache-none")
    if precision == "fp32":
        args.append("--force-fp32")
    return args


def extra_model_paths_yaml(folders: Mapping[str, Iterable[Path | str]]) -> str:
    """ComfyUI's ``extra_model_paths.yaml``: one entry per folder kind (``diffusion_models``,
    ``text_encoders``, ``vae``, ``loras``), each a literal block with one folder per line (ComfyUI's
    loader splits the value on newlines; no escaping is needed for Windows paths)."""
    lines = ["# written by VR Photo Studio (qwenedit worker); ComfyUI reads it with --extra-model-paths-config",
             "vrphoto_qwenedit:"]
    for kind in ("diffusion_models", "text_encoders", "vae", "loras"):
        paths = list(dict.fromkeys(str(p) for p in folders.get(kind, ())))
        if not paths:
            continue
        lines.append(f"  {kind}: |")
        lines.extend(f"    {p}" for p in paths)
    return "\n".join(lines) + "\n"


def default_view_timeout(steps: int, cfg: float) -> float:
    """The time limit of one view when the request gives none: :data:`DEFAULT_VIEW_TIMEOUT_S` for up
    to :data:`TIMEOUT_REF_PASSES` model passes (steps, twice with CFG > 1), proportionally more
    above that (40 steps at CFG 4 without the Lightning LoRA: the maximum, 4 hours)."""
    passes = max(1, int(steps)) * (2 if float(cfg) > 1.0 else 1)
    t = DEFAULT_VIEW_TIMEOUT_S * passes / TIMEOUT_REF_PASSES
    return float(min(VIEW_TIMEOUT_RANGE[1], max(DEFAULT_VIEW_TIMEOUT_S, t)))


#: Node ids of the reference chain (picture 1 = the input, 2 / 3 = ``references``).
_REF_LOAD = ("10", "12", "13")
_REF_SCALE = ("11", "14", "15")
_REF_LATENT = ("30", "16", "17")
_POS_CHAIN = ("24", "25", "26")
_NEG_CHAIN = ("27", "28", "29")


def build_graph(view: Mapping[str, Any], *, seed: int, steps: int, cfg: float, prompt: str = "",
                files: Mapping[str, str | None], image: str, width: int, height: int,
                references: Sequence[str] = (), reference_sizes: Sequence[tuple[int, int]] | None = None,
                az_sign: int = AZ_SIGN, prefix: str = "qwenedit") -> dict[str, Any]:
    """The ComfyUI API graph of one view (ComfyUI's Qwen-Image-Edit-2511 template with GGUF loaders
    and the two LoRAs).

    ``view`` = ``{"name", "phi_deg"}``; ``prompt`` = the extra text ('Describe the hidden side'),
    the positive prompt is :func:`angle_prompt`; ``files`` = model file names as ComfyUI lists them
    (``unet``, ``clip``, ``vae``, ``angles_lora`` and ``lightning_lora`` - None: no Lightning LoRA);
    ``image`` / ``references`` = uploaded picture names (``LoadImage`` values); ``width`` x
    ``height`` = the output size; ``reference_sizes`` = the size of each reference latent (default:
    the output size).

    The reference latents are made explicitly (``ImageScale`` -> ``VAEEncode`` -> a chain of
    ``ReferenceLatent`` nodes on the positive and the negative conditioning), and the two
    ``TextEncodeQwenImageEditPlus`` nodes get NO ``vae``: with one connected, ComfyUI v0.39.2 would
    encode every picture at a fixed ~1 megapixel whatever its size, so ``long_side`` and the
    out-of-memory ladder's smaller size would hardly change the work.  The input's latent (picture
    1) is also the starting latent.  Node ids are fixed strings, so tests and logs can name them."""
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} reference pictures")
    sizes = list(reference_sizes) if reference_sizes is not None else [(width, height)] * len(references)
    if len(sizes) != len(references):
        raise ValueError("one reference size per reference picture")
    for key in ("unet", "clip", "vae", "angles_lora"):
        if not files.get(key):
            raise ValueError(f"missing model file name {key!r}")
    positive = angle_prompt(float(view["phi_deg"]), extra=prompt, az_sign=az_sign)
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(view.get("name") or "view")).strip("_") or "view"
    g: dict[str, Any] = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": files["unet"]}},
        "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": files["clip"], "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
        "4": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": files["angles_lora"], "strength_model": 1.0}},
    }
    model: list[Any] = ["4", 0]
    if files.get("lightning_lora"):
        g["5"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model, "lora_name": files["lightning_lora"], "strength_model": 1.0}}
        model = ["5", 0]
    g["6"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": SHIFT}}
    g["7"] = {"class_type": "CFGNorm", "inputs": {"model": ["6", 0], "strength": CFG_NORM}}
    pictures: dict[str, Any] = {}
    for k, (pic, (w, h)) in enumerate(zip([image, *references], [(width, height), *sizes], strict=True)):
        load, scale, latent = _REF_LOAD[k], _REF_SCALE[k], _REF_LATENT[k]
        g[load] = {"class_type": "LoadImage", "inputs": {"image": pic}}
        g[scale] = {"class_type": "ImageScale", "inputs": {"image": [load, 0], "upscale_method": "lanczos",
                                                          "width": int(w), "height": int(h), "crop": "disabled"}}
        g[latent] = {"class_type": "VAEEncode", "inputs": {"pixels": [scale, 0], "vae": ["3", 0]}}
        pictures[f"image{k + 1}"] = [scale, 0]  # the vision-language tokens (ComfyUI shrinks to 384 x 384)
    g["20"] = {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"clip": ["2", 0], "prompt": positive, **pictures}}
    g["21"] = {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"clip": ["2", 0], "prompt": "", **pictures}}
    pos: list[Any] = ["20", 0]
    neg: list[Any] = ["21", 0]
    for k in range(len(pictures)):  # picture 1, 2, 3 in order (ReferenceLatent appends)
        latent = [_REF_LATENT[k], 0]
        g[_POS_CHAIN[k]] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": latent}}
        g[_NEG_CHAIN[k]] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": latent}}
        pos, neg = [_POS_CHAIN[k], 0], [_NEG_CHAIN[k], 0]
    g["22"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": pos, "reference_latents_method": REFERENCE_METHOD}}
    g["23"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": neg, "reference_latents_method": REFERENCE_METHOD}}
    g["31"] = {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "seed": int(seed), "steps": int(steps), "cfg": float(cfg),
                          "sampler_name": SAMPLER, "scheduler": SCHEDULER, "positive": ["22", 0],
                          "negative": ["23", 0], "latent_image": ["30", 0], "denoise": 1.0}}
    g["32"] = {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["3", 0]}}
    g["33"] = {"class_type": "SaveImage", "inputs": {"images": ["32", 0], "filename_prefix": f"{prefix}_{name}"}}
    return g


# =========================================================================== parameters
@dataclass
class Settings:
    """Validated request parameters."""

    views: list[dict[str, Any]] = field(default_factory=lambda: [dict(v) for v in DEFAULT_VIEWS])
    candidates: int = DEFAULT_CANDIDATES
    seed: int = 0
    steps: int | None = None  # None: by the Lightning LoRA (4 / 40)
    cfg: float | None = None  # None: by the Lightning LoRA (1.0 / 4.0)
    prompt_extra: str = ""
    long_side: int = DEFAULT_LONG_SIDE
    ram_tier: str = "auto"
    lightning: str = "auto"
    az_sign: int = AZ_SIGN
    view_timeout_s: float | None = None  # None: default_view_timeout(steps, cfg)
    name: str = "views"
    warnings: list[str] = field(default_factory=list)


def _whole(v: Any) -> int:
    if isinstance(v, bool):
        raise ValueError(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and math.isfinite(v) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*[+-]?\d+\s*", v):
        return int(v)
    raise ValueError(v)


def _view_name(v: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(v or "")).strip("_")[:40]


def parse_views(raw: Any) -> list[dict[str, Any]]:
    """``[{"name", "phi_deg"}]`` (``azimuth_deg`` accepted too), phi wrapped into [0, 360), unique
    names; :class:`WorkerError` for anything malformed."""
    if raw is None or raw == []:
        return [dict(v) for v in DEFAULT_VIEWS]
    if not isinstance(raw, (list, tuple)):
        raise WorkerError("Invalid value for 'views': a list of {name, phi_deg} is expected.")
    if len(raw) > MAX_VIEWS:
        raise WorkerError(f"Too many views ({len(raw)}): at most {MAX_VIEWS} per run.")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for k, v in enumerate(raw):
        if not isinstance(v, Mapping):
            raise WorkerError(f"Invalid view #{k + 1}: {v!r} (use {{\"name\": ..., \"phi_deg\": ...}}).")
        phi = v.get("phi_deg", v.get("azimuth_deg"))
        try:
            phi_f = float(phi)
        except (TypeError, ValueError):
            phi_f = float("nan")
        if not math.isfinite(phi_f):
            raise WorkerError(f"Invalid view #{k + 1}: 'phi_deg' must be a number of degrees (got {phi!r}).")
        name = _view_name(v.get("name")) or f"az{int(round(wrap_phi(phi_f))):03d}"
        if name in seen:
            raise WorkerError(f"Two views are both called '{name}'.")
        seen.add(name)
        out.append({"name": name, "phi_deg": wrap_phi(phi_f)})
    return out


def parse_params(params: Mapping[str, Any] | None) -> Settings:
    """Validate the request parameters (numbers out of range are clamped like
    :func:`_common.param`; malformed values raise :class:`WorkerError`)."""
    p = dict(params or {})
    s = Settings()
    s.views = parse_views(p.get("views"))
    s.candidates = int(param(p, "candidates", DEFAULT_CANDIDATES, _whole, lo=CANDIDATES_RANGE[0],
                             hi=CANDIDATES_RANGE[1]))
    seed_raw = p.get("seed", 0)
    try:
        seed = _whole(0 if seed_raw is None else seed_raw)
    except ValueError:
        seed = -1
    if not 0 <= seed <= MAX_SEED:
        raise WorkerError(f"Invalid seed {seed_raw!r}: use a whole number from 0 to {MAX_SEED}.")
    s.seed = seed
    if p.get("steps") not in (None, 0, "0", ""):
        s.steps = int(param(p, "steps", DEFAULT_STEPS_LIGHTNING, _whole, lo=STEPS_RANGE[0], hi=STEPS_RANGE[1]))
    if p.get("cfg") not in (None, ""):
        cfg = param(p, "cfg", DEFAULT_CFG_LIGHTNING, float)
        if not math.isfinite(cfg):
            raise WorkerError(f"Invalid value for 'cfg': {p.get('cfg')!r}")
        s.cfg = min(CFG_RANGE[1], max(CFG_RANGE[0], float(cfg)))
    s.prompt_extra = " ".join(str(p.get("prompt_extra") or p.get("prompt") or "").split())[:MAX_PROMPT_CHARS]
    long_side = int(param(p, "long_side", DEFAULT_LONG_SIDE, _whole, lo=LONG_SIDE_RANGE[0], hi=LONG_SIDE_RANGE[1]))
    s.long_side = int(round(long_side / SIZE_MULTIPLE)) * SIZE_MULTIPLE
    tier = str(p.get("ram_tier") or "auto").strip().lower()
    if tier not in RAM_TIERS:
        raise WorkerError(f"Invalid value for 'ram_tier': {p.get('ram_tier')!r} (use {', '.join(RAM_TIERS)}).")
    s.ram_tier = tier
    light = p.get("lightning", "auto")
    if isinstance(light, bool):
        s.lightning = "on" if light else "off"
    else:
        lt = str(light if light is not None else "auto").strip().lower()
        mapping = {"auto": "auto", "true": "on", "on": "on", "1": "on", "yes": "on",
                   "false": "off", "off": "off", "0": "off", "no": "off"}
        if lt not in mapping:
            raise WorkerError(f"Invalid value for 'lightning': {light!r} (use auto, true or false).")
        s.lightning = mapping[lt]
    az = p.get("az_sign", AZ_SIGN)
    try:
        az_i = _whole(az)
    except ValueError:
        az_i = 0
    if az_i not in (1, -1):
        raise WorkerError(f"Invalid value for 'az_sign': {az!r} (use 1 or -1).")
    s.az_sign = az_i
    if p.get("view_timeout_s") not in (None, ""):
        timeout = param(p, "view_timeout_s", DEFAULT_VIEW_TIMEOUT_S, float)
        if not math.isfinite(timeout):
            raise WorkerError(f"Invalid value for 'view_timeout_s': {p.get('view_timeout_s')!r}")
        s.view_timeout_s = min(VIEW_TIMEOUT_RANGE[1], max(VIEW_TIMEOUT_RANGE[0], float(timeout)))
    s.name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(p.get("name") or "views")).strip("_")[:40] or "views"
    return s


@dataclass(frozen=True)
class Job:
    """One graph to run: a view and a candidate number."""

    index: int
    view: dict[str, Any]
    candidate: int
    candidates: int
    seed: int


def plan_jobs(views: Sequence[Mapping[str, Any]], *, candidates: int, seed: int) -> list[Job]:
    """Every (view, candidate) in order; back views get ``candidates`` tries (seeds ``seed + k``),
    the others one (``seed``)."""
    jobs: list[Job] = []
    for v in views:
        n = int(candidates) if is_back_view(float(v["phi_deg"])) else 1
        for k in range(n):
            jobs.append(Job(len(jobs), dict(v), k, n, int(seed) + k))
    return jobs


# =========================================================================== system / files
def total_ram_gb() -> float | None:
    """Physical memory in GiB (None when it cannot be read)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MemoryStatusEx()
            st.dwLength = ctypes.sizeof(MemoryStatusEx)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
                return None
            return st.ullTotalPhys / 1024**3
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and size > 0:
            return pages * size / 1024**3
    except (OSError, ValueError, AttributeError):
        return None
    return None


def comfy_dir() -> Path:
    """Where the installer extracted ComfyUI (``COMFYUI_DIR``)."""
    return Path(os.environ.get("COMFYUI_DIR") or src_dir("ComfyUI"))


def models_dir() -> Path:
    """Where the installer put the model files (``QWENEDIT_MODELS``)."""
    return Path(os.environ.get("QWENEDIT_MODELS") or addon_dir() / "MODELS")


def _nonempty(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def find_models(root: Path) -> dict[str, Path | None]:
    """``kind -> path`` of each model file below ``root`` (by file name; the shallowest match)."""
    found: dict[str, Path | None] = dict.fromkeys(MODEL_FILES)
    if not root.is_dir():
        return found
    wanted = {name.lower(): kind for kind, (_folder, name) in MODEL_FILES.items()}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]  # skip .cache/huggingface
        for fn in filenames:
            kind = wanted.get(fn.lower())
            if kind is None:
                continue
            p = Path(dirpath) / fn
            old = found[kind]
            if _nonempty(p) and (old is None or len(p.parts) < len(old.parts)):
                found[kind] = p
    return found


def missing_files(comfy: Path, models: Mapping[str, Path | None]) -> list[str]:
    """Required ComfyUI / model files that are missing or empty (display names); the vision file
    must also sit next to the text encoder and match its name (:func:`mmproj_matches`)."""
    out = [f"ComfyUI/{f}" for f in COMFY_FILES if not (comfy / f).is_file()]
    for kind, (_folder, name) in MODEL_FILES.items():
        if kind not in OPTIONAL_MODELS and models.get(kind) is None:
            out.append(name)
    clip, mmproj = models.get("clip"), models.get("mmproj")
    if clip is not None and mmproj is not None:
        if mmproj.parent != clip.parent or not mmproj_matches(clip.name, mmproj.name):
            out.append(f"{MMPROJ_FILE} next to {CLIP_FILE}")
    return out


def require_files(comfy: Path, models: Mapping[str, Path | None]) -> None:
    missing = missing_files(comfy, models)
    if missing:
        more = f" and {len(missing) - 4} more" if len(missing) > 4 else ""
        raise WorkerError(f"Qwen photo editor files are missing: {', '.join(missing[:4])}{more}.", hint=REINSTALL_HINT)


def model_folders(models: Mapping[str, Path | None]) -> dict[str, list[Path]]:
    """ComfyUI folder kind -> the folders holding our files (so ComfyUI lists them by bare name)."""
    out: dict[str, list[Path]] = {}
    for kind, (folder_kind, _name) in MODEL_FILES.items():
        p = models.get(kind)
        if p is not None and kind != "mmproj":
            out.setdefault(folder_kind, [])
            if p.parent not in out[folder_kind]:
                out[folder_kind].append(p.parent)
    return out


def load_picture(path: Path, *, max_side: int | None = None, size: tuple[int, int] | None = None
                 ) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a picture and composite it on :data:`GREY` (RGB uint8).  ``size``: resized to exactly
    ``(w, h)``; ``max_side``: shrunk so the long side is at most that.  Returns ``(rgb, info)`` with
    ``info = {"width", "height", "transparent", "warnings"}`` (the original size)."""
    from PIL import Image

    try:
        im = Image.open(path)
        im.load()
    except Exception as e:
        raise WorkerError(f"Cannot read the picture {Path(path).name}: {e}") from e
    w0, h0 = im.size
    warnings: list[str] = []
    rgba = im.convert("RGBA") if ("A" in im.getbands() or im.mode in ("P", "LA")) else None
    if rgba is not None:
        alpha = np.asarray(rgba)[..., 3]
        if not np.any(alpha > 0):
            raise WorkerError(f"The picture {Path(path).name} is empty (fully transparent).",
                              hint="The cut-out of the subject failed; try another photo.")
        transparent = bool(np.any(alpha < 255))
    else:
        transparent = False
    target = size
    if target is None and max_side and max(w0, h0) > max_side:
        s = max_side / max(w0, h0)
        target = (max(1, round(w0 * s)), max(1, round(h0 * s)))
    if rgba is not None:
        if target is not None and target != (w0, h0):  # premultiplied resampling: no dark fringes
            rgba = rgba.convert("RGBa").resize(target, Image.Resampling.LANCZOS).convert("RGBA")
        a = np.asarray(rgba).astype(np.float32) / 255.0
        grey = np.asarray(GREY, np.float32) / 255.0
        out = a[..., :3] * a[..., 3:4] + (1.0 - a[..., 3:4]) * grey
        rgb = (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    else:
        rgb_im = im.convert("RGB")
        if target is not None and target != (w0, h0):
            rgb_im = rgb_im.resize(target, Image.Resampling.LANCZOS)
        rgb = np.asarray(rgb_im, dtype=np.uint8)
    if not transparent:
        warnings.append(f"{Path(path).name} has no transparent background, so it was used as it is (the Qwen "
                        "photo editor expects a cut-out).")
    return rgb, {"width": int(w0), "height": int(h0), "transparent": transparent, "warnings": warnings}


def save_png(rgb: np.ndarray, path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(rgb)).save(path)
    return path


def is_black(path: Path) -> bool:
    """True for a (nearly) black picture - what NaNs in ComfyUI turn into."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32)
    except Exception:
        return False
    return bool(a.size) and float(a.mean()) < 3.0 and float(a.max()) < 16.0


# =========================================================================== the ComfyUI process
def _parent_death_preexec() -> Callable[[], None] | None:
    """Linux: a ``preexec_fn`` that makes the kernel kill ComfyUI when this worker dies
    (``PR_SET_PDEATHSIG``).  ``prctl`` is looked up here, in the parent, so the forked child only
    makes one foreign call (no imports or lookups between fork and exec)."""
    try:
        import ctypes
        import signal

        prctl = ctypes.CDLL("libc.so.6", use_errno=True).prctl
        sig = int(signal.SIGKILL)
    except Exception:
        return None

    def preexec() -> None:  # pragma: no cover - runs in the child
        prctl(1, sig)  # PR_SET_PDEATHSIG

    return preexec


class ComfyServer:
    """A private ComfyUI child process (start / stop / restart) and its log reader.

    Its stdout and stderr go to one pipe; a reader thread splits it on ``\\r`` / ``\\n``, sends tqdm
    bars to :attr:`on_step` and everything else to the add-on log, and keeps the last lines for
    error messages."""

    def __init__(self, comfy: Path, work: Path, yaml_path: Path, *, tier: str = "normal",
                 python: str | None = None, start_timeout_s: float = START_TIMEOUT_S,
                 log_fn: Callable[[str, str], None] | None = None):
        self.comfy = Path(comfy)
        self.work = Path(work)
        self.yaml_path = Path(yaml_path)
        self.tier = tier
        self.python = python or sys.executable
        self.start_timeout_s = float(start_timeout_s)
        self.log_fn = log_fn or (lambda msg, level="info": log(msg, level))
        self.folders = {k: self.work / k for k in ("input", "output", "temp", "user")}
        self.proc: subprocess.Popen | None = None
        self.client: Any = None
        self.port = 0
        self.vram_mode = "lowvram"
        self.precision = "auto"
        self.version = ""
        self.stats: dict[str, Any] = {}
        self.tail: collections.deque[str] = collections.deque(maxlen=60)
        self.on_step: Callable[[int, int], None] | None = None
        self.starts = 0
        self._reader: threading.Thread | None = None

    # ----------------------------------------------------------------- process
    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def exited(self, timeout_s: float = 0.0) -> bool:
        """True when the process has ended (waiting up to ``timeout_s`` for it: a dying ComfyUI
        drops its connections a moment before the OS reports its exit)."""
        proc = self.proc
        if proc is None:
            return True
        try:
            proc.wait(timeout=max(0.0, timeout_s))
            return True
        except subprocess.TimeoutExpired:
            return False

    def tail_text(self, n: int = 8) -> str:
        lines = [x for x in list(self.tail)[-n:] if x.strip()]
        return " | ".join(lines)[-1200:]

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                    "DO_NOT_TRACK": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1"})
        env.pop("TQDM_DISABLE", None)
        return env

    def _read(self, stream: Any) -> None:
        buf = b""
        while True:
            try:
                chunk = stream.read(4096)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                self._line(raw)
        if buf:
            self._line(buf)

    def _line(self, raw: bytes) -> None:
        line = strip_ansi(raw.decode("utf-8", "replace")).rstrip()
        if not line.strip():
            return
        step = parse_tqdm(line)
        if step is not None:
            cb = self.on_step
            if cb is not None:
                try:
                    cb(*step)
                except Exception:
                    pass
            return
        self.tail.append(line)
        low = line.lower()
        level = "warning" if ("error" in low or "traceback" in low or "out of memory" in low) else "info"
        self.log_fn(f"[comfyui] {line[:500]}", level)

    def start(self, *, vram_mode: str = "lowvram", precision: str = "auto", cancel: Any = None) -> Any:
        """Start ComfyUI and wait until it answers; returns its :class:`_comfy.ComfyClient`."""
        import _comfy

        self.stop()
        for f in self.folders.values():
            f.mkdir(parents=True, exist_ok=True)
        last_error = ""
        for _attempt in range(3):  # another program may grab the free port before ComfyUI binds it
            self.port = free_port()
            cmd = [self.python, "-u", str(self.comfy / "main.py"),
                   *comfy_args(self.port, folders=self.folders, yaml_path=self.yaml_path, tier=self.tier,
                               vram_mode=vram_mode, precision=precision)]
            kwargs: dict[str, Any] = {}
            if sys.platform == "win32":
                # no console window; NOT a new process group / breakaway: ComfyUI stays in the app's
                # kill-on-close Job object together with this worker
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            elif sys.platform.startswith("linux"):
                preexec = _parent_death_preexec()
                if preexec is not None:
                    kwargs["preexec_fn"] = preexec
            self.log_fn("starting ComfyUI: " + " ".join(cmd[2:]), "info")
            self.tail.clear()
            self.proc = subprocess.Popen(cmd, cwd=str(self.comfy), env=self._env(), stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, **kwargs)
            self.starts += 1
            self._reader = threading.Thread(target=self._read, args=(self.proc.stdout,), name="comfyui-log",
                                            daemon=True)
            self._reader.start()
            self.vram_mode, self.precision = vram_mode, precision
            self.client = _comfy.ComfyClient(f"http://127.0.0.1:{self.port}", timeout=30)
            try:
                self.stats = self.client.wait_ready(self.start_timeout_s, cancel=cancel, alive=lambda: self.alive,
                                                    poll_s=READY_POLL_S)
            except _comfy.ComfyError as e:
                self.stop()  # joins the log reader first: ComfyUI's last lines say why it ended
                text = self.tail_text(3)[-400:]
                if port_taken(list(self.tail)):  # another program took the port before ComfyUI bound it
                    self.log_fn(f"ComfyUI's port {self.port} was taken; trying another one", "warning")
                    last_error = text
                    continue
                if e.kind == "timeout":
                    raise WorkerError(f"ComfyUI did not start within {self.start_timeout_s:.0f} s.",
                                      hint=REINSTALL_HINT) from e
                raise WorkerError("ComfyUI stopped while starting" + (f": {text}" if text else "."),
                                  hint=REINSTALL_HINT) from e
            self.version = str((self.stats.get("system") or {}).get("comfyui_version") or "")
            return self.client
        raise WorkerError(f"ComfyUI could not open a network port on this PC ({last_error[-300:]}).",
                          hint="Restart the PC or close programs that use many network ports, then try again.")

    def stop(self) -> None:
        """End the ComfyUI process (terminate, then kill) and its log reader.  Never raises."""
        proc, self.proc = self.proc, None
        self.client = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(STOP_WAIT_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(STOP_WAIT_S)
        except Exception as e:
            self.log_fn(f"stopping ComfyUI: {type(e).__name__}: {e}", "warning")
        finally:
            if self._reader is not None:
                self._reader.join(timeout=5)
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass

    def check(self, models: Mapping[str, Path | None]) -> None:
        """Every node of :func:`build_graph` exists and the loaders list our model files
        (:class:`WorkerError` otherwise)."""
        import _comfy

        try:
            missing = self.client.missing_nodes(REQUIRED_NODES)
        except _comfy.ComfyError as e:
            raise WorkerError(f"ComfyUI did not answer as expected ({e}).", hint=REINSTALL_HINT) from e
        if missing:
            gguf = [n for n in missing if "GGUF" in n]
            what = "the ComfyUI-GGUF nodes did not load" if gguf else "this ComfyUI is too old or damaged"
            raise WorkerError(f"ComfyUI is missing nodes ({', '.join(missing)}): {what}.", hint=REINSTALL_HINT)
        lists = {"unet": ("UnetLoaderGGUF", "unet_name"), "clip": ("CLIPLoaderGGUF", "clip_name"),
                 "vae": ("VAELoader", "vae_name"), "angles_lora": ("LoraLoaderModelOnly", "lora_name"),
                 "lightning_lora": ("LoraLoaderModelOnly", "lora_name")}
        unseen = []
        for kind, (node, inp) in lists.items():
            p = models.get(kind)
            if p is None:
                continue
            try:
                choices = self.client.choices(node, inp)
            except _comfy.ComfyError as e:
                raise WorkerError(f"ComfyUI did not answer as expected ({e}).", hint=REINSTALL_HINT) from e
            if p.name not in choices:
                unseen.append(p.name)
        if unseen:
            raise WorkerError(f"ComfyUI cannot see the model files {', '.join(unseen)}.", hint=REINSTALL_HINT)


# =========================================================================== the task
class JobFailed(Exception):
    """One view could not be made; ``kind`` = "oom" | "crash" | "black" | "timeout" | "error"."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _label(job: Job) -> str:
    name = str(job.view["name"]).replace("_", " ")
    tries = f", candidate {job.candidate + 1}/{job.candidates}" if job.candidates > 1 else ""
    return f"{name} view{tries}"


def generate(jobs: Sequence[Job], *, run_one: Callable[[Job, dict[str, Any]], Path],
             restart: Callable[[dict[str, Any]], None], state: dict[str, Any],
             warnings: list[str], on_done: Callable[[Job, Path, dict[str, Any]], None]) -> list[dict[str, Any]]:
    """Run every job with the fallback ladder (:func:`ladder_step`).

    ``run_one(job, state)`` returns the finished picture or raises :class:`JobFailed` (it starts
    ComfyUI again itself when it is not running); ``restart(state)`` restarts ComfyUI with the
    state's ``vram_mode`` / ``precision``.  A view the ladder cannot save is skipped with a warning.
    Returns the skipped jobs ``[{"name", "candidate", "reason"}]``.  Raises :class:`WorkerError`
    when every job failed, and at once when the first view fails for another reason than memory or
    a black picture (the setup is broken: the other views would fail the same way)."""
    skipped: list[dict[str, Any]] = []
    first_error: JobFailed | None = None
    done = 0
    for job in jobs:
        while True:
            try:
                path = run_one(job, state)
            except JobFailed as e:
                step = ladder_step(state, e.kind)
                if step == "novram":
                    log(f"{_label(job)}: {e}; restarting ComfyUI with --novram", "warning")
                    warnings.append("The Qwen photo editor ran out of memory and continued with ComfyUI's "
                                    "--novram mode (slower).")
                    state["vram_mode"] = "novram"
                    restart(state)
                    continue
                if step == "smaller":
                    state["scale"] = OOM_SCALE
                    log(f"{_label(job)}: {e}; continuing at {OOM_SCALE:g} x the size", "warning")
                    warnings.append(f"The Qwen photo editor ran out of memory; the views were made at "
                                    f"{OOM_SCALE * 100:.0f} % of the size.")
                    continue  # run_one restarts a crashed ComfyUI itself
                if step == "fp32":
                    state["precision"] = "fp32"
                    log(f"{_label(job)}: black picture; restarting ComfyUI with --force-fp32", "warning")
                    warnings.append("The Qwen photo editor made a black picture; it continued in full precision "
                                    "(--force-fp32, slower).")
                    restart(state)
                    continue
                first_error = first_error or e
                if not done and e.kind in ("error", "timeout"):
                    raise WorkerError(f"The Qwen photo editor could not make the {_label(job)} ({e}).",
                                      hint=FALLBACK_HINT) from e
                reason = str(e)
                skipped.append({"name": job.view["name"], "candidate": job.candidate, "reason": reason})
                warnings.append(f"The {_label(job)} was skipped ({reason}).")
                break
            on_done(job, path, state)
            done += 1
            break
    if not done:
        msg = str(first_error) if first_error else "no views were requested"
        hint = OOM_HINT if first_error is not None and first_error.kind in ("oom", "crash") else FALLBACK_HINT
        raise WorkerError(f"The Qwen photo editor could not make any view ({msg}).", hint=hint)
    return skipped


def handle(task: str, inputs: dict, params: dict, out_dir: Path, progress: Progress) -> dict:
    if task != TASK:
        raise WorkerError(f"Unknown task '{task}'.")
    # everything that can be checked without ComfyUI comes first
    image_path = require_input(inputs, "image")
    refs_raw = inputs.get("references") or []
    if isinstance(refs_raw, (str, os.PathLike)):
        refs_raw = [refs_raw]
    if len(refs_raw) > MAX_REFERENCES:
        raise WorkerError(f"Too many reference pictures ({len(refs_raw)}): at most {MAX_REFERENCES}.")
    ref_paths = []
    for k, r in enumerate(refs_raw):
        p = Path(r)
        if not p.is_file():
            raise WorkerError(f"Input file not found: {p} (reference picture {k + 1}).")
        ref_paths.append(p)
    s = parse_params(params)
    out_dir = Path(out_dir)
    ram = total_ram_gb()
    tier = ram_tier(ram) if s.ram_tier == "auto" else s.ram_tier
    if tier == "refuse":
        raise WorkerError(f"This PC has {ram:.0f} GB of memory (RAM); the Qwen photo editor needs at least "
                          f"{MIN_RAM_GB:.0f} GB.", hint=FALLBACK_HINT)
    comfy = comfy_dir()
    models = find_models(models_dir())
    require_files(comfy, models)
    warnings: list[str] = list(s.warnings)
    lightning = models.get("lightning_lora") is not None and s.lightning != "off"
    if s.lightning != "off" and models.get("lightning_lora") is None:  # its install step is optional
        warnings.append(f"The Lightning LoRA is not installed, so every view takes {DEFAULT_STEPS_BASE} steps at "
                        f"CFG {DEFAULT_CFG_BASE:g} instead of {DEFAULT_STEPS_LIGHTNING} at CFG "
                        f"{DEFAULT_CFG_LIGHTNING:g} (about 20 times slower). {REINSTALL_HINT}")
    steps = s.steps if s.steps is not None else (DEFAULT_STEPS_LIGHTNING if lightning else DEFAULT_STEPS_BASE)
    cfg = s.cfg if s.cfg is not None else (DEFAULT_CFG_LIGHTNING if lightning else DEFAULT_CFG_BASE)
    view_timeout_s = s.view_timeout_s if s.view_timeout_s is not None else default_view_timeout(steps, cfg)
    if tier == "low":
        warnings.append(f"This PC has {ram:.0f} GB of memory (RAM); ComfyUI runs without its cache (slower)."
                        if ram else "The PC's memory (RAM) could not be measured; ComfyUI runs without its cache.")

    # inputs: the cut-out on #c8c8c8 at the output size, references at most 1024 px
    rgb, info = load_picture(image_path)
    width, height = output_size(info["width"], info["height"], s.long_side)
    rgb, _ = load_picture(image_path, size=(width, height))
    warnings += info["warnings"]
    work = out_dir / f"{s.name}_comfyui"
    input_png = save_png(rgb, out_dir / f"{s.name}_input.png")
    ref_pngs = []
    ref_sizes = []  # each reference latent at the output's long side (scaled with the ladder's 0.8)
    for k, p in enumerate(ref_paths):
        ref_rgb, _ref_info = load_picture(p, max_side=1024)
        ref_pngs.append(save_png(ref_rgb, out_dir / f"{s.name}_reference{k + 1}.png"))
        ref_sizes.append(output_size(ref_rgb.shape[1], ref_rgb.shape[0], s.long_side))
    files = {kind: (p.name if p is not None else None) for kind, p in models.items()}
    if not lightning:
        files["lightning_lora"] = None
    yaml_path = work / "extra_model_paths.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(extra_model_paths_yaml(model_folders(models)), encoding="utf-8")
    jobs = plan_jobs(s.views, candidates=s.candidates, seed=s.seed)
    log(f"{len(jobs)} views to make at {width} x {height}, {steps} steps, CFG {cfg:g}, Lightning "
        f"{'on' if lightning else 'off'}, RAM {ram if ram is None else round(ram, 1)} GB ({tier}), seed {s.seed}, "
        f"at most {view_timeout_s:.0f} s per view")

    import _comfy

    server = ComfyServer(comfy, work, yaml_path, tier=tier)
    state: dict[str, Any] = {"vram_mode": "lowvram", "precision": "auto", "scale": 1.0, "uploads": {}}
    results: list[dict[str, Any]] = []
    partial = out_dir / "views.partial.jsonl"
    partial.unlink(missing_ok=True)  # this run's views only
    gen_lo, gen_hi = 0.12, 0.98
    span = (gen_hi - gen_lo) / max(1, len(jobs))

    def start_server(st: dict[str, Any]) -> None:
        if server.starts:
            progress(gen_lo + len(results) * span, "Restarting ComfyUI")
        server.start(vram_mode=st["vram_mode"], precision=st["precision"])
        dev = (server.stats.get("devices") or [{}])[0]
        if str(dev.get("type") or "cuda") == "cpu":
            raise WorkerError("ComfyUI could not use the graphics card.",
                              hint="This add-on needs an NVIDIA GPU with a recent driver (CUDA 12.8: driver 570 "
                                   "or newer). " + FALLBACK_HINT)
        server.check(models)
        try:
            st["uploads"] = {"image": server.client.upload_image(input_png),
                             "refs": [server.client.upload_image(p) for p in ref_pngs]}
        except _comfy.ComfyError as e:
            raise WorkerError(f"The picture could not be handed to ComfyUI ({e}).", hint=REINSTALL_HINT) from e
        log(f"ComfyUI {server.version or '?'} on {dev.get('name', '?')} ({st['vram_mode']}, {st['precision']})")

    def run_one(job: Job, st: dict[str, Any]) -> Path:
        if not server.alive:  # crashed while making the previous view
            start_server(st)
        lo = gen_lo + job.index * span
        label = _label(job)
        first = job.index == 0 and server.starts <= 1
        progress(lo, f"Making the {label}" + (" (loading the Qwen models first; this takes a few minutes)"
                                             if first else ""))
        w, h = scaled_size(width, height, float(st["scale"]))
        ref_wh = [scaled_size(rw, rh, float(st["scale"])) for rw, rh in ref_sizes]

        def on_step(n: int, total: int) -> None:
            if total == steps:
                progress(lo + span * (0.1 + 0.85 * n / total), f"Making the {label} (step {n}/{total})")

        graph = build_graph(job.view, seed=job.seed, steps=steps, cfg=cfg, prompt=s.prompt_extra, files=files,
                            image=st["uploads"]["image"], width=w, height=h, references=st["uploads"]["refs"],
                            reference_sizes=ref_wh, az_sign=s.az_sign, prefix=s.name)
        t0 = time.monotonic()
        server.on_step = on_step
        try:
            pid = server.client.queue(graph)
            entry = server.client.wait(pid, poll_s=POLL_S, timeout_s=view_timeout_s)
            refs = server.client.outputs(entry)
            if not refs:
                raise JobFailed("error", "ComfyUI returned no picture")
            dest = out_dir / f"{s.name}_{job.view['name']}_c{job.candidate}.png"
            path = server.client.download(refs[-1], dest)
        except _comfy.ComfyError as e:
            if server.exited(5.0 if e.kind == "connection" else 0.0):
                raise JobFailed("crash", "ComfyUI stopped unexpectedly"
                                + (f": {server.tail_text(4)}" if server.tail_text(4) else "")) from e
            if e.kind == "oom":
                raise JobFailed("oom", "out of memory") from e
            if e.kind == "timeout":
                raise JobFailed("timeout", f"it took longer than {view_timeout_s:.0f} s") from e
            if e.kind in ("rejected", "protocol", "http", "connection"):
                raise WorkerError(f"ComfyUI could not run the Qwen workflow: {e}", hint=REINSTALL_HINT) from e
            raise JobFailed("error", str(e)) from e
        finally:
            server.on_step = None
        if is_black(path):
            raise JobFailed("black", "the picture came out black")
        st["_seconds"] = time.monotonic() - t0
        return path

    def on_done(job: Job, path: Path, st: dict[str, Any]) -> None:
        w, h = scaled_size(width, height, float(st["scale"]))
        entry = {"name": job.view["name"], "phi_deg": job.view["phi_deg"], "file": str(path),
                 "candidate": job.candidate, "seed": job.seed}
        results.append(entry)
        line = {**entry, "width": w, "height": h, "steps": steps, "cfg": cfg,
                "prompt": angle_prompt(job.view["phi_deg"], extra=s.prompt_extra, az_sign=s.az_sign),
                "seconds": round(float(st.get("_seconds", 0.0)), 1)}
        with open(partial, "a", encoding="utf-8") as fh:  # one line per finished view (crash-safe)
            fh.write(json.dumps(line) + "\n")
        log(f"{_label(job)}: saved {path.name} in {line['seconds']:.0f} s")
        progress(gen_lo + (job.index + 1) * span, f"Made the {_label(job)}")

    try:
        progress(0.02, "Starting ComfyUI (a minute or two)")
        start_server(state)
        skipped = generate(jobs, run_one=run_one, restart=start_server, state=state, warnings=warnings,
                           on_done=on_done)
    finally:
        server.stop()
    progress(1.0, "Done")
    return {
        "views": results,
        "frame": "nominal",
        "warnings": list(dict.fromkeys(warnings)),
        "comfyui_version": server.version,
        "input": str(input_png),
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "lightning": lightning,
        "ram_tier": tier,
        "ram_gb": None if ram is None else round(ram, 1),
        "vram_mode": state["vram_mode"],
        "precision": state["precision"],
        "scale": state["scale"],
        "skipped": skipped,
        "az_sign": s.az_sign,
        "view_timeout_s": view_timeout_s,
        "worker_version": VERSION,
    }


def check(require_gpu: bool) -> dict:
    """Install check: PyTorch sees a supported GPU, every file is in place, and a ComfyUI start lists
    every node and model file (about a minute)."""
    info = cuda_info(require=require_gpu)
    comfy = comfy_dir()
    models = find_models(models_dir())
    require_files(comfy, models)
    ram = total_ram_gb()
    info["ram_gb"] = None if ram is None else round(ram, 1)
    info["ram_tier"] = ram_tier(ram)
    info["lightning"] = models.get("lightning_lora") is not None
    import tempfile

    with tempfile.TemporaryDirectory(prefix="qwenedit-check-") as tmp:
        work = Path(tmp)
        yaml_path = work / "extra_model_paths.yaml"
        yaml_path.write_text(extra_model_paths_yaml(model_folders(models)), encoding="utf-8")
        server = ComfyServer(comfy, work, yaml_path, tier="low")
        try:
            server.start(vram_mode="lowvram" if info.get("cuda") else "cpu")
            server.check(models)
            info["comfyui_version"] = server.version
            info["comfyui_devices"] = [d.get("name") for d in server.stats.get("devices") or []]
        finally:
            server.stop()
    if info["ram_tier"] == "refuse":
        info["warnings"] = [f"This PC has {ram:.0f} GB of memory (RAM); the Qwen photo editor needs at least "
                            f"{MIN_RAM_GB:.0f} GB and will refuse to run."]
    return info


if __name__ == "__main__":
    sys.exit(run(handle, name="qwenedit", version=VERSION, check=check, tasks=[TASK]))
