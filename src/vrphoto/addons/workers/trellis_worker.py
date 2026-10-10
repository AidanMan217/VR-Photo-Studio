"""TRELLIS v1 worker: one photo -> textured GLB + 3D Gaussian PLY.

Runs inside the ``trellis`` add-on venv (Python 3.11, torch 2.7.0 + CUDA 12.8) with the
Windows-friendly fp16 fork https://github.com/IgorAherne/trellis-stable-projectorz
extracted to ``<addon_dir>/src/trellis`` (the working directory).

Task ``image_to_3d``:
    inputs:  ``image`` - path of a PNG/JPG; an RGBA image with a real alpha channel is used
             as-is (pre-segmented by the app), otherwise the pipeline removes the background.
    params:  ``seed`` (int, 0), ``ss_steps`` (12), ``ss_cfg`` (7.5), ``slat_steps`` (12),
             ``slat_cfg`` (3.0), ``simplify`` (0.95 = ratio of faces removed), ``texture_size``
             (1024), ``fp16`` (true), ``glb`` (true), ``ply`` (true), ``name`` ("model").
    outputs: ``glb`` (textured mesh, glTF y-up), ``ply`` (3D Gaussians), ``preview`` (the
             preprocessed 518x518 input).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from _common import (
    Progress,
    WorkerError,
    add_sys_path,
    cuda_info,
    free_cuda,
    log,
    param,
    require_input,
    run,
    src_dir,
)

VERSION = "1.0"
DEFAULT_MODEL = "microsoft/TRELLIS-image-large"

os.environ.setdefault("ATTN_BACKEND", "xformers")  # must be set before trellis is imported
os.environ.setdefault("SPCONV_ALGO", "native")


def _setup_path() -> Path:
    src = src_dir("trellis")
    if not (src / "trellis" / "__init__.py").is_file():
        raise WorkerError(f"TRELLIS sources are missing in {src}.", hint="Reinstall the TRELLIS add-on.")
    add_sys_path(src)
    return src


def _model_id() -> str:
    return os.environ.get("TRELLIS_MODEL") or DEFAULT_MODEL


def _load_pipeline(fp16: bool):
    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipe = TrellisImageTo3DPipeline.from_pretrained(_model_id())
    if fp16:
        pipe.to(torch.float16)  # halves VRAM (the fork supports fp16 end-to-end)
        if "image_cond_model" in pipe.models:
            pipe.models["image_cond_model"].half()
    # the fork moves sub-models to the GPU one at a time inside run(); keep them on the CPU
    return pipe


class _StageTqdm:
    """Patches the sampler's tqdm so each sampling stage reports into its own range."""

    def __init__(self, ranges: list[tuple[Progress, str]]):
        self.ranges = ranges
        self.calls = 0

    def factory(self):
        from _common import tqdm_factory

        outer = self

        def make(iterable=None, *args, **kwargs):
            idx = min(outer.calls, len(outer.ranges) - 1)
            outer.calls += 1
            prog, msg = outer.ranges[idx]
            return tqdm_factory(prog, msg)(iterable, *args, **kwargs)

        return make


def handle(task: str, inputs: dict, params: dict, out_dir: Path, progress: Progress) -> dict:
    if task != "image_to_3d":
        raise WorkerError(f"Unknown task '{task}'.")
    image_path = require_input(inputs, "image")
    seed = param(params, "seed", 0, int)
    ss_steps = param(params, "ss_steps", 12, int, lo=1, hi=100)
    ss_cfg = param(params, "ss_cfg", 7.5, float, lo=0.0, hi=20.0)
    slat_steps = param(params, "slat_steps", 12, int, lo=1, hi=100)
    slat_cfg = param(params, "slat_cfg", 3.0, float, lo=0.0, hi=20.0)
    simplify = param(params, "simplify", 0.95, float, lo=0.0, hi=0.99)
    texture_size = param(params, "texture_size", 1024, int, lo=256, hi=4096)
    fp16 = bool(params.get("fp16", True))
    want_glb = bool(params.get("glb", True))
    want_ply = bool(params.get("ply", True))
    name = str(params.get("name") or "model")

    cuda_info(require=True)
    _setup_path()
    progress(0.02, "Loading TRELLIS")
    import trellis.pipelines.samplers.flow_euler as flow_euler
    from PIL import Image
    from trellis.utils import postprocessing_utils

    pipe = _load_pipeline(fp16)
    progress(0.15, "Model loaded")

    image = Image.open(image_path)
    image.load()
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.mode else "RGB")

    stages = _StageTqdm([(progress.sub(0.20, 0.45), "Generating the shape"),
                         (progress.sub(0.45, 0.70), "Generating details")])
    flow_euler.tqdm = stages.factory()

    progress(0.17, "Preparing the image")
    preview = pipe.preprocess_image(image)
    preview_path = out_dir / f"{name}_input.png"
    preview.save(preview_path)
    outputs = pipe.run(
        preview,
        seed=seed,
        sparse_structure_sampler_params={"steps": ss_steps, "cfg_strength": ss_cfg},
        slat_sampler_params={"steps": slat_steps, "cfg_strength": slat_cfg},
        formats=["mesh", "gaussian"] if want_glb else ["gaussian"],
        preprocess_image=False,
    )
    progress(0.75, "Decoded 3D representations")
    free_cuda()
    result: dict[str, str] = {"preview": str(preview_path)}
    gaussian = outputs["gaussian"][0]
    if want_ply:
        ply_path = out_dir / f"{name}_gaussians.ply"
        gaussian.save_ply(str(ply_path))
        result["ply"] = str(ply_path)
        progress(0.78, "Saved Gaussians")
    if want_glb:
        progress(0.80, "Baking the textured mesh (takes a minute)")
        glb = postprocessing_utils.to_glb(
            gaussian,
            outputs["mesh"][0],
            simplify=simplify,
            texture_size=texture_size,
            verbose=False,
        )
        glb_path = out_dir / f"{name}.glb"
        glb.export(str(glb_path))
        result["glb"] = str(glb_path)
    del outputs
    free_cuda()
    progress(1.0, "Done")
    return result


_IMPORTS = ("xformers", "spconv.pytorch", "nvdiffrast.torch", "diff_gaussian_rasterization", "utils3d",
            "xatlas", "pyvista", "pymeshfix", "igraph", "rembg", "plyfile")


def check(require_gpu: bool) -> dict:
    import importlib

    info = cuda_info(require=require_gpu)
    failures: dict[str, str] = {}
    for mod in _IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failures[mod] = f"{type(e).__name__}: {e}"
    _setup_path()
    if not failures or require_gpu:
        try:
            importlib.import_module("trellis.pipelines")
        except Exception as e:
            failures["trellis"] = f"{type(e).__name__}: {e}"
    if failures:
        detail = "; ".join(f"{k}: {v}" for k, v in failures.items())
        gpu_only = all(("DLL" in v or "CUDA" in v or "nvcuda" in v) for v in failures.values())
        if require_gpu or not gpu_only:
            raise WorkerError(f"Broken TRELLIS environment ({detail})", hint="Reinstall the TRELLIS add-on.")
        info["warnings"] = [f"{k} needs a GPU driver: {v}" for k, v in failures.items()]
    info["model"] = _model_id()
    return info


def prefetch(progress: Progress) -> dict:
    """Download the DINOv2 image encoder through torch.hub (cached in TORCH_HOME)."""
    import json

    import torch

    name = "dinov2_vitl14_reg"
    try:
        from huggingface_hub import hf_hub_download

        cfg = json.loads(Path(hf_hub_download(_model_id(), "pipeline.json")).read_text(encoding="utf-8"))
        name = cfg["args"].get("image_cond_model", name)
    except Exception as e:
        log(f"Could not read pipeline.json ({e}); using {name}", "warning")
    progress(0.05, f"Downloading {name}")
    torch.hub.load("facebookresearch/dinov2", name, pretrained=True, trust_repo=True, skip_validation=True)
    progress(0.85, "Downloading the background remover (u2net)")
    rembg_ok = True
    try:
        # the pipeline removes the background of non-RGBA photos with rembg's u2net (CPU);
        # fetch it now (into U2NET_HOME) so the first such run also works offline
        import rembg

        rembg.new_session("u2net", providers=["CPUExecutionProvider"])
    except Exception as e:
        rembg_ok = False
        log(f"Could not prefetch the u2net background remover ({e}); it downloads on first use", "warning")
    progress(1.0, f"{name} ready")
    return {"image_cond_model": name, "torch_home": os.environ.get("TORCH_HOME", ""), "u2net": rembg_ok}


if __name__ == "__main__":
    sys.exit(run(handle, name="trellis", version=VERSION, check=check, prefetch=prefetch, tasks=["image_to_3d"]))
