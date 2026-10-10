"""Hunyuan3D-2 worker (experimental): one photo -> 3D shape (GLB), optional texture.

Runs inside the ``hunyuan`` add-on venv (Python 3.11, torch 2.8.0 + CUDA 12.8) with the
``hy3dgen`` package from https://github.com/Tencent-Hunyuan/Hunyuan3D-2 installed.

Task ``image_to_3d``:
    inputs:  ``image`` - PNG/JPG; RGBA with alpha is used as the object mask, otherwise the
             background is removed with rembg.
    params:  ``model`` ("mini-turbo" | "mini" | "2.0" | "2.1"), ``steps`` (5 for turbo, else 30),
             ``octree_resolution`` (380), ``guidance`` (5.0), ``seed`` (0), ``num_chunks``
             (20000), ``texture`` (false; needs the optional custom_rasterizer extension),
             ``max_faces`` (100000; 0 = keep), ``name`` ("model").
    outputs: ``glb``, optionally ``textured`` (true/false as string) and ``warning``.

License: Tencent Hunyuan 3D 2.0 Community License - not valid in the EU, UK and South Korea.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _common import Progress, WorkerError, cuda_info, free_cuda, log, param, require_input, run, tqdm_factory

VERSION = "1.0"

#: model key -> (HF repo, subfolder, turbo)
MODELS = {
    "mini-turbo": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo", True),
    "mini": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini", False),
    "2.0": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0", False),
    "2.1": ("tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", False),
}
PAINT_REPO = "tencent/Hunyuan3D-2"
PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0-turbo"


def _load_shape_pipeline(key: str):
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    repo, sub, turbo = MODELS[key]
    last: Exception | None = None
    for use_safetensors in (True, False):
        try:
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                repo, subfolder=sub, use_safetensors=use_safetensors, variant="fp16",
                device="cuda", dtype=torch.float16,
            )
            break
        except (FileNotFoundError, OSError) as e:
            last = e
    else:
        raise WorkerError(f"Could not load the Hunyuan3D model {repo}/{sub}: {last}",
                          hint="Check your internet connection (the model downloads on first use) or reinstall.")
    if turbo:
        try:
            pipe.enable_flashvdm(topk_mode="merge")
        except Exception as e:
            log(f"FlashVDM not enabled: {e}", "warning")
    return pipe, turbo


def _prepare_image(path: Path):
    from PIL import Image

    img = Image.open(path)
    img.load()
    if img.mode == "RGBA":
        import numpy as np

        alpha = np.asarray(img)[:, :, 3]
        if (alpha < 250).any():
            return img
    from hy3dgen.rembg import BackgroundRemover

    return BackgroundRemover()(img.convert("RGB"))


def _reduce_faces(mesh, max_faces: int):
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh
    try:
        from hy3dgen.shapegen import FaceReducer, FloaterRemover

        mesh = FloaterRemover()(mesh)
        return FaceReducer()(mesh, max_facenum=max_faces)
    except Exception as e:
        log(f"Face reduction skipped: {e}", "warning")
        return mesh


def handle(task: str, inputs: dict, params: dict, out_dir: Path, progress: Progress) -> dict:
    if task != "image_to_3d":
        raise WorkerError(f"Unknown task '{task}'.")
    image_path = require_input(inputs, "image")
    key = str(params.get("model") or "mini-turbo")
    if key not in MODELS:
        raise WorkerError(f"Unknown Hunyuan3D model '{key}' (choose {', '.join(MODELS)}).")
    turbo = MODELS[key][2]
    steps = param(params, "steps", 5 if turbo else 30, int, lo=1, hi=100)
    octree = param(params, "octree_resolution", 380, int, lo=128, hi=512)
    guidance = param(params, "guidance", 5.0, float, lo=0.0, hi=20.0)
    seed = param(params, "seed", 0, int)
    chunks = param(params, "num_chunks", 20000, int, lo=1000, hi=200000)
    max_faces = param(params, "max_faces", 100_000, int, lo=0, hi=2_000_000)
    want_texture = bool(params.get("texture", False))
    name = str(params.get("name") or "model")

    cuda_info(require=True)
    import torch

    progress(0.02, "Loading Hunyuan3D (downloads on first use)")
    import hy3dgen.shapegen.pipelines as shape_pipelines

    pipe, _ = _load_shape_pipeline(key)
    progress(0.20, "Removing the background")
    image = _prepare_image(image_path)
    shape_pipelines.tqdm = tqdm_factory(progress.sub(0.25, 0.70), "Generating the shape")
    mesh = pipe(
        image=image,
        num_inference_steps=steps,
        octree_resolution=octree,
        guidance_scale=guidance,
        num_chunks=chunks,
        generator=torch.manual_seed(seed),
        output_type="trimesh",
    )[0]
    progress(0.75, "Cleaning the mesh")
    mesh = _reduce_faces(mesh, max_faces)
    del pipe
    free_cuda()
    result: dict[str, str] = {}
    textured = False
    if want_texture:
        try:
            import custom_rasterizer  # noqa: F401
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            progress(0.78, "Painting the texture (downloads paint models on first use)")
            paint = Hunyuan3DPaintPipeline.from_pretrained(PAINT_REPO, subfolder=PAINT_SUBFOLDER)
            mesh = paint(mesh, image=image)
            textured = True
            del paint
            free_cuda()
        except ImportError as e:
            msg = f"Texture painting is unavailable ({e}); exported the untextured shape."
            log(msg, "warning")
            result["warning"] = msg
    glb_path = out_dir / f"{name}.glb"
    mesh.export(str(glb_path))
    result.update({"glb": str(glb_path), "textured": "true" if textured else "false"})
    progress(1.0, "Done")
    return result


def check(require_gpu: bool) -> dict:
    import importlib

    info = cuda_info(require=require_gpu)
    for mod in ("hy3dgen", "hy3dgen.shapegen", "diffusers", "transformers", "pymeshlab", "skimage", "rembg"):
        try:
            importlib.import_module(mod)
        except Exception as e:
            raise WorkerError(f"Broken Hunyuan3D environment ({mod}: {type(e).__name__}: {e})",
                              hint="Reinstall the Hunyuan3D add-on.") from e
    try:
        importlib.import_module("custom_rasterizer")
        info["texture_painting"] = True
    except Exception:
        info["texture_painting"] = False
    return info


if __name__ == "__main__":
    sys.exit(run(handle, name="hunyuan", version=VERSION, check=check, tasks=["image_to_3d"]))
