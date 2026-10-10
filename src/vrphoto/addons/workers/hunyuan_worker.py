"""Hunyuan3D-2 worker (experimental): one photo -> 3D shape (GLB), optional texture.

Runs inside the ``hunyuan`` add-on venv (Python 3.11, torch 2.8.0 + CUDA 12.8) with the
``hy3dgen`` package from https://github.com/Tencent-Hunyuan/Hunyuan3D-2 installed.

Task ``image_to_3d``:
    inputs:  ``image`` - PNG/JPG; RGBA with alpha is used as the object mask, otherwise the
             background is removed with rembg.
    params:  ``model`` ("mini-turbo" | "mini" | "2.0" | "2.1"), ``steps`` (5 for turbo, else 30),
             ``octree_resolution`` (380), ``guidance`` (5.0), ``seed`` (0), ``num_chunks``
             (20000), ``texture`` (false; paints a texture - needs the optional
             custom_rasterizer extension; the paint models download on the first textured
             run, with progress), ``max_faces`` (100000; 0 = keep), ``name`` ("model").
    outputs: ``glb``, ``textured`` (true/false as string), optionally ``warning``.

License: Tencent Hunyuan 3D 2.0 Community License - not valid in the EU, UK and South Korea.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from _common import (
    Progress,
    WorkerError,
    addon_dir,
    cuda_info,
    free_cuda,
    log,
    param,
    require_input,
    run,
    tqdm_factory,
)

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
DELIGHT_SUBFOLDER = "hunyuan3d-delight-v2-0"
PAINT_COMPLETE = ".vrphoto-paint-complete"
PAINT_FULL_VRAM_GB = 20  # below this the paint models are offloaded to the CPU between steps


def flag(params: dict, name: str, default: bool = False) -> bool:
    """Boolean parameter that also accepts "true"/"false"/"1"/"0" strings."""
    v = params.get(name, default)
    if isinstance(v, str):
        v = v.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("", "0", "false", "no", "off", "none"):
            return False
        raise WorkerError(f"Invalid value for '{name}': {params.get(name)!r} (expected true or false).")
    return bool(v)


def paint_models_dir() -> Path:
    """Where the texture models live: ``$HY3DGEN_MODELS/tencent/Hunyuan3D-2`` - the local
    folder ``Hunyuan3DPaintPipeline.from_pretrained`` looks in before any download."""
    base = os.environ.get("HY3DGEN_MODELS") or str(addon_dir() / "models")
    return Path(base).expanduser() / PAINT_REPO


def ensure_paint_models(progress: Progress, *, download=None) -> Path:
    """Download the texture-painting models (delight + paint turbo, several GB) on first
    use, reporting progress; later runs use the local copy without any network access.

    ``download`` (tests) replaces ``huggingface_hub.snapshot_download``.  Raises
    :class:`WorkerError` when the download fails.
    """
    local = paint_models_dir()
    if (local / PAINT_COMPLETE).is_file():
        return local
    patterns = [f"{DELIGHT_SUBFOLDER}/*", f"{PAINT_SUBFOLDER}/*"]
    total = 0
    if download is None:
        from huggingface_hub import HfApi, snapshot_download

        download = snapshot_download
        try:
            from huggingface_hub.utils import filter_repo_objects

            info = HfApi().model_info(PAINT_REPO, files_metadata=True)
            sizes = {s.rfilename: s.size or 0 for s in (info.siblings or [])}
            total = sum(sizes[f] for f in filter_repo_objects(list(sizes), allow_patterns=patterns))
        except Exception as e:  # size unknown: progress without percentage
            log(f"No size information for the texture models: {e}", "warning")
    local.mkdir(parents=True, exist_ok=True)
    errors: list[BaseException] = []

    def fetch() -> None:
        try:
            download(repo_id=PAINT_REPO, allow_patterns=patterns, local_dir=str(local), max_workers=4)
        except BaseException as e:  # noqa: BLE001 - reported to the main thread
            errors.append(e)

    t = threading.Thread(target=fetch, name="paint-models", daemon=True)
    t.start()
    msg = "Downloading the texture models (first textured run only)"
    while t.is_alive():
        t.join(1.0)
        if total:
            done = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
            progress(min(0.99, done / total), f"{msg}: {done / 1e9:.1f} of {total / 1e9:.1f} GB")
        else:
            progress(0.0, msg)
    if errors:
        e = errors[0]
        raise WorkerError(f"The texture models could not be downloaded ({type(e).__name__}: {e}).",
                          hint="Check your internet connection and try again; the download resumes.")
    missing = [d for d in (DELIGHT_SUBFOLDER, PAINT_SUBFOLDER) if not (local / d).is_dir()]
    if missing:
        raise WorkerError(f"The texture models are incomplete (missing {', '.join(missing)}).",
                          hint="Try again; the download resumes where it stopped.")
    (local / PAINT_COMPLETE).write_text("ok\n", encoding="utf-8")
    progress(1.0, "Texture models ready")
    return local


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
    want_texture = flag(params, "texture")
    name = str(params.get("name") or "model")

    gpu = cuda_info(require=True)
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
        paint = None
        try:
            import custom_rasterizer  # noqa: F401
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            models = ensure_paint_models(progress.sub(0.76, 0.85))
            progress(0.86, "Painting the texture")
            paint = Hunyuan3DPaintPipeline.from_pretrained(str(models), subfolder=PAINT_SUBFOLDER)
            if float(gpu.get("vram_gb") or 0) < PAINT_FULL_VRAM_GB:
                paint.enable_model_cpu_offload()  # upstream's low-VRAM mode
            mesh = paint(mesh, image=image)
            textured = True
            paint = None
            free_cuda()
        except ImportError as e:
            msg = (f"Texture painting is unavailable ({e}): the optional texture-painting extension "
                   "was not built when the add-on was installed. Exported the untextured shape.")
            log(msg, "warning")
            result["warning"] = msg
        except WorkerError as e:
            msg = f"{e} Exported the untextured shape."
            log(msg, "warning")
            result["warning"] = msg
        except Exception as e:  # e.g. CUDA out of memory: keep the shape that is already made
            oom = "out of memory" in str(e).lower()
            msg = (f"Texture painting failed ({type(e).__name__}: {str(e)[:300]}); exported the untextured shape."
                   + (" The GPU ran out of memory: close other apps or turn off Paint texture." if oom else ""))
            log(msg, "warning")
            result["warning"] = msg
            paint = None
            free_cuda()
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
    info["texture_models_downloaded"] = (paint_models_dir() / PAINT_COMPLETE).is_file()
    return info


if __name__ == "__main__":
    sys.exit(run(handle, name="hunyuan", version=VERSION, check=check, tasks=["image_to_3d"]))
