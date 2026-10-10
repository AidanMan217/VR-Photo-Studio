"""TRELLIS.2 worker (experimental): one photo -> PBR-textured GLB.

Runs inside the ``trellis2`` add-on venv (Python 3.11, torch 2.8.0 + CUDA 12.8) with the
Windows build https://github.com/IgorAherne/TRELLIS.2-stableprojectorz extracted to
``<addon_dir>/src/trellis2``.  The working directory is the add-on folder, whose
``MODELS/dinov3`` and ``MODELS/RMBG-2.0`` (filled by the installer) the fork picks up.

Task ``image_to_3d``:
    inputs:  ``image`` - PNG/JPG; RGBA with a real alpha channel skips background removal.
    params:  ``seed`` (0), ``pipeline_type`` ("1024_cascade"; "512" is faster / lighter,
             "1536_cascade" sharper), ``ss_steps`` / ``shape_steps`` / ``tex_steps`` (12),
             ``decimation_target`` (500000 faces), ``texture_size`` (2048), ``name`` ("model").
    outputs: ``glb`` (PBR mesh), ``preview`` (preprocessed input).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from _common import (
    Progress,
    WorkerError,
    add_sys_path,
    addon_dir,
    cuda_info,
    free_cuda,
    log,
    param,
    require_input,
    run,
    src_dir,
    tqdm_factory,
)

VERSION = "1.0"
DEFAULT_MODEL = "microsoft/TRELLIS.2-4B"
PIPELINE_TYPES = ("512", "1024", "1024_cascade", "1536_cascade")

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _setup_path() -> Path:
    src = src_dir("trellis2")
    if not (src / "trellis2" / "__init__.py").is_file():
        raise WorkerError(f"TRELLIS.2 sources are missing in {src}.", hint="Reinstall the TRELLIS.2 add-on.")
    add_sys_path(src)
    if not (addon_dir() / "MODELS" / "dinov3" / "config.json").is_file():
        raise WorkerError(
            "The DINOv3 image encoder is missing.",
            hint="Reinstall TRELLIS.2 with a Hugging Face token that has access to facebook/dinov3-vitl16-pretrain-lvd1689m.",
        )
    return src


def _apply_windows_patches() -> None:
    """flex_gemm on Windows: force the explicit-GEMM sparse conv and provide a torch
    fallback for the Triton gather kernel (same approach as the fork's pipeline worker)."""
    import torch

    try:
        import flex_gemm.ops.spconv as spconv
        from flex_gemm.ops.spconv import Algorithm

        spconv.ALGORITHM = Algorithm.EXPLICIT_GEMM
    except Exception as e:
        log(f"flex_gemm spconv patch not applied: {e}", "warning")
    try:
        import flex_gemm.kernels as fgk

        if not hasattr(fgk, "triton"):

            class _TritonFallback:
                @staticmethod
                def indice_weighed_sum_fwd(feats, indices, weights):
                    n = feats.shape[0]
                    idx = indices.long().clamp(min=0, max=n - 1)
                    out = torch.zeros((idx.shape[0], feats.shape[-1]), dtype=feats.dtype, device=feats.device)
                    for i in range(idx.shape[1]):
                        out += feats[idx[:, i]] * weights[:, i].unsqueeze(-1)
                    return out

                @staticmethod
                def indice_weighed_sum_bwd_input(grad_output, indices, weights, n):
                    c = grad_output.shape[1]
                    idx = indices.long().clamp(min=0, max=n - 1)
                    weighted = grad_output.unsqueeze(1) * weights.unsqueeze(-1)
                    grad = torch.zeros(n, c, device=grad_output.device, dtype=grad_output.dtype)
                    grad.scatter_add_(0, idx.reshape(-1, 1).expand(-1, c), weighted.reshape(-1, c))
                    return grad

            fgk.triton = _TritonFallback()
    except Exception as e:
        log(f"flex_gemm triton fallback not applied: {e}", "warning")


def handle(task: str, inputs: dict, params: dict, out_dir: Path, progress: Progress) -> dict:
    if task != "image_to_3d":
        raise WorkerError(f"Unknown task '{task}'.")
    image_path = require_input(inputs, "image")
    seed = param(params, "seed", 0, int)
    pipeline_type = str(params.get("pipeline_type") or "1024_cascade")
    if pipeline_type not in PIPELINE_TYPES:
        raise WorkerError(f"pipeline_type must be one of {', '.join(PIPELINE_TYPES)}")
    ss_steps = param(params, "ss_steps", 12, int, lo=1, hi=100)
    shape_steps = param(params, "shape_steps", 12, int, lo=1, hi=100)
    tex_steps = param(params, "tex_steps", 12, int, lo=1, hi=100)
    decimation = param(params, "decimation_target", 500_000, int, lo=10_000, hi=2_000_000)
    texture_size = param(params, "texture_size", 2048, int, lo=512, hi=4096)
    name = str(params.get("name") or "model")

    cuda_info(require=True)
    _setup_path()
    os.chdir(addon_dir())  # the fork resolves MODELS/... relative to the working directory
    _apply_windows_patches()
    progress(0.02, "Loading TRELLIS.2 (first run takes a while)")
    import o_voxel
    import torch
    import trellis2.pipelines.samplers.flow_euler as flow_euler
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(os.environ.get("TRELLIS2_MODEL") or DEFAULT_MODEL)
    pipe.cuda()
    progress(0.15, "Model loaded")

    ranges = [progress.sub(0.20, 0.35), progress.sub(0.35, 0.55), progress.sub(0.55, 0.70), progress.sub(0.70, 0.75)]
    calls = {"n": 0}

    def staged_tqdm(iterable=None, *args, **kwargs):
        i = min(calls["n"], len(ranges) - 1)
        calls["n"] += 1
        return tqdm_factory(ranges[i], str(kwargs.get("desc") or "Generating"))(iterable, *args, **kwargs)

    flow_euler.tqdm = staged_tqdm

    image = Image.open(image_path)
    image.load()
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.mode else "RGB")
    progress(0.17, "Preparing the image")
    preview = pipe.preprocess_image(image)
    preview_path = out_dir / f"{name}_input.png"
    preview.save(preview_path)
    meshes = pipe.run(
        preview,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params={"steps": ss_steps},
        shape_slat_sampler_params={"steps": shape_steps},
        tex_slat_sampler_params={"steps": tex_steps},
        pipeline_type=pipeline_type,
    )
    mesh = meshes[0]
    mesh.simplify(16777216)  # nvdiffrast limit
    progress(0.78, "Exporting the textured mesh")
    for model in pipe.models.values():
        model.cpu()
    free_cuda()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs.float(),
        coords=mesh.coords,
        attr_layout=pipe.pbr_attr_layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    glb_path = out_dir / f"{name}.glb"
    glb.export(str(glb_path))
    del mesh, meshes, glb
    free_cuda()
    torch.cuda.synchronize()
    progress(1.0, "Done")
    return {"glb": str(glb_path), "preview": str(preview_path)}


_IMPORTS = ("xformers", "o_voxel", "cumesh", "flex_gemm", "nvdiffrast.torch", "utils3d", "transformers", "kornia", "timm")


def check(require_gpu: bool) -> dict:
    import importlib

    info = cuda_info(require=require_gpu)
    failures: dict[str, str] = {}
    for mod in _IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failures[mod] = f"{type(e).__name__}: {e}"
    try:
        _setup_path()
    except WorkerError as e:
        failures["models"] = str(e)
    if failures:
        detail = "; ".join(f"{k}: {v}" for k, v in failures.items())
        gpu_only = all(("DLL" in v or "CUDA" in v or "nvcuda" in v) for v in failures.values())
        if require_gpu or not gpu_only:
            raise WorkerError(f"Broken TRELLIS.2 environment ({detail})", hint="Reinstall the TRELLIS.2 add-on.")
        info["warnings"] = [f"{k} needs a GPU driver: {v}" for k, v in failures.items()]
    return info


if __name__ == "__main__":
    sys.exit(run(handle, name="trellis2", version=VERSION, check=check, tasks=["image_to_3d"]))
