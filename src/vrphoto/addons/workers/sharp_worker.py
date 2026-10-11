"""Apple SHARP worker (experimental, research-only weights): one photo -> 3D Gaussian splat.

Runs inside the ``sharp`` add-on venv (Python 3.11, torch 2.8.0 + CUDA 12.8) with the
``sharp`` package from https://github.com/apple/ml-sharp installed (without gsplat - the
renderer is not needed) and the checkpoint at ``SHARP_CHECKPOINT``.

Task ``image_to_splat``:
    inputs:  ``image`` - JPG/PNG/HEIC.
    params:  ``hfov_deg`` (horizontal field of view; default: from EXIF, else 30 mm
             equivalent like the SHARP CLI), ``device`` ("auto" | "cuda" | "cpu"), ``name``.
    outputs: ``ply`` (3DGS PLY, OpenCV frame: x right, y down, z forward, metres),
             ``frame`` = "opencv", ``f_px``, ``width``, ``height``.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from _common import (
    Progress,
    WorkerError,
    addon_dir,
    cuda_info,
    free_cuda,
    gpu_arch_problem,
    log,
    param,
    require_input,
    require_supported_gpu,
    run,
)

VERSION = "1.0"
INTERNAL_SHAPE = (1536, 1536)
DEFAULT_CKPT = "sharp_2572gikvuh.pt"


def _checkpoint() -> Path:
    p = Path(os.environ.get("SHARP_CHECKPOINT") or (addon_dir() / "models" / DEFAULT_CKPT))
    if not p.is_file():
        raise WorkerError(f"The SHARP model file is missing ({p}).", hint="Reinstall the SHARP add-on.")
    return p


def _device(choice: str) -> str:
    import torch

    if choice in ("cuda", "cpu"):
        if choice == "cuda":
            if not torch.cuda.is_available():
                raise WorkerError("CUDA was requested but no NVIDIA GPU is available.")
            require_supported_gpu()
        return choice
    if not torch.cuda.is_available():
        return "cpu"
    problem = gpu_arch_problem()
    if problem is not None:  # the PyTorch build has no kernels for this GPU: CPU still works
        log(f"{problem[0]} Running SHARP on the CPU instead (slower).", "warning")
        return "cpu"
    return "cuda"


def _predict(predictor, image, f_px: float, device: str):
    """Same steps as ``sharp predict`` (resize to 1536^2, predict in NDC, unproject)."""
    import torch
    import torch.nn.functional as F
    from sharp.utils.gaussians import unproject_gaussians

    dev = torch.device(device)
    img = torch.from_numpy(image.copy()).float().to(dev).permute(2, 0, 1) / 255.0
    _, height, width = img.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(dev)
    resized = F.interpolate(img[None], size=(INTERNAL_SHAPE[1], INTERNAL_SHAPE[0]), mode="bilinear",
                            align_corners=True)
    with torch.no_grad():
        ndc = predictor(resized, disparity_factor)
        K = torch.tensor([[f_px, 0, width / 2, 0], [0, f_px, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float().to(dev)
        K_resized = K.clone()
        K_resized[0] *= INTERNAL_SHAPE[0] / width
        K_resized[1] *= INTERNAL_SHAPE[1] / height
        return unproject_gaussians(ndc, torch.eye(4).to(dev), K_resized, INTERNAL_SHAPE)


def handle(task: str, inputs: dict, params: dict, out_dir: Path, progress: Progress) -> dict:
    if task != "image_to_splat":
        raise WorkerError(f"Unknown task '{task}'.")
    image_path = require_input(inputs, "image")
    hfov = params.get("hfov_deg")
    device = _device(str(params.get("device") or "auto"))
    name = str(params.get("name") or "sharp")

    progress(0.02, "Loading SHARP")
    import torch
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io as sharp_io
    from sharp.utils.gaussians import save_ply

    state = torch.load(str(_checkpoint()), map_location="cpu", weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state)
    predictor.eval().to(device)
    progress(0.30, f"Model loaded ({device})")

    image, _icc, f_px = sharp_io.load_rgb(Path(image_path))
    height, width = image.shape[:2]
    if hfov is not None:
        h = param(params, "hfov_deg", 60.0, float, lo=10.0, hi=170.0)
        f_px = (width / 2.0) / math.tan(math.radians(h) / 2.0)
    progress(0.40, "Predicting the 3D scene")
    gaussians = _predict(predictor, image, float(f_px), device)
    progress(0.85, "Saving the splat")
    ply_path = out_dir / f"{name}.ply"
    save_ply(gaussians, float(f_px), (height, width), ply_path)
    del gaussians, predictor
    free_cuda()
    progress(1.0, "Done")
    return {"ply": str(ply_path), "frame": "opencv", "f_px": float(f_px), "width": int(width), "height": int(height)}


def check(require_gpu: bool) -> dict:
    import importlib

    # SHARP also runs on the CPU (a few seconds per photo), so a missing NVIDIA GPU is not
    # an install failure - the info just reports cuda = False
    info = cuda_info(require=False)
    info["gpu_required"] = require_gpu
    if info.get("gpu_supported") is False:  # unsupported GPU architecture: jobs use the CPU
        info["warnings"] = [f"{info.get('gpu_problem')} SHARP will run on the CPU (slower)."]
        log(info["warnings"][0], "warning")
    for mod in ("sharp.models", "sharp.utils.gaussians", "sharp.utils.io", "timm", "plyfile"):
        try:
            importlib.import_module(mod)
        except Exception as e:
            raise WorkerError(f"Broken SHARP environment ({mod}: {type(e).__name__}: {e})",
                              hint="Reinstall the SHARP add-on.") from e
    info["checkpoint"] = str(_checkpoint())
    return info


if __name__ == "__main__":
    sys.exit(run(handle, name="sharp", version=VERSION, check=check, tasks=["image_to_splat"]))
