"""Apple SHARP worker (experimental, research-only weights): one photo -> 3D Gaussian splat.

Runs inside the ``sharp`` add-on venv (Python 3.11, torch 2.8.0 + CUDA 12.8) with the
``sharp`` package from https://github.com/apple/ml-sharp installed (without gsplat - the
renderer is not needed) and the checkpoint at ``SHARP_CHECKPOINT``.

Task ``image_to_splat``:
    inputs:  ``image`` - JPG/PNG/HEIC; and / or ``images`` - a list of pictures (SHARP 360
             runs SHARP on the back views it got, design 3.5.6 / M6).  With both, ``image``
             comes first unless the list already holds it.  Neither -> "Missing input 'image'".
    params:  ``hfov_deg`` (horizontal field of view in degrees, 10-170; a number for every
             picture or a list with one value - or null - per picture; default: from EXIF,
             else 30 mm equivalent like the SHARP CLI), ``device`` ("auto" | "cuda" | "cpu"),
             ``name``.
    outputs: ``ply`` (3DGS PLY, OpenCV frame: x right, y down, z forward, metres),
             ``frame`` = "opencv", ``f_px``, ``width``, ``height`` - of ``image`` (exactly
             these keys, plus ``warnings`` after a CPU fallback); with ``images`` they
             describe the first picture and the per-picture lists ``plys``, ``f_pxs``,
             ``widths``, ``heights``, ``devices`` (and ``warnings``) are added, in input order
             (files ``<name>_<i>.ply``).

The model is loaded once for all pictures.  A picture that runs the graphics card out of
memory is predicted on the CPU (slower) after freeing the cached GPU memory; the next one
tries the GPU again.
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

VERSION = "1.1"  # 1.1: list input (``images``), per-picture CPU fallback on out of memory
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


def is_oom(exc: BaseException) -> bool:
    """True for a CUDA / allocator out-of-memory error (``torch.cuda.OutOfMemoryError`` & co.)."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc)
    return ("out of memory" in text.lower() or "CUBLAS_STATUS_ALLOC_FAILED" in text
            or "cudaErrorMemoryAllocation" in text)


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0][:200] if text else type(exc).__name__


def input_images(inputs: dict) -> tuple[list[Path], bool]:
    """``(pictures, list_mode)``: ``images`` (a list of paths; with ``image`` too, that one
    first unless the list holds it) or the single ``image`` -> :class:`WorkerError` for a
    missing input, a malformed list or a file that does not exist."""
    images = inputs.get("images")
    if images is None or (isinstance(images, (list, tuple)) and not images):
        return [require_input(inputs, "image")], False
    if isinstance(images, (str, os.PathLike)):
        images = [images]
    if not isinstance(images, (list, tuple)):
        raise WorkerError(f"Input 'images' must be a list of picture files (got {type(images).__name__}).")
    paths: list[Path] = []
    for i, v in enumerate(images):
        if not isinstance(v, (str, os.PathLike)) or not str(v).strip():
            raise WorkerError(f"Input 'images' item {i + 1} is not a picture file ({v!r}).")
        p = Path(v)
        if not p.exists():
            raise WorkerError(f"Input file not found: {p}")
        paths.append(p)
    if inputs.get("image"):
        first = require_input(inputs, "image")
        if not any(_same_file(first, p) for p in paths):
            paths.insert(0, first)
    return paths, True


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def hfov_list(value: object, n: int) -> list[float | None]:
    """One field of view (degrees, clamped to 10-170) or None (EXIF / default) per picture
    from ``hfov_deg``: a number for all of them or a list with one entry per picture."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            what = "picture" if n == 1 else "pictures"
            raise WorkerError(f"'hfov_deg' has {len(value)} values for {n} {what}; give one per picture "
                              "or a single number for all of them.")
        values = list(value)
    else:
        values = [value] * n
    return [None if v is None else float(param({"hfov_deg": v}, "hfov_deg", 60.0, float, lo=10.0, hi=170.0))
            for v in values]


def _place(predictor, device: str, warnings: list[str]) -> str:
    """Move the model to ``device``; out of memory on the GPU -> the CPU for every picture."""
    oom = None
    try:
        predictor.to(device)
        return device
    except Exception as e:
        if device != "cuda" or not is_oom(e):
            raise
        oom = _first_line(e)
    log(f"the SHARP model does not fit on the graphics card ({oom}); using the CPU", "warning")
    warnings.append("The SHARP model did not fit on the graphics card, so it ran on the CPU (slower).")
    predictor.to("cpu")
    free_cuda()
    return "cpu"


def predict_with_fallback(predictor, image, f_px: float, device: str, *, label: str,
                          warnings: list[str]) -> tuple[object, str, str]:
    """``(gaussians, device used, device for the next picture)``.  Out of memory on the GPU
    -> free the cached memory, predict this picture on the CPU and move the model back to
    the GPU for the next one (if that runs out of memory too, it stays on the CPU)."""
    oom = None
    try:
        return _predict(predictor, image, f_px, device), device, device
    except Exception as e:
        if device != "cuda" or not is_oom(e):
            raise
        oom = _first_line(e)
    # freed AFTER the except block: inside it the traceback still references the frames that
    # hold the failed attempt's CUDA tensors
    log(f"{label}: the graphics card ran out of memory ({oom}); predicting it on the CPU", "warning")
    warnings.append(f"The graphics card ran out of memory for {label}, so SHARP made it on the CPU (slower).")
    predictor.to("cpu")
    free_cuda()
    gaussians = _predict(predictor, image, f_px, "cpu")
    back = None
    try:
        predictor.to("cuda")
        return gaussians, "cpu", "cuda"
    except Exception as e:
        if not is_oom(e):
            raise
        back = _first_line(e)
    log(f"the model could not go back to the GPU ({back}); the remaining pictures use the CPU", "warning")
    predictor.to("cpu")
    free_cuda()
    return gaussians, "cpu", "cpu"


def handle(task: str, inputs: dict, params: dict, out_dir: Path, progress: Progress) -> dict:
    if task != "image_to_splat":
        raise WorkerError(f"Unknown task '{task}'.")
    # everything that can be checked without torch comes first
    images, list_mode = input_images(inputs)
    hfovs = hfov_list(params.get("hfov_deg"), len(images))
    name = str(params.get("name") or "sharp")
    device = _device(str(params.get("device") or "auto"))

    progress(0.02, "Loading SHARP")
    import torch
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io as sharp_io
    from sharp.utils.gaussians import save_ply

    state = torch.load(str(_checkpoint()), map_location="cpu", weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state)
    del state
    warnings: list[str] = []
    device = _place(predictor.eval(), device, warnings)
    progress(0.30, f"Model loaded ({device})")

    n = len(images)
    plys: list[str] = []
    f_pxs: list[float] = []
    widths: list[int] = []
    heights: list[int] = []
    used: list[str] = []
    for i, (path, hfov) in enumerate(zip(images, hfovs, strict=True)):
        sub = progress.sub(0.30 + 0.70 * i / n, 0.30 + 0.70 * (i + 1) / n) if list_mode else progress
        of = f" (picture {i + 1}/{n})" if list_mode else ""
        image, _icc, f_px = sharp_io.load_rgb(Path(path))
        height, width = image.shape[:2]
        if hfov is not None:
            f_px = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)
        sub(0.40 if not list_mode else 0.0, f"Predicting the 3D scene{of}")
        label = f"picture {i + 1} ({Path(path).name})" if list_mode else "the picture"
        gaussians, dev, device = predict_with_fallback(predictor, image, float(f_px), device, label=label,
                                                       warnings=warnings)
        sub(0.85, f"Saving the splat{of}")
        ply_path = out_dir / (f"{name}_{i}.ply" if list_mode else f"{name}.ply")
        save_ply(gaussians, float(f_px), (height, width), ply_path)
        del gaussians, image
        free_cuda()
        plys.append(str(ply_path))
        f_pxs.append(float(f_px))
        widths.append(int(width))
        heights.append(int(height))
        used.append(dev)
    del predictor
    free_cuda()
    progress(1.0, "Done")
    out: dict = {"ply": plys[0], "frame": "opencv", "f_px": f_pxs[0], "width": widths[0], "height": heights[0]}
    if list_mode:
        out.update(plys=plys, f_pxs=f_pxs, widths=widths, heights=heights, devices=used, warnings=warnings)
    elif warnings:
        out["warnings"] = warnings
    return out


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
