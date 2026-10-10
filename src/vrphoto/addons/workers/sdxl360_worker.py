"""SDXL outpainting worker: fill the empty parts of a VR180 / 360 equirect image.

Runs inside the ``sdxl360`` add-on venv (Python 3.11, torch 2.8.0 + CUDA 12.8, diffusers).

Task ``outpaint_360``:
    inputs:  ``image`` - equirect PNG/JPG (360: 2:1, VR180 eye: 1:1);
             ``mask`` (optional) - PNG, white (> 127) = fill.  Without a mask, transparent
             pixels of an RGBA ``image`` are filled.
    params:  ``prompt`` (""), ``negative_prompt``, ``steps`` (30), ``strength`` (0.99),
             ``guidance`` (7.0), ``seed`` (0), ``work_width`` (2048 for 360 / 1024 for 180),
             ``seam_band`` (0.125 of the width), ``wrap`` ("auto" | "yes" | "no"), ``name``.
    outputs: ``image`` (PNG at the input resolution; unmasked pixels are unchanged).

Algorithm: masked regions are pre-filled with a smooth push-pull guess, the image is
inpainted at a working resolution, then (for full 360 images) rolled by half its width
so the left/right seam sits in the middle and a band around it is inpainted again; the
result is upscaled and composited into the original with an inward feather.

The pure-numpy helpers at the top are unit-tested in the app's test-suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from _common import Progress, WorkerError, cuda_info, free_cuda, log, param, require_input, run

VERSION = "1.0"
DEFAULT_PROMPT = "seamless 360 degree panorama, natural continuation of the scene, consistent lighting, high detail"
DEFAULT_NEGATIVE = "text, watermark, logo, frame, border, blurry, distorted, duplicate, seam, people"


# --------------------------------------------------------------------------- pure helpers
def is_full_360(width: int, height: int) -> bool:
    """True for a full 2:1 equirect panorama (whose left and right edges meet)."""
    return abs(width - 2 * height) <= max(2, int(0.01 * width))


def work_size(width: int, height: int, work_width: int) -> tuple[int, int]:
    """Working resolution: ``work_width`` wide (never upscaling), multiples of 8."""
    ww = min(int(work_width), int(width))
    wh = int(round(ww * height / width))
    ww = max(64, ww // 8 * 8)
    wh = max(64, wh // 8 * 8)
    return ww, wh


def roll_half(arr: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Roll horizontally by half the width (moves the 360 seam to the centre)."""
    w = arr.shape[1]
    shift = w // 2
    return np.roll(arr, -shift if inverse else shift, axis=1)


def seam_band_mask(height: int, width: int, band_frac: float) -> np.ndarray:
    """Bool mask of a vertical band of ``band_frac * width`` centred in the image."""
    band = max(8, int(round(band_frac * width)))
    m = np.zeros((height, width), dtype=bool)
    c = width // 2
    m[:, max(0, c - band // 2): min(width, c + band - band // 2)] = True
    return m


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur with edge clamping (float32, any channels)."""
    if r <= 0:
        return img.astype(np.float32)
    out = img.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r + 1, r)
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float64)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation by a square of radius ``r`` (box filter based)."""
    if r <= 0:
        return mask.copy()
    return _box_blur(mask.astype(np.float32), r) > 1e-6


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion by a square of radius ``r``."""
    if r <= 0:
        return mask.copy()
    return ~dilate(~mask, r)


def inward_feather(mask: np.ndarray, px: int) -> np.ndarray:
    """Float weight in [0,1]: 1 deep inside ``mask``, ramps down to 0 at its border and is
    exactly 0 outside (so compositing never changes unmasked pixels)."""
    m = mask.astype(bool)
    if px <= 0:
        return m.astype(np.float32)
    w = _box_blur(erode(m, px).astype(np.float32), px)
    w[~m] = 0.0
    return np.clip(w, 0.0, 1.0)


def prefill(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked pixels with a smooth push-pull estimate from the known pixels (uint8)."""
    known = ~mask.astype(bool)
    if known.all() or not known.any():
        return img.copy()
    levels = []
    cur = img.astype(np.float32) * known[..., None]
    wgt = known.astype(np.float32)
    while True:
        levels.append((cur, wgt))
        h, w = wgt.shape
        if h <= 1 and w <= 1:
            break
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        pc = np.pad(cur, ((0, h2 * 2 - h), (0, w2 * 2 - w), (0, 0)), mode="edge")
        pw = np.pad(wgt, ((0, h2 * 2 - h), (0, w2 * 2 - w)), mode="edge")
        cur = pc.reshape(h2, 2, w2, 2, -1).sum(axis=(1, 3))
        wgt = pw.reshape(h2, 2, w2, 2).sum(axis=(1, 3))
    fill_c, fill_w = levels[-1]
    est = fill_c / np.maximum(fill_w, 1e-6)[..., None]
    for c, w in reversed(levels[:-1]):
        h, wd = w.shape
        up = np.repeat(np.repeat(est, 2, axis=0), 2, axis=1)[:h, :wd]
        here = c / np.maximum(w, 1e-6)[..., None]
        a = np.clip(w, 0.0, 1.0)[..., None]
        est = a * here + (1.0 - a) * up
    out = img.astype(np.float32).copy()
    out[~known] = est[~known]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def composite(original: np.ndarray, generated: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``original * (1 - w) + generated * w`` (uint8)."""
    w = weight.astype(np.float32)[..., None]
    out = original.astype(np.float32) * (1.0 - w) + generated.astype(np.float32) * w
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def load_inputs(image_path: Path, mask_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """RGB uint8 image and bool fill-mask (True = fill)."""
    from PIL import Image

    im = Image.open(image_path)
    im.load()
    rgba = np.asarray(im.convert("RGBA"))
    rgb = np.ascontiguousarray(rgba[..., :3])
    if mask_path is not None:
        m = np.asarray(Image.open(mask_path).convert("L").resize(im.size, Image.Resampling.NEAREST)) > 127
    elif "A" in im.mode:
        m = rgba[..., 3] < 128
    else:
        raise WorkerError("No mask given and the image has no transparency - nothing to fill.")
    return rgb, m


# --------------------------------------------------------------------------- diffusion
def _load_pipeline(progress: Progress):
    import os

    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline

    model = os.environ.get("SDXL_INPAINT_MODEL") or "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    vae_id = os.environ.get("SDXL_VAE_MODEL") or "madebyollin/sdxl-vae-fp16-fix"
    kwargs = {"torch_dtype": torch.float16, "variant": "fp16", "use_safetensors": True}
    try:
        kwargs["vae"] = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
    except Exception as e:
        log(f"fp16 VAE not available ({e}); using the default VAE", "warning")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model, **kwargs)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram < 12.0:
        pipe.enable_model_cpu_offload()
        log(f"{vram:.1f} GB VRAM: model CPU offload enabled")
    else:
        pipe.to("cuda")
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _inpaint(pipe, rgb: np.ndarray, mask: np.ndarray, *, prompt: str, negative: str, steps: int,
             strength: float, guidance: float, seed: int, progress: Progress, msg: str) -> np.ndarray:
    import torch
    from PIL import Image

    h, w = mask.shape

    def on_step(_pipe, step, _t, cb_kwargs):
        progress((step + 1) / max(steps, 1), msg)
        return cb_kwargs

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=Image.fromarray(rgb),
        mask_image=Image.fromarray(mask.astype(np.uint8) * 255),
        width=w,
        height=h,
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=guidance,
        generator=gen,
        callback_on_step_end=on_step,
    ).images[0]
    out = np.asarray(result.convert("RGB"))
    if out.shape[:2] != (h, w):
        out = np.asarray(result.convert("RGB").resize((w, h), Image.Resampling.LANCZOS))
    return out


def handle(task: str, inputs: dict, params: dict, out_dir: Path, progress: Progress) -> dict:
    if task != "outpaint_360":
        raise WorkerError(f"Unknown task '{task}'.")
    from PIL import Image

    image_path = require_input(inputs, "image")
    mask_path = require_input(inputs, "mask") if inputs.get("mask") else None
    rgb, fill = load_inputs(image_path, mask_path)
    H, W = fill.shape
    if not fill.any():
        out = out_dir / f"{params.get('name') or 'outpainted'}.png"
        Image.fromarray(rgb).save(out)
        return {"image": str(out)}
    wrap_opt = str(params.get("wrap") or "auto").lower()
    wrap = is_full_360(W, H) if wrap_opt == "auto" else wrap_opt in ("yes", "true", "1")
    prompt = str(params.get("prompt") or "").strip() or DEFAULT_PROMPT
    negative = str(params.get("negative_prompt") or DEFAULT_NEGATIVE)
    steps = param(params, "steps", 30, int, lo=1, hi=150)
    strength = param(params, "strength", 0.99, float, lo=0.05, hi=1.0)
    guidance = param(params, "guidance", 7.0, float, lo=0.0, hi=20.0)
    seed = param(params, "seed", 0, int)
    work_w = param(params, "work_width", 2048 if wrap else 1024, int, lo=256, hi=4096)
    band = param(params, "seam_band", 0.125, float, lo=0.02, hi=0.5)
    name = str(params.get("name") or "outpainted")

    cuda_info(require=True)
    progress(0.02, "Loading SDXL inpainting")
    pipe = _load_pipeline(progress)
    progress(0.15, "Model loaded")

    ww, wh = work_size(W, H, work_w)
    small = np.asarray(Image.fromarray(prefill(rgb, fill)).resize((ww, wh), Image.Resampling.LANCZOS))
    small_mask = np.asarray(Image.fromarray(fill.astype(np.uint8) * 255).resize((ww, wh), Image.Resampling.NEAREST)) > 127
    grow = max(2, ww // 256)
    work_mask = dilate(small_mask, grow)
    gen = _inpaint(pipe, small, work_mask, prompt=prompt, negative=negative, steps=steps, strength=strength,
                   guidance=guidance, seed=seed, progress=progress.sub(0.15, 0.70 if wrap else 0.90),
                   msg="Outpainting")
    result_small = composite(small, gen, inward_feather(work_mask, grow) if grow else work_mask)
    if wrap:
        rolled = roll_half(result_small)
        seam = seam_band_mask(wh, ww, band) & roll_half(dilate(small_mask, grow))
        if seam.any():
            gen2 = _inpaint(pipe, rolled, seam, prompt=prompt, negative=negative, steps=steps,
                            strength=min(strength, 0.85), guidance=guidance, seed=seed + 1,
                            progress=progress.sub(0.70, 0.90), msg="Blending the 360 seam")
            rolled = composite(rolled, gen2, inward_feather(seam, max(4, int(band * ww / 6))))
            result_small = roll_half(rolled, inverse=True)
    del pipe
    free_cuda()
    progress(0.92, "Compositing")
    big = np.asarray(Image.fromarray(result_small).resize((W, H), Image.Resampling.LANCZOS))
    feather_px = max(2, W // 512)
    final = composite(rgb, big, inward_feather(fill, feather_px))
    out = out_dir / f"{name}.png"
    Image.fromarray(final).save(out)
    progress(1.0, "Done")
    return {"image": str(out), "wrapped": "true" if wrap else "false"}


def check(require_gpu: bool) -> dict:
    import importlib

    info = cuda_info(require=require_gpu)
    for mod in ("diffusers", "transformers", "accelerate", "safetensors"):
        try:
            importlib.import_module(mod)
        except Exception as e:
            raise WorkerError(f"Broken SDXL environment ({mod}: {type(e).__name__}: {e})",
                              hint="Reinstall the outpainting add-on.") from e
    import diffusers

    info["diffusers"] = diffusers.__version__
    return info


if __name__ == "__main__":
    sys.exit(run(handle, name="sdxl360", version=VERSION, check=check, tasks=["outpaint_360"]))
