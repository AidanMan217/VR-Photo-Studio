"""Run a fixed-size inpainting model on crops around each hole.

Algorithm (``inpaint_by_crops``):

1. Connected components (8-connected) of the mask.  Each component gets a square crop
   centred on its bounding box: side = ``(1 + margin) * max(bbox)`` (at least ``min_crop``
   px and the bbox plus 8 px context on each side), shifted/clamped into the image.
2. Components whose crops overlap are merged (repeated until stable), so every crop
   contains only its own holes and crops never overlap.
3. Whole-image mode: if the mask covers ``whole_mask_frac`` of the image, or the crops
   would cover ``whole_frac`` of it and the whole image needs at most ``max_downscale``,
   the whole image is processed as one (downscaled) crop.
4. Tiled mode: a crop larger than ``max_downscale * size`` whose holes are thin (deepest
   hole pixel within a quarter tile of known pixels, e.g. the band behind a subject's
   outline) is split into tiles processed one after another at <= ``max_downscale``; each
   tile treats the holes filled by earlier tiles as known, so tiles join seamlessly.
   Large compact holes keep one downscaled crop (global context matters more there).
5. Per crop: holes are pre-filled from known pixels only (masked area-downscale +
   push-pull) so hole content never smears into the context, a non-square crop is
   reflect-padded to a square, resized to ``size`` (area when shrinking, cubic when
   enlarging), the mask resized conservatively (any coverage = hole), ``fn`` runs, and the
   result is resized back.
6. Seam blending: the difference between the original and the model output on the
   known pixels around the hole is extrapolated into the hole and faded out over
   ``feather`` px, so the fill meets the original surroundings without a visible seam.
7. Only masked pixels are pasted; unmasked pixels are bit-identical to the input.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np

from ..core.errors import InferenceError, InputError
from ..core.fill import push_pull_fill
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, ProgressReporter
from ..segment import _runtime as rt

log = get_logger(__name__)

__all__ = ["inpaint_by_crops", "plan_crops", "tile_cells", "masked_downscale", "seam_blend", "CropFn"]

#: ``fn(crop_rgb [size x size x 3 uint8], crop_mask [size x size bool, True = fill]) -> RGB``
CropFn = Callable[[RGB, np.ndarray], RGB]

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)

_MIN_CONTEXT = 8  # px of known context around a bbox, at least


def _square(box: Box, w: int, h: int, margin: float, min_crop: int) -> Box:
    x0, y0, x1, y1 = box
    ext = max(x1 - x0, y1 - y0)
    side = max(int(min_crop), int(math.ceil(ext * (1.0 + max(0.0, margin)))), ext + 2 * _MIN_CONTEXT)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def span(c: float, lim: int) -> tuple[int, int]:
        if side >= lim:
            return 0, lim
        a = int(round(c - side / 2.0))
        a = min(max(a, 0), lim - side)
        return a, a + side

    sx0, sx1 = span(cx, w)
    sy0, sy1 = span(cy, h)
    return sx0, sy0, sx1, sy1


def _group(squares: list[Box], w: int, h: int) -> list[int]:
    """Group id per square; overlapping squares (transitively) share a group."""
    scale = min(1.0, 1024.0 / max(w, h))
    cw, ch = int(math.ceil(w * scale)) + 1, int(math.ceil(h * scale)) + 1
    canvas = np.zeros((ch, cw), np.uint8)
    cells: list[tuple[int, int]] = []
    for x0, y0, x1, y1 in squares:
        a, b = int(math.floor(x0 * scale)), int(math.floor(y0 * scale))
        c = max(a + 1, int(math.ceil(x1 * scale)))
        d = max(b + 1, int(math.ceil(y1 * scale)))
        canvas[b:d, a:c] = 1
        cells.append((b, a))
    # rounding outward may merge squares that merely touch; merging is always safe
    _n, labels = cv2.connectedComponents(canvas, connectivity=4)
    return [int(labels[b, a]) for b, a in cells]


def plan_crops(
    mask: np.ndarray,
    *,
    margin: float = 1.0,
    min_crop: int = 128,
    whole_frac: float = 0.6,
    whole_mask_frac: float = 0.5,
    max_side: int | None = None,
) -> list[Box]:
    """Crop boxes ``(x0, y0, x1, y1)`` covering all masked pixels (see module docstring).

    ``max_side``: the whole image is only substituted for crops covering ``whole_frac``
    of it when its long side is at most ``max_side`` (None = always).  Returned boxes
    never overlap; ``[(0, 0, W, H)]`` means "whole image"."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return []
    if float(stats[1:, cv2.CC_STAT_AREA].sum()) >= whole_mask_frac * w * h:
        return [(0, 0, w, h)]
    st = stats[1:]
    boxes: list[Box] = [
        (int(x), int(y), int(x + bw), int(y + bh))
        for x, y, bw, bh in zip(
            st[:, cv2.CC_STAT_LEFT],
            st[:, cv2.CC_STAT_TOP],
            st[:, cv2.CC_STAT_WIDTH],
            st[:, cv2.CC_STAT_HEIGHT],
        )
    ]
    while True:
        squares = [_square(b, w, h, margin, min_crop) for b in boxes]
        if len(squares) == 1:
            break
        gid = _group(squares, w, h)
        groups: dict[int, list[int]] = {}
        for i, g in enumerate(gid):
            groups.setdefault(g, []).append(i)
        if len(groups) == len(boxes):
            break
        boxes = [
            (
                min(boxes[i][0] for i in idx),
                min(boxes[i][1] for i in idx),
                max(boxes[i][2] for i in idx),
                max(boxes[i][3] for i in idx),
            )
            for idx in groups.values()
        ]
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in squares)
    if area >= whole_frac * w * h and (max_side is None or max(w, h) <= max_side):
        return [(0, 0, w, h)]
    return squares


def tile_cells(box: Box, core: int) -> list[Box]:
    """Split ``box`` into a near-uniform grid of cells no larger than ``core`` px."""
    x0, y0, x1, y1 = box
    nx = max(1, math.ceil((x1 - x0) / core))
    ny = max(1, math.ceil((y1 - y0) / core))
    xs = np.linspace(x0, x1, nx + 1).round().astype(int)
    ys = np.linspace(y0, y1, ny + 1).round().astype(int)
    return [
        (int(xs[i]), int(ys[j]), int(xs[i + 1]), int(ys[j + 1]))
        for j in range(ny)
        for i in range(nx)
        if xs[i + 1] > xs[i] and ys[j + 1] > ys[j]
    ]


def _check_result(res: object, size: int) -> np.ndarray:
    if not isinstance(res, np.ndarray):
        raise InferenceError(f"Inpainting returned {type(res).__name__} instead of an image.")
    if res.ndim == 3 and res.shape[2] == 4:
        res = res[..., :3]
    if res.shape != (size, size, 3):
        raise InferenceError(
            f"Inpainting returned an image of shape {res.shape}; expected ({size}, {size}, 3)."
        )
    if res.dtype != np.uint8:
        res = np.clip(np.rint(np.nan_to_num(res.astype(np.float32))), 0, 255).astype(np.uint8)
    return res


def seam_blend(orig: np.ndarray, out: np.ndarray, hole: np.ndarray, feather: float) -> np.ndarray:
    """Fade the model-vs-original difference measured around the hole into the hole.

    The difference on the known ring around the hole (``2*feather+2`` px wide) is
    extrapolated with a normalised Gaussian (sigma = feather) and added to hole pixels
    with weight 1 at the border falling linearly to 0 at ``feather`` px inside."""
    if feather <= 0:
        return out
    known = ~hole
    if not known.any():
        return out
    f = float(feather)
    d_hole = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
    band = hole & (d_hole < f + 1.0)
    ys, xs = np.nonzero(band)
    if ys.size == 0:
        return out
    sigma = max(1.0, f)
    pad = int(math.ceil(3.0 * sigma + 2.0 * f + 3.0))
    h, w = hole.shape
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    k = known[y0:y1, x0:x1]
    d_known = cv2.distanceTransform(k.astype(np.uint8), cv2.DIST_L2, 3)
    ring = (k & (d_known <= 2.0 * f + 2.0)).astype(np.float32)
    o = out[y0:y1, x0:x1].astype(np.float32)
    delta = (orig[y0:y1, x0:x1].astype(np.float32) - o) * ring[..., None]
    num = cv2.GaussianBlur(delta, (0, 0), sigma)
    den = cv2.GaussianBlur(ring, (0, 0), sigma)
    sel = band[y0:y1, x0:x1]
    wgt = np.clip(1.0 - (d_hole[y0:y1, x0:x1][sel] - 1.0) / f, 0.0, 1.0)
    dsel = den[sel]
    corr = np.where(dsel[:, None] > 1e-4, num[sel] / np.maximum(dsel, 1e-4)[:, None], 0.0)
    res = out.copy()
    region = res[y0:y1, x0:x1]
    region[sel] = np.clip(np.rint(o[sel] + corr * wgt[:, None]), 0, 255).astype(np.uint8)
    return res


def _pad(a: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pads
    if not (top or bottom or left or right):
        return a
    return cv2.copyMakeBorder(a, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def masked_downscale(img: np.ndarray, hole: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Shrink ``img`` to ``w x h`` averaging only KNOWN pixels (``~hole``), so hole content
    (often the foreground being removed) never leaks into the context.

    Returns ``(small uint8, small_hole bool)``; the small hole is conservative (any hole
    coverage counts as hole) and is pre-filled smoothly from known pixels.  uint8 maths
    keeps this cheap for 12+ MP images."""
    img_z = img.copy()
    img_z[hole] = 0
    num = cv2.resize(img_z, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    if num.ndim == 2:
        num = num[..., None]
    cover = cv2.resize(hole.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    den = 1.0 - cover
    valid = den >= 0.5  # mostly known: num/den is accurate to ~1 level
    small = num / np.maximum(den, 1e-3)[..., None]
    if not valid.any():
        small = np.full(small.shape, 128.0, np.float32)
    elif not valid.all():
        small = push_pull_fill(small, valid)
    small_hole = cover > 1e-4  # any hole coverage -> hole (conservative)
    return np.clip(np.rint(small), 0, 255).astype(np.uint8), small_hole


def _model_input(
    sub: np.ndarray, hole: np.ndarray, pads: tuple[int, int, int, int], side: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """``size x size`` model input (holes pre-filled from known pixels only) + bool mask."""
    known = ~hole
    if side > size:
        return masked_downscale(_pad(sub, pads), _pad(hole.astype(np.uint8), pads) > 0, size, size)
    pre = push_pull_fill(sub, known) if known.any() else np.full_like(sub, 128)
    small = rt.resize_to(_pad(pre, pads), size, size)
    msq = _pad(hole.astype(np.uint8), pads).astype(np.float32)
    if side == size:
        return np.ascontiguousarray(small), msq > 0.5
    msmall = cv2.resize(msq, (size, size), interpolation=cv2.INTER_LINEAR) > 1e-3
    return np.ascontiguousarray(small), msmall


def _run_crop(sub: np.ndarray, hole: np.ndarray, fn: CropFn, size: int, feather: float) -> np.ndarray:
    ch, cw = hole.shape
    side = max(ch, cw)
    top, left = (side - ch) // 2, (side - cw) // 2
    pads = (top, side - ch - top, left, side - cw - left)
    small, msmall = _model_input(sub, hole, pads, side, size)
    res = _check_result(fn(small, msmall), size)
    big = rt.resize_to(res, side, side)[top : top + ch, left : left + cw]
    return seam_blend(sub, big, hole, feather)


def _work_units(
    m: np.ndarray, boxes: list[Box], *, max_side: int | None, margin: float, whole_mask_frac: float
) -> list[tuple[str, Box]]:
    """("crop", box) units, or ("cell", cell) units for large crops with thin holes."""
    h, w = m.shape
    units: list[tuple[str, Box]] = []
    dt: np.ndarray | None = None
    is_whole = boxes == [(0, 0, w, h)] and float(m.mean()) >= whole_mask_frac
    for box in boxes:
        x0, y0, x1, y1 = box
        if max_side and not is_whole and max(x1 - x0, y1 - y0) > max_side:
            if dt is None:
                dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
            core = max(16, int(max_side / (1.0 + max(0.0, margin))))
            sub = m[y0:y1, x0:x1]
            if float(dt[y0:y1, x0:x1].max()) <= 0.25 * core and sub.any():
                ys, xs = np.nonzero(sub)
                bb = (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
                units.extend(("cell", c) for c in tile_cells(bb, core))
                continue
        units.append(("crop", box))
    return units


def inpaint_by_crops(
    img: RGB,
    mask: np.ndarray,
    fn: CropFn,
    *,
    size: int = 512,
    margin: float = 1.0,
    feather: float = 4,
    min_crop: int = 128,
    whole_frac: float = 0.6,
    whole_mask_frac: float = 0.5,
    max_downscale: float | None = 2.0,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> RGB:
    """Inpaint ``mask`` (bool HxW, True = fill) in ``img`` by running ``fn`` on
    ``size x size`` crops around each hole (see module docstring for the algorithm).

    ``fn`` receives the crop (holes pre-filled smoothly from known pixels) and its bool
    mask (True = fill) and returns the filled ``size x size x 3 uint8`` crop.
    ``max_downscale`` (None disables tiling) bounds how much thin holes are downscaled.
    Unmasked pixels of the result are bit-identical to ``img``."""
    rgb = rt.as_rgb(img)
    h, w = rgb.shape[:2]
    m = rt.as_bool_mask(mask, (h, w))
    if int(size) < 8:
        raise InputError(f"Inpainting crop size {size} is too small.")
    size = int(size)
    work = rgb.copy()
    if not m.any():
        return work
    max_side = int(round(max_downscale * size)) if max_downscale else None
    boxes = plan_crops(
        m,
        margin=margin,
        min_crop=min_crop,
        whole_frac=whole_frac,
        whole_mask_frac=whole_mask_frac,
        max_side=max_side,
    )
    units = _work_units(m, boxes, max_side=max_side, margin=margin, whole_mask_frac=whole_mask_frac)
    log.debug("inpaint_by_crops: %d crop(s), %d unit(s) for %dx%d image", len(boxes), len(units), w, h)
    remaining = m.copy()
    for i, (kind, (x0, y0, x1, y1)) in enumerate(units):
        if cancel is not None:
            cancel.check()
        if not remaining[y0:y1, x0:x1].any():
            continue
        rt.report(progress, i / len(units), f"Inpainting region {i + 1}/{len(units)}")
        if kind == "crop":
            cx0, cy0, cx1, cy1 = x0, y0, x1, y1
        else:
            cx0, cy0, cx1, cy1 = _square((x0, y0, x1, y1), w, h, margin, min_crop)
        hole = remaining[cy0:cy1, cx0:cx1].copy()
        filled = _run_crop(work[cy0:cy1, cx0:cx1], hole, fn, size, feather)
        paste = hole
        if kind == "cell":
            paste = np.zeros_like(hole)
            paste[y0 - cy0 : y1 - cy0, x0 - cx0 : x1 - cx0] = hole[y0 - cy0 : y1 - cy0, x0 - cx0 : x1 - cx0]
        region = work[cy0:cy1, cx0:cx1]
        region[paste] = filled[paste]
        remaining[cy0:cy1, cx0:cx1][paste] = False
    rt.report(progress, 1.0, "Inpainting done")
    return work
