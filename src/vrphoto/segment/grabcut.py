"""Model-free subject segmentation with OpenCV GrabCut.

Used when no AI segmentation model is available.  Without a user rectangle the subject
region is guessed from a *boundary-connectivity saliency* map: the geodesic colour
distance of every pixel from the top/left/right image borders (smooth gradients cost
nothing, colour edges do), weighted by a mild centre prior.  The bottom border is not a
background seed because subjects (people, objects on a table) usually touch it.

The salient region seeds GrabCut (probable foreground), the rectangle limits it, and the
final binary mask is upsampled and snapped to image edges with a guided filter.
"""

from __future__ import annotations

import heapq

import cv2
import numpy as np

from ..core.log import get_logger
from ..core.types import CancelToken, FloatMap, ProgressReporter
from . import _runtime as rt
from ._onnxseg import Rect, expand_rect, validate_rect
from .matte import fill_mask_holes, guided_filter_gray

log = get_logger(__name__)

__all__ = ["GrabCutSegmenter", "saliency_map", "rect_from_saliency"]


def _small(rgb: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    sw, sh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    return rt.resize_to(rgb, sw, sh)


def saliency_map(
    rgb: np.ndarray, *, grid: int = 64, tau: float = 6.0, centre_sigma: float = 0.45
) -> FloatMap:
    """Boundary-connectivity saliency in [0, 1] at the image resolution.

    Geodesic distance (Dijkstra on a ``grid``-pixel-wide Lab image, 4-neighbours) from the
    top, left and right borders with edge cost ``max(0, |dLab| - tau)``, multiplied by a
    Gaussian centre prior (``centre_sigma`` of the image size) and max-normalised."""
    rgb = rt.as_rgb(rgb)
    h, w = rgb.shape[:2]
    small = cv2.GaussianBlur(_small(rgb, grid), (0, 0), 0.7)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab).astype(np.float32)
    sh, sw = lab.shape[:2]
    cost_h = np.maximum(np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=2) - tau, 0.0)  # (sh, sw-1)
    cost_v = np.maximum(np.linalg.norm(lab[1:] - lab[:-1], axis=2) - tau, 0.0)  # (sh-1, sw)
    ch = cost_h.tolist()
    cv = cost_v.tolist()

    dist = [[float("inf")] * sw for _ in range(sh)]
    heap: list[tuple[float, int, int]] = []
    seeds = [(0, x) for x in range(sw)] + [(y, 0) for y in range(sh)] + [(y, sw - 1) for y in range(sh)]
    for y, x in seeds:
        if dist[y][x] != 0.0:
            dist[y][x] = 0.0
            heap.append((0.0, y, x))
    heapq.heapify(heap)
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y][x]:
            continue
        if x + 1 < sw:
            nd = d + ch[y][x]
            if nd < dist[y][x + 1]:
                dist[y][x + 1] = nd
                heapq.heappush(heap, (nd, y, x + 1))
        if x > 0:
            nd = d + ch[y][x - 1]
            if nd < dist[y][x - 1]:
                dist[y][x - 1] = nd
                heapq.heappush(heap, (nd, y, x - 1))
        if y + 1 < sh:
            nd = d + cv[y][x]
            if nd < dist[y + 1][x]:
                dist[y + 1][x] = nd
                heapq.heappush(heap, (nd, y + 1, x))
        if y > 0:
            nd = d + cv[y - 1][x]
            if nd < dist[y - 1][x]:
                dist[y - 1][x] = nd
                heapq.heappush(heap, (nd, y - 1, x))
    sal = np.asarray(dist, np.float32)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    prior = np.exp(
        -0.5 * (((xx + 0.5) / sw - 0.5) ** 2 + ((yy + 0.5) / sh - 0.5) ** 2) / (centre_sigma**2)
    ).astype(np.float32)
    sal *= prior
    mx = float(sal.max())
    if mx > 1e-6:
        sal /= mx
    return cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)


def rect_from_saliency(sal: FloatMap, *, thresh: float = 0.3, margin: float = 0.12) -> Rect:
    """Bounding rectangle ``(x, y, w, h)`` of the salient region, grown by ``margin``.

    Falls back to a centred rectangle (8% margins, touching the bottom) when the map
    has no clear salient region."""
    h, w = sal.shape
    fallback = (int(0.08 * w), int(0.08 * h), max(2, int(0.84 * w)), max(2, h - int(0.08 * h)))
    if float(sal.max()) <= 1e-6:
        return fallback
    m = (sal >= thresh * float(sal.max())).astype(np.uint8)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return fallback
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.nonzero(areas >= 0.1 * areas.max())[0] + 1
    x0 = int(stats[keep, cv2.CC_STAT_LEFT].min())
    y0 = int(stats[keep, cv2.CC_STAT_TOP].min())
    x1 = int((stats[keep, cv2.CC_STAT_LEFT] + stats[keep, cv2.CC_STAT_WIDTH]).max())
    y1 = int((stats[keep, cv2.CC_STAT_TOP] + stats[keep, cv2.CC_STAT_HEIGHT]).max())
    if (x1 - x0) * (y1 - y0) < 0.002 * w * h:
        return fallback
    x, y, rw, rh = expand_rect((x0, y0, x1 - x0, y1 - y0), margin, w, h)
    if h - (y + rh) <= 0.05 * h:  # nearly touching the bottom: let it
        rh = h - y
    return x, y, rw, rh


class GrabCutSegmenter:
    """Segment the main subject with GrabCut (no model download needed)."""

    id = "grabcut"

    def __init__(self, *, max_side: int = 512, iterations: int = 5, refine: bool = True) -> None:
        self.max_side = int(max_side)
        self.iterations = max(1, int(iterations))
        self.refine = refine

    def segment(
        self,
        img: np.ndarray,
        *,
        rect: Rect | None = None,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> FloatMap:
        """Foreground alpha ``float32 HxW`` in [0, 1].

        ``rect`` = ``(x, y, width, height)``: everything outside it is background.  Without
        it the rectangle is derived from a saliency map (or the image centre)."""
        rgb = rt.as_rgb(img)
        h, w = rgb.shape[:2]
        user_rect = validate_rect(rect, w, h)
        rt.report(progress, 0.0, "Finding subject")
        small = _small(rgb, self.max_side) if max(h, w) > self.max_side else rgb
        sh, sw = small.shape[:2]
        sx, sy = sw / w, sh / h
        sal = saliency_map(small)
        if user_rect is not None:
            x, y, rw, rh = user_rect
            r = validate_rect((x * sx, y * sy, max(2.0, rw * sx), max(2.0, rh * sy)), sw, sh)
            assert r is not None
        else:
            r = rect_from_saliency(sal)
        if cancel is not None:
            cancel.check()

        fg_small = self._grabcut(small, sal, r, user_rect is not None, progress, cancel)
        fg_small = self._tidy(fg_small)
        alpha = cv2.resize(fg_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        if self.refine and fg_small.any():
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            rad = max(2, int(np.ceil(1.0 / min(sx, sy))) + 1)
            alpha = guided_filter_gray(gray, alpha, rad, 1e-3)
        if user_rect is not None:
            x, y, rw, rh = user_rect
            outside = np.ones((h, w), bool)
            outside[y : y + rh, x : x + rw] = False
            alpha[outside] = 0.0
        rt.report(progress, 1.0, "Subject found")
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ internals
    def _grabcut(
        self,
        small: np.ndarray,
        sal: FloatMap,
        rect: Rect,
        definite_outside: bool,
        progress: ProgressReporter | None,
        cancel: CancelToken | None,
    ) -> np.ndarray:
        sh, sw = small.shape[:2]
        x, y, rw, rh = rect
        inside = np.zeros((sh, sw), bool)
        inside[y : y + rh, x : x + rw] = True
        mask = np.full((sh, sw), cv2.GC_BGD if definite_outside else cv2.GC_PR_BGD, np.uint8)
        mask[inside] = cv2.GC_PR_BGD
        seeds = inside & (sal >= 0.3 * max(float(sal[inside].max()), 1e-6))
        if seeds.sum() < 0.02 * inside.sum():  # weak saliency: classic rect init
            seeds = inside
        mask[seeds] = cv2.GC_PR_FGD
        if not definite_outside:
            # a thin ring along the top/left/right borders outside the rectangle is background
            ring = np.zeros_like(inside)
            b = max(1, int(round(0.02 * min(sh, sw))))
            ring[:b, :] = True
            ring[:, :b] = True
            ring[:, -b:] = True
            mask[ring & ~inside] = cv2.GC_BGD
        if not (mask == cv2.GC_BGD).any() and not (mask == cv2.GC_PR_BGD).any():
            mask[0, :] = cv2.GC_BGD  # GrabCut needs some background samples
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            for i in range(self.iterations):
                if cancel is not None:
                    cancel.check()
                mode = cv2.GC_INIT_WITH_MASK if i == 0 else cv2.GC_EVAL
                cv2.grabCut(bgr, mask, None, bgd, fgd, 1, mode)
                rt.report(progress, 0.1 + 0.8 * (i + 1) / self.iterations, "Separating subject")
        except cv2.error as e:
            # degenerate input (e.g. a few pixels, no background samples): use the seeds
            log.warning("GrabCut failed (%s); using the saliency estimate", str(e).strip().splitlines()[-1])
            return seeds
        return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

    @staticmethod
    def _tidy(fg: np.ndarray) -> np.ndarray:
        """Drop speckles (< 5% of the largest component) and fill small holes."""
        if not fg.any():
            return fg
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        if n > 2:
            areas = stats[:, cv2.CC_STAT_AREA].copy()
            areas[0] = 0
            keep = areas >= 0.05 * areas.max()
            keep[0] = False
            fg = keep[labels]
        return fill_mask_holes(fg, 0.02)
