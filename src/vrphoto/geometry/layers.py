"""Layered depth images: split a photo + inverse depth into back-to-front layers.

Algorithm C (docs/CONTRACTS.md).  The background layer is what makes "look behind"
work: behind every depth edge it carries a band of inpainted background (pushed to the
background depth), so when the virtual camera moves, the areas revealed next to
foreground edges show plausible background instead of stretched pixels.

* ``S = f*Bmax*inv_z`` (pixels of parallax at the maximum baseline).  Depth edges are
  pixels whose 4-neighbour ``|dS| > 1 px``.
* Band width ``k = ceil(1.3*f*Bmax*(p99(inv_z) - p1(inv_z))) + 4`` px, at most 15% of W.
* Background depth ``d_bg``: erosion (local min of inverse depth = farthest) with an
  elliptical structuring element of radius ``k``, then a box blur of size ``k/2``.
* Hallucination mask ``H = (inv_z - d_bg > 1/(f*Bmax)) & dilate(E, k)`` then a 3 px
  morphological open (corner-preserving) + close.
* Background layer: colour inpainted in ``H``; inverse depth ``min(d_bg, inv_z)`` in ``H``.
  Without an inpainter, ``H`` is push-pull filled from the background side only
  (foreground pixels next to the band are excluded from the context) + a light blur.
* Clean plate: ``H = dilate(alpha > 0.5, 8 px)``; the subject becomes its own front
  layer with the matte as alpha.  If the rest of the scene has depth edges too, a band
  background is added behind the plate: layers ``[band background, clean plate,
  subject]`` (the plate keeps the original pixels outside the subject so the source
  view is reproduced exactly).

Large structuring elements are evaluated on a min/max-pooled grid (exact for small
radii), which keeps 24 MP inputs fast while erring on the side of a wider band.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Literal

import cv2
import numpy as np

from ..core.errors import InferenceError, InputError, VRPhotoError
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, FloatMap, Intrinsics, Layer, LayeredImage, ProgressReporter
from .warp import _block_reduce, _upsample_blocks, push_pull_fill_fast, sanitize_inv_z

log = get_logger(__name__)

LayerMode = Literal["band", "cleanplate", "none"]
MODES: tuple[str, ...] = ("band", "cleanplate", "none")

_FULLRES_MORPH_MAX = 24  # larger radii are computed on a pooled grid
_CLEANPLATE_DILATE_PX = 8
_FALLBACK_BLUR_SIGMA = 1.5


# --------------------------------------------------------------------------- helpers
def _check(cancel: CancelToken | None) -> None:
    if cancel is not None:
        cancel.check()


def _as_reporter(progress: Any) -> ProgressReporter:
    if progress is None:
        return ProgressReporter.null()
    if isinstance(progress, ProgressReporter):
        return progress
    if callable(progress):
        return ProgressReporter(progress)
    raise InputError("progress must be a ProgressReporter or a callable(frac, msg).")


def _ellipse(radius: int) -> np.ndarray:
    d = 2 * int(radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _morph_disk(a: np.ndarray, radius: int, op: str) -> np.ndarray:
    """Erode (``op="min"``) or dilate (``op="max"``) with a disk of ``radius`` px.

    Radii above ``_FULLRES_MORPH_MAX`` run on a pooled grid (pooling with the same op,
    so the result is never narrower than the exact one) and are upsampled.
    """
    if radius <= 0:
        return a.copy()
    fn = cv2.erode if op == "min" else cv2.dilate
    if radius <= _FULLRES_MORPH_MAX:
        return fn(a, _ellipse(radius), borderType=cv2.BORDER_REPLICATE)
    f = int(math.ceil(radius / _FULLRES_MORPH_MAX))
    small = _block_reduce(a, f, op)
    small = fn(small, _ellipse(int(math.ceil(radius / f))), borderType=cv2.BORDER_REPLICATE)
    return _upsample_blocks(small, f, a.shape[:2])


def _smooth_bg_depth(w: np.ndarray, k: int) -> np.ndarray:
    """``d_bg = box_blur(erode(w, disk k), k/2)``, clamped to ``<= w``."""
    kb = max(1, k // 2) | 1
    h, wd = w.shape
    if k <= _FULLRES_MORPH_MAX:
        er = cv2.erode(w, _ellipse(k), borderType=cv2.BORDER_REPLICATE)
        d = cv2.blur(er, (kb, kb), borderType=cv2.BORDER_REPLICATE)
    else:
        f = int(math.ceil(k / _FULLRES_MORPH_MAX))
        small = _block_reduce(w, f, "min")
        small = cv2.erode(small, _ellipse(int(math.ceil(k / f))), borderType=cv2.BORDER_REPLICATE)
        kbs = max(1, int(round(kb / f))) | 1
        small = cv2.blur(small, (kbs, kbs), borderType=cv2.BORDER_REPLICATE)
        sh, sw = small.shape
        # block centres of the pooled grid sit at (i + 0.5) * f - 0.5 -> resize aligns them
        d = cv2.resize(small, (sw * f, sh * f), interpolation=cv2.INTER_LINEAR)[:h, :wd]
    return np.minimum(d, w).astype(np.float32, copy=False)


def depth_edges(w: np.ndarray, fb: float) -> np.ndarray:
    """Pixels where the parallax ``fb * w`` changes by more than 1 px to a 4-neighbour."""
    thr = np.float32(1.0 / fb)
    e = np.zeros(w.shape, bool)
    dh = np.abs(np.diff(w, axis=1)) > thr
    e[:, 1:] |= dh
    e[:, :-1] |= dh
    dv = np.abs(np.diff(w, axis=0)) > thr
    e[1:, :] |= dv
    e[:-1, :] |= dv
    return e


def band_mask(
    w: np.ndarray, K: Intrinsics, max_baseline_m: float, cancel: CancelToken | None = None
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Hallucination mask ``H`` and background depth ``d_bg`` of algorithm C.

    Returns ``(H, d_bg, k)``; ``H`` and ``d_bg`` are None when the image has no depth
    edges at this baseline (nothing can be revealed).
    """
    fb = float(K.fx) * float(max_baseline_m)
    edges = depth_edges(w, fb)
    if not edges.any():
        return None, None, 0
    _check(cancel)
    h, wd = w.shape
    st = max(1, int(math.sqrt(h * wd / 1_000_000.0)))
    p1, p99 = (float(v) for v in np.percentile(w[::st, ::st], [1.0, 99.0]))
    k = int(math.ceil(1.3 * fb * max(0.0, p99 - p1))) + 4
    k = max(1, min(k, int(0.15 * wd)))
    d_bg = _smooth_bg_depth(w, k)
    _check(cancel)
    band = _morph_disk(edges.astype(np.uint8), k, "max") > 0
    H = ((w - d_bg) > np.float32(1.0 / fb)) & band
    # 3 px open (drop speckles) + close (fill pinholes).  Pixels of H touching the opened
    # mask are restored so shape corners survive the opening.
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    H8 = H.astype(np.uint8)
    opened = cv2.morphologyEx(H8, cv2.MORPH_OPEN, ker)
    opened |= H8 & cv2.dilate(opened, ker)
    H = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, ker) > 0
    if not H.any():
        return None, None, k
    return H, d_bg, k


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise InputError(f"Expected an RGB image (HxWx3), got shape {img.shape}.")
    img = img[..., :3]
    if img.dtype == np.uint16:  # 16-bit photo: scale, do not saturate
        img = ((img.astype(np.uint32) + 128) // 257).astype(np.uint8)
    elif img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        f = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if np.issubdtype(img.dtype, np.floating) and f.size and float(f.max()) <= 1.0 + 1e-6:
            f *= 255.0  # float image in [0, 1]
        img = np.clip(np.rint(f), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _call_inpainter(
    inpainter: Any, img: RGB, mask: np.ndarray, progress: ProgressReporter, cancel: CancelToken | None
) -> RGB:
    """Run ``inpainter.inpaint(img, mask[, progress=, cancel=])`` and validate its output.

    Only pixels inside ``mask`` are taken from the result.
    """
    fn = getattr(inpainter, "inpaint", None)
    if not callable(fn):
        raise InputError("The inpainter object has no inpaint(img, mask) method.")
    kwargs: dict[str, Any] = {}
    try:
        params = inspect.signature(fn).parameters
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        if var_kw or "progress" in params:
            kwargs["progress"] = progress
        if var_kw or "cancel" in params:
            kwargs["cancel"] = cancel
    except (TypeError, ValueError):  # builtins / C extensions without a signature
        pass
    name = getattr(inpainter, "id", type(inpainter).__name__)
    try:
        out = fn(img, mask, **kwargs)
    except VRPhotoError:
        raise
    except Exception as e:  # model / third-party failure -> human-readable error
        raise InferenceError(f"Inpainting the background with '{name}' failed: {e}") from e
    out = np.asarray(out) if out is not None else None
    if out is None or out.ndim not in (2, 3) or out.shape[:2] != img.shape[:2]:
        got = "nothing" if out is None else f"an array of shape {out.shape}"
        raise InferenceError(
            f"Inpainter '{name}' returned {got}; expected an image of {img.shape[1]}x{img.shape[0]} px."
        )
    out = _to_rgb_u8(out)
    res = img.copy()
    res[mask] = out[mask]
    return res


def _foreground_context(w: np.ndarray, band: np.ndarray, d_bg: np.ndarray, k: int, fb: float) -> np.ndarray:
    """Pixels next to a look-behind band that are nearer than the band's background.

    The band ``H`` sits on the near side of a depth edge; its outer side touches the
    true background, its inner side the foreground interior.  A context-blind fill
    (push-pull) would smear that foreground colour into the band, so revealed areas
    would look like stretched foreground again.  This marks the foreground pixels within
    ``2k`` px of the band whose inverse depth exceeds the band's background depth
    (propagated from the band by a grey dilation) by more than 1 px of parallax, so the
    band is filled from the background side.
    """
    ref = np.where(band, d_bg, np.float32(0.0)).astype(np.float32)
    ref = _morph_disk(ref, 2 * k, "max")  # nearest band background within 2k px
    fg = w > ref + np.float32(1.0 / fb)
    fg &= ref > 0
    fg &= ~band
    return fg


def _fallback_fill(img: RGB, mask: np.ndarray, exclude: np.ndarray | None = None) -> RGB:
    """No-model background fill: push-pull from the unmasked pixels plus a light blur.

    ``exclude`` marks pixels that must not be used as context (foreground next to a
    look-behind band); it is ignored if it would leave no context at all.
    """
    known = ~mask
    if exclude is not None:
        k2 = known & ~exclude
        if k2.any():
            known = k2
    filled = push_pull_fill_fast(img, known)
    blurred = cv2.GaussianBlur(filled, (0, 0), _FALLBACK_BLUR_SIGMA)
    res = img.copy()
    res[mask] = blurred[mask]
    return res


def _inpaint(
    inpainter: Any,
    img: RGB,
    mask: np.ndarray,
    progress: ProgressReporter,
    cancel: CancelToken | None,
    exclude: np.ndarray | None = None,
) -> RGB:
    """Fill ``mask`` with ``inpainter`` (or the push-pull fallback when it is None).

    ``exclude`` (foreground context) is only used by the fallback; model inpainters
    receive exactly ``mask`` (Inpainter protocol).
    """
    if inpainter is None:
        return _fallback_fill(img, mask, exclude)
    return _call_inpainter(inpainter, img, mask, progress, cancel)


# --------------------------------------------------------------------------- public API
def build_layers(
    img: RGB,
    inv_z: FloatMap,
    K: Intrinsics,
    *,
    max_baseline_m: float,
    inpainter: Any | None,
    alpha: FloatMap | None = None,
    mode: LayerMode = "band",
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> LayeredImage:
    """Split ``img`` + ``inv_z`` into a back-to-front :class:`LayeredImage`.

    * ``"none"``: a single layer (the photo with its depth).
    * ``"band"``: background layer with inpainted bands behind depth edges + the photo
      as an opaque front layer.  If the image has no depth edges at ``max_baseline_m``
      a single layer is returned (nothing can be revealed).
    * ``"cleanplate"``: the subject (``alpha``) is removed from the background entirely;
      the front layer is the photo with the matte as alpha.  Requires ``alpha``.  When
      the remaining scene has depth edges at ``max_baseline_m``, three layers are
      returned: ``[band background, clean plate, subject]`` (the plate is the original
      photo with the subject inpainted, the band background additionally has the
      look-behind bands of the other edges).  ``max_baseline_m = 0`` disables the bands.

    In band mode the hallucinated region ``H`` is exactly where
    ``layers[0].inv_z < layers[1].inv_z`` (mesh export uses this to keep the inpainted
    background triangles).

    ``max_baseline_m`` is the largest camera offset (metres) the result should support.
    ``inpainter`` follows the ``Inpainter`` protocol (``inpaint(img, mask_bool) -> RGB``);
    ``None`` uses push-pull filling with a light blur.  ``alpha`` is ignored except in
    clean-plate mode.
    """
    rep = _as_reporter(progress)
    if mode not in MODES:
        raise InputError(f"Unknown layer mode {mode!r}. Valid modes: {', '.join(MODES)}.")
    img = _to_rgb_u8(img)
    h, wd = img.shape[:2]
    inv_z = np.asarray(inv_z)
    if inv_z.shape != (h, wd):
        got = f"{inv_z.shape[1]}x{inv_z.shape[0]}" if inv_z.ndim == 2 else f"of shape {inv_z.shape}"
        raise InputError(f"Depth map {got} does not match the image size {wd}x{h}.")
    if (K.width, K.height) != (wd, h):
        raise InputError(f"Camera intrinsics are for {K.width}x{K.height} but the image is {wd}x{h}.")
    w = sanitize_inv_z(inv_z)
    if w is inv_z:
        w = w.copy()  # never alias the caller's array inside the returned layers
    ones = np.ones((h, wd), np.float32)
    _check(cancel)

    if mode == "none":
        rep(1.0, "Layers ready")
        return LayeredImage([Layer(img, w, ones)], K)

    bmax = float(max_baseline_m)
    if not math.isfinite(bmax) or bmax < 0 or (mode == "band" and bmax <= 0):
        raise InputError(f"max_baseline_m must be a positive distance in metres (got {max_baseline_m}).")

    if mode == "cleanplate":
        if alpha is None:
            raise InputError("Clean-plate mode needs a subject mask; segment the subject first.")
        a_in = np.asarray(alpha)
        if a_in.shape != (h, wd):
            raise InputError(f"Subject mask of shape {a_in.shape} does not match the image size {wd}x{h}.")
        a = a_in.astype(np.float32)
        if a_in.dtype == np.uint8:  # 8-bit matte (0..255)
            a *= np.float32(1.0 / 255.0)
        a = np.clip(np.nan_to_num(a), 0.0, 1.0)
        subj = a > 0.5
        if not subj.any():
            log.warning("Clean-plate requested but the subject mask is empty; using band mode")
            if bmax <= 0:
                return LayeredImage([Layer(img, w, ones)], K)
            return build_layers(
                img, w, K, max_baseline_m=bmax, inpainter=inpainter, mode="band", progress=rep, cancel=cancel
            )
        rep(0.05, "Removing subject")
        hc = cv2.dilate(subj.astype(np.uint8), _ellipse(_CLEANPLATE_DILATE_PX)) > 0
        if hc.all():  # subject fills the frame: no background to learn depth from
            log.warning("The subject covers the whole image; the clean plate is a guess")
            w_bg = np.full_like(w, float(w.min()))
        else:
            w_bg = np.minimum(push_pull_fill_fast(w, ~hc), w)
        _check(cancel)
        hb, d_bg, kb = None, None, 0
        if bmax > 0:
            rep(0.15, "Finding depth edges")
            hb, d_bg, kb = band_mask(w_bg, K, bmax, cancel)
        H = hc if hb is None else (hc | hb)
        excl = None
        if inpainter is None and hb is not None and d_bg is not None:
            excl = _foreground_context(w_bg, hb & ~hc, d_bg, kb, float(K.fx) * bmax)
        rep(0.3, "Inpainting background")
        rgb_bg = _inpaint(inpainter, img, H, rep.sub(0.3, 0.95), cancel, excl)
        _check(cancel)
        subject = Layer(img, w, a)
        if hb is None or d_bg is None:
            rep(1.0, "Layers ready")
            return LayeredImage([Layer(rgb_bg, w_bg, ones), subject], K)
        # Three layers: the band-inpainted background, the clean plate (original scene
        # with the subject removed) and the subject.  The plate keeps the original
        # pixels in the bands of the remaining depth edges, so the source view is
        # reproduced exactly outside the subject; the bands only show through tears.
        plate_rgb = rgb_bg.copy()
        keep = hb & ~hc
        plate_rgb[keep] = img[keep]
        back = Layer(rgb_bg, np.where(hb, d_bg, w_bg).astype(np.float32), ones)
        plate = Layer(plate_rgb, w_bg.astype(np.float32, copy=False), ones.copy())
        rep(1.0, "Layers ready")
        return LayeredImage([back, plate, subject], K)

    # band
    rep(0.05, "Finding depth edges")
    H, d_bg, k = band_mask(w, K, bmax, cancel)
    if H is None or d_bg is None:
        log.info("No depth edges at baseline %.3f m; using a single layer", bmax)
        rep(1.0, "Layers ready")
        return LayeredImage([Layer(img, w, ones)], K)
    log.debug("Look-behind band: k=%d px, %.2f%% of pixels inpainted", k, 100.0 * H.mean())
    excl = _foreground_context(w, H, d_bg, k, float(K.fx) * bmax) if inpainter is None else None
    rep(0.3, "Inpainting background")
    rgb_bg = _inpaint(inpainter, img, H, rep.sub(0.3, 0.95), cancel, excl)
    _check(cancel)
    inv_bg = np.where(H, d_bg, w).astype(np.float32)
    rep(1.0, "Layers ready")
    return LayeredImage([Layer(rgb_bg, inv_bg, ones.copy()), Layer(img, w, ones)], K)


__all__ = ["build_layers", "band_mask", "depth_edges", "MODES", "LayerMode"]
