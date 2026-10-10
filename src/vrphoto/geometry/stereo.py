"""Stereo pairs, frame-packing layouts and wiggle animations (algorithm B).

* Baseline ``B = ipd * strength``; parallax ``x_L - x_R = f*B/Z``.
* Flat (3D TV / media player) mode converges by shifting the principal points,
  ``cx_L = cx - f*B/(2*Zc)``, ``cx_R = cx + f*B/(2*Zc)``, so a point at the convergence
  distance ``Zc`` lands on the same pixel in both views and
  ``u_L - u_R = f*B*(1/Z - 1/Zc)`` (> 0 = crossed = in front of the screen).
  ``Zc`` defaults to the 35th-percentile depth; if the largest |parallax| exceeds
  ``max_parallax_pct`` % of the width, ``B`` is scaled down.
* VR mode: parallel eyes (``Zc = inf``) with ``B = ipd``.
* Eyes are rendered as ``render_view(li, Pose.translate(-/+ B/2, 0, 0), K_eye)``
  (left eye at ``x = -B/2``).
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

from ..core.errors import InputError
from ..core.log import get_logger
from ..core.types import (
    RGB,
    CancelToken,
    FloatMap,
    Intrinsics,
    LayeredImage,
    Pose,
    ProgressReporter,
    StereoParams,
)
from .warp import render_filled

log = get_logger(__name__)

LAYOUTS: tuple[str, ...] = (
    "sbs",
    "half_sbs",
    "tb",
    "half_tb",
    "anaglyph",
    "anaglyph_gray",
    "cross",
    "left",
    "right",
)

# Dubois least-squares red-cyan anaglyph matrices (applied to linear RGB).
DUBOIS_LEFT = np.array(
    [[0.456, 0.500, 0.176], [-0.040, -0.038, -0.016], [-0.015, -0.021, -0.005]], dtype=np.float32
)
DUBOIS_RIGHT = np.array(
    [[-0.043, -0.088, -0.002], [0.378, 0.734, -0.018], [-0.072, -0.113, 1.226]], dtype=np.float32
)
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
_STRIP_ROWS_PX = 1_048_576


def _srgb_to_linear_lut() -> np.ndarray:
    c = np.arange(256, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin.astype(np.float32)


def _linear_to_srgb_lut(n: int = 4096) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


_TO_LIN = _srgb_to_linear_lut()
_TO_SRGB = _linear_to_srgb_lut()


# --------------------------------------------------------------------------- convergence
def _finite_positive(inv_z: np.ndarray, max_samples: int = 1_000_000) -> np.ndarray:
    a = np.asarray(inv_z, dtype=np.float32)
    if a.size > max_samples and a.ndim == 2:
        st = int(math.ceil(math.sqrt(a.size / max_samples)))
        a = a[::st, ::st]
    a = a.reshape(-1)
    return a[np.isfinite(a) & (a > 0)]


def auto_convergence(inv_z: FloatMap, pct: float = 35.0) -> float:
    """Convergence distance (metres): the ``pct``-th percentile of scene depth.

    35 % of the (sampled) pixels are nearer than the returned distance.
    """
    if not 0.0 <= pct <= 100.0:
        raise InputError(f"Percentile must be between 0 and 100 (got {pct}).")
    v = _finite_positive(inv_z)
    if v.size == 0:
        raise InputError("The depth map has no valid values; cannot choose a convergence distance.")
    # depth percentile pct  <=>  inverse-depth percentile (100 - pct)
    w = float(np.percentile(v, 100.0 - pct))
    return 1.0 / max(w, 1e-9)


def visible_inv_z(li: LayeredImage) -> FloatMap:
    """Inverse depth of what the source camera sees (front-most layer with alpha >= 0.5)."""
    if not li.layers:
        raise InputError("The layered image has no layers.")
    out = np.array(li.layers[0].inv_z, dtype=np.float32, copy=True)
    for lay in li.layers[1:]:
        m = np.asarray(lay.alpha) >= 0.5
        out[m] = np.asarray(lay.inv_z, dtype=np.float32)[m]
    return out


def resolve_stereo(
    li: LayeredImage, params: StereoParams, dst_K: Intrinsics | None = None
) -> tuple[float, float]:
    """Effective ``(baseline_m, convergence_m)`` for ``params`` (convergence inf in VR mode).

    Flat mode applies auto convergence and the parallax clamp (measured in pixels of
    ``dst_K``, default ``li.K``).  ``max_parallax_pct <= 0`` disables the clamp.
    """
    if params.ipd_m <= 0 or not math.isfinite(params.ipd_m):
        raise InputError(f"Eye distance (IPD) must be positive (got {params.ipd_m} m).")
    if params.mode == "vr":
        return float(params.ipd_m), math.inf
    if params.mode != "flat":
        raise InputError(f"Unknown stereo mode {params.mode!r}; use 'flat' or 'vr'.")
    if params.strength < 0 or not math.isfinite(params.strength):
        raise InputError(f"3D strength must be >= 0 (got {params.strength}).")
    B = float(params.ipd_m) * float(params.strength)
    w = visible_inv_z(li)
    if params.convergence_m is None:
        zc = auto_convergence(w)
    else:
        zc = float(params.convergence_m)
        if not zc > 0:
            raise InputError(f"Convergence distance must be positive (got {params.convergence_m} m).")
    K = dst_K if dst_K is not None else li.K
    if B > 0 and params.max_parallax_pct and params.max_parallax_pct > 0:
        v = _finite_positive(w)
        if v.size:
            inv_c = 0.0 if math.isinf(zc) else 1.0 / zc
            span = max(abs(float(v.max()) - inv_c), abs(float(v.min()) - inv_c))
            pmax = K.fx * B * span
            limit = params.max_parallax_pct / 100.0 * K.width
            if pmax > limit > 0:
                log.debug("Parallax %.1f px exceeds %.1f px; scaling baseline down", pmax, limit)
                B *= limit / pmax
    return B, zc


def _eye_K(K: Intrinsics, x_m: float, zc: float) -> Intrinsics:
    """Principal-point shifted intrinsics for an eye at ``x = x_m`` converging at ``zc``."""
    if math.isinf(zc) or x_m == 0.0:
        return K
    return K.with_principal(K.cx + K.fx * x_m / zc, K.cy)


def _offset_pose(center: Pose | None, x_m: float) -> Pose:
    if center is None:
        return Pose.translate(x_m, 0.0, 0.0)
    R = np.asarray(center.R, dtype=np.float64)
    t = np.asarray(center.t, dtype=np.float64) + R @ np.array([x_m, 0.0, 0.0])
    return Pose(R.copy(), t)


def eye_views(
    li: LayeredImage,
    params: StereoParams,
    dst_K: Intrinsics | None = None,
    *,
    center: Pose | None = None,
) -> tuple[tuple[Pose, Intrinsics], tuple[Pose, Intrinsics]]:
    """``((pose_L, K_L), (pose_R, K_R))`` for rendering a stereo pair.

    ``center`` (default identity) is the cyclopean camera pose, e.g. a frame of a camera
    path; the eyes are offset by ``-/+ B/2`` along its x axis.  ``swap_eyes`` is NOT
    applied here (see :func:`stereo_pair`).
    """
    K = dst_K if dst_K is not None else li.K
    B, zc = resolve_stereo(li, params, K)
    h = B / 2.0
    return (_offset_pose(center, -h), _eye_K(K, -h, zc)), (_offset_pose(center, h), _eye_K(K, h, zc))


def stereo_pair(
    li: LayeredImage,
    params: StereoParams,
    dst_K: Intrinsics | None = None,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> tuple[RGB, RGB]:
    """Render a hole-filled ``(left, right)`` stereo pair (swapped if ``params.swap_eyes``)."""
    (pl, kl), (pr, kr) = eye_views(li, params, dst_K)
    if progress is not None:
        progress(0.0, "Rendering left eye")
    left = render_filled(li, pl, kl, cancel=cancel)
    if progress is not None:
        progress(0.5, "Rendering right eye")
    right = render_filled(li, pr, kr, cancel=cancel)
    if progress is not None:
        progress(1.0, "Stereo pair ready")
    if params.swap_eyes:
        return right, left
    return left, right


# --------------------------------------------------------------------------- layouts
def _anaglyph_dubois(left: RGB, right: RGB) -> RGB:
    h, w = left.shape[:2]
    out = np.empty_like(left)
    rows = max(1, _STRIP_ROWS_PX // max(1, w))
    ml = DUBOIS_LEFT.T
    mr = DUBOIS_RIGHT.T
    n = _TO_SRGB.size - 1
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        lin = _TO_LIN[left[r0:r1]] @ ml
        lin += _TO_LIN[right[r0:r1]] @ mr
        np.clip(lin, 0.0, 1.0, out=lin)
        lin *= np.float32(n)
        lin += np.float32(0.5)
        out[r0:r1] = _TO_SRGB[lin.astype(np.int32)]
    return out


def _gray(img: RGB) -> np.ndarray:
    g = img.astype(np.float32) @ _LUMA
    return np.clip(np.rint(g), 0, 255).astype(np.uint8)


def compose(left: RGB, right: RGB, layout: str) -> RGB:
    """Pack a stereo pair into one frame.

    ``"sbs"`` (H x 2W), ``"half_sbs"`` (each eye squeezed to W//2 -> H x 2*(W//2)),
    ``"tb"`` (2H x W, left on top), ``"half_tb"`` (2*(H//2) x W), ``"anaglyph"`` (Dubois
    red-cyan, linear light), ``"anaglyph_gray"`` (red = left luma, green/blue = right
    luma), ``"cross"`` (right | left, for cross-eyed free viewing), ``"left"``, ``"right"``.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim != 3 or left.shape[2] != 3 or left.dtype != np.uint8:
        raise InputError("Stereo images must be HxWx3 uint8 RGB.")
    if right.shape != left.shape or right.dtype != left.dtype:
        raise InputError(
            f"Left and right images differ in size: {left.shape[1]}x{left.shape[0]} vs "
            f"{right.shape[1]}x{right.shape[0]}."
        )
    h, w = left.shape[:2]
    layout = _norm_layout(layout)
    if layout == "sbs":
        return np.ascontiguousarray(np.concatenate([left, right], axis=1))
    if layout == "cross":
        return np.ascontiguousarray(np.concatenate([right, left], axis=1))
    if layout == "tb":
        return np.ascontiguousarray(np.concatenate([left, right], axis=0))
    if layout == "half_sbs":
        hw = max(1, w // 2)
        sq = [cv2.resize(e, (hw, h), interpolation=cv2.INTER_AREA) for e in (left, right)]
        return np.ascontiguousarray(np.concatenate(sq, axis=1))
    if layout == "half_tb":
        hh = max(1, h // 2)
        sq = [cv2.resize(e, (w, hh), interpolation=cv2.INTER_AREA) for e in (left, right)]
        return np.ascontiguousarray(np.concatenate(sq, axis=0))
    if layout == "anaglyph":
        return _anaglyph_dubois(left, right)
    if layout == "anaglyph_gray":
        gl, gr = _gray(left), _gray(right)
        return np.ascontiguousarray(np.stack([gl, gr, gr], axis=-1))
    if layout == "left":
        return left.copy()
    if layout == "right":
        return right.copy()
    raise InputError(f"Unknown stereo layout {layout!r}. Valid layouts: {', '.join(LAYOUTS)}.")


def _norm_layout(layout: str) -> str:
    """Case/whitespace-insensitive layout name (``"SBS"`` -> ``"sbs"``)."""
    return str(layout).strip().lower()


def layout_size(width: int, height: int, layout: str) -> tuple[int, int]:
    """Output ``(width, height)`` of :func:`compose` for eyes of ``width x height``."""
    layout = _norm_layout(layout)
    table: dict[str, tuple[int, int]] = {
        "sbs": (2 * width, height),
        "cross": (2 * width, height),
        "tb": (width, 2 * height),
        "half_sbs": (2 * max(1, width // 2), height),
        "half_tb": (width, 2 * max(1, height // 2)),
    }
    if layout in table:
        return table[layout]
    if layout in LAYOUTS:
        return width, height
    raise InputError(f"Unknown stereo layout {layout!r}. Valid layouts: {', '.join(LAYOUTS)}.")


# --------------------------------------------------------------------------- wiggle
def wiggle_frames(
    li: LayeredImage,
    params: StereoParams,
    n: int = 8,
    *,
    dst_K: Intrinsics | None = None,
    cancel: CancelToken | None = None,
    progress: ProgressReporter | None = None,
) -> list[RGB]:
    """Frames for a looping "wiggle" animation between the left and right eye.

    The camera follows ``x_i = -(B/2) cos(2 pi i / n)`` (starts at the left eye, reaches
    the right eye at ``i = n/2``) with the flat-mode convergence pivot, so the
    convergence plane stays still while nearer / farther content rocks.  Always uses
    flat-mode baseline/convergence (a wiggle is viewed on a flat screen).
    """
    if int(n) < 2:
        raise InputError("A wiggle animation needs at least 2 frames.")
    n = int(n)
    K = dst_K if dst_K is not None else li.K
    flat = dataclasses.replace(params, mode="flat")
    B, zc = resolve_stereo(li, flat, K)
    frames: list[RGB] = []
    for i in range(n):
        if cancel is not None:
            cancel.check()
        x = -(B / 2.0) * math.cos(2.0 * math.pi * i / n)
        frames.append(render_filled(li, Pose.translate(x, 0.0, 0.0), _eye_K(K, x, zc), cancel=cancel))
        if progress is not None:
            progress((i + 1) / n, f"Wiggle frame {i + 1}/{n}")
    return frames


__all__ = [
    "auto_convergence",
    "stereo_pair",
    "compose",
    "wiggle_frames",
    "resolve_stereo",
    "eye_views",
    "visible_inv_z",
    "layout_size",
    "LAYOUTS",
    "DUBOIS_LEFT",
    "DUBOIS_RIGHT",
]
