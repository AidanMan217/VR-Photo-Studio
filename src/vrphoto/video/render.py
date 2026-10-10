"""Render the frames of a camera-path video from a layered 3D photo.

* :func:`render_path_frames` - flat video: mono, or stereo packed as SBS / TB / anaglyph
  etc. (``geometry.stereo.compose`` layouts).  The output camera is zoomed in by
  ``overscan`` (focal x (1 + overscan), algorithm D: 4% for video) so the frame edges
  never reveal the empty space beyond the photo while the camera moves.
* :func:`render_vr180_frames` - VR180 video: per frame a stereo pair (VR mode: baseline =
  IPD, convergence at infinity) mapped into side-by-side 180 equirect frames with
  ``geometry.projection.vr180_frame``.  The eyes are rendered at the resolution the
  equirect actually samples (:func:`vr180_eye_K`), not at the full photo resolution,
  which makes frames of large photos several times cheaper with visually identical output.

Both are generators (one frame per pose of the path), so frames can be streamed into
:class:`vrphoto.video.encode.VideoWriter` without holding the video in memory.

Stereo eyes follow algorithm B, applied around every pose of the path: the eyes sit at
``-/+ B/2`` along the path camera's own x axis; flat mode converges at ``Zc`` by the
principal-point shift ``cx -/+ f*B/(2*Zc)``.  Baseline and convergence are resolved once
per video (from the layered image), so they do not pump from frame to frame.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Any

import numpy as np

from ..core.errors import InputError
from ..core.log import get_logger
from ..core.types import (
    RGB,
    CancelToken,
    Intrinsics,
    LayeredImage,
    Pose,
    ProgressReporter,
    StereoParams,
)
from .paths import CameraPath

log = get_logger(__name__)

#: Layouts accepted by :func:`render_path_frames` ("mono" + ``geometry.stereo.compose``).
FLAT_LAYOUTS: tuple[str, ...] = (
    "mono", "sbs", "half_sbs", "tb", "half_tb", "anaglyph", "anaglyph_gray", "cross", "left", "right",
)  # fmt: skip
VR180_LAYOUTS: tuple[str, ...] = ("sbs", "tb")
MAX_OVERSCAN = 0.5
#: Eye images for VR180 are rendered with this much more angular pixel density than the
#: equirect output at the image centre.  ``geometry.projection`` area-filters any source
#: denser than 1.25x down to the output density, so more pixels are wasted work; 2x keeps
#: the result within ~40-45 dB PSNR of rendering at the full photo resolution.
VR180_DENSITY_MARGIN = 2.0
#: Mirror of ``geometry.projection``'s cap on the enlarged VR180 field of view.
_VR180_MAX_FOV_DEG = 150.0


# --------------------------------------------------------------------------- geometry access
def _warp() -> Any:
    from ..geometry import warp

    return warp


def _stereo() -> Any:
    from ..geometry import stereo

    return stereo


def _projection() -> Any:
    from ..geometry import projection

    return projection


@lru_cache(maxsize=16)
def _accepts_cancel(fn: Callable[..., Any]) -> bool:
    try:
        return "cancel" in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins / C functions
        return False


def _render(li: LayeredImage, pose: Pose, K: Intrinsics, cancel: CancelToken | None) -> RGB:
    """``geometry.warp.render_filled`` (passing ``cancel`` when it is supported)."""
    fn = _warp().render_filled
    if cancel is not None and _accepts_cancel(fn):
        return fn(li, pose, K, cancel=cancel)
    return fn(li, pose, K)


# --------------------------------------------------------------------------- stereo setup
def _visible_inv_z(li: LayeredImage) -> np.ndarray:
    out = np.array(li.layers[0].inv_z, dtype=np.float32, copy=True)
    for lay in li.layers[1:]:
        m = np.asarray(lay.alpha) >= 0.5
        out[m] = np.asarray(lay.inv_z, dtype=np.float32)[m]
    return out


def _resolve_stereo_local(li: LayeredImage, params: StereoParams, K: Intrinsics) -> tuple[float, float]:
    """Algorithm B baseline / convergence from contract functions only (fallback)."""
    if not (params.ipd_m > 0 and math.isfinite(params.ipd_m)):
        raise InputError(f"Eye distance (IPD) must be positive (got {params.ipd_m} m).")
    if params.mode == "vr":
        return float(params.ipd_m), math.inf
    if params.mode != "flat":
        raise InputError(f"Unknown stereo mode {params.mode!r}; use 'flat' or 'vr'.")
    if not (params.strength >= 0 and math.isfinite(params.strength)):
        raise InputError(f"3D strength must be >= 0 (got {params.strength}).")
    B = float(params.ipd_m) * float(params.strength)
    w = _visible_inv_z(li)
    if params.convergence_m is None:
        zc = float(_stereo().auto_convergence(w))
    else:
        zc = float(params.convergence_m)
        if not zc > 0:
            raise InputError(f"Convergence distance must be positive (got {params.convergence_m} m).")
    v = w[np.isfinite(w) & (w > 0)]
    if B > 0 and params.max_parallax_pct and params.max_parallax_pct > 0 and v.size:
        inv_c = 0.0 if math.isinf(zc) else 1.0 / zc
        crossed = K.fx * B * max(0.0, float(v.max()) - inv_c)
        uncrossed = K.fx * B * max(0.0, inv_c - float(v.min()))
        limit = params.max_parallax_pct / 100.0 * K.width
        # uncrossed parallax never wider than the eye separation on the virtual screen
        frac = params.max_parallax_pct / 100.0
        sw = float(params.screen_width_m or 0.0)
        if sw > 0 and math.isfinite(sw):
            frac = min(frac, float(params.ipd_m) / sw)
        far_limit = frac * K.width
        scale = 1.0
        if crossed > limit > 0:
            scale = limit / crossed
        if uncrossed > far_limit > 0:
            scale = min(scale, far_limit / uncrossed)
        B *= scale
    return B, zc


def resolve_stereo(li: LayeredImage, params: StereoParams, K: Intrinsics) -> tuple[float, float]:
    """Effective ``(baseline_m, convergence_m)`` for a video rendered with intrinsics ``K``.

    Uses ``geometry.stereo.resolve_stereo`` when available (identical to still stereo
    photos), else an equivalent implementation of algorithm B.
    """
    fn = getattr(_stereo(), "resolve_stereo", None)
    if callable(fn):
        B, zc = fn(li, params, K)
        return float(B), float(zc)
    return _resolve_stereo_local(li, params, K)


def eye_pose(center: Pose, x_m: float) -> Pose:
    """``center`` moved by ``x_m`` along its own x axis (eye offset)."""
    R = np.asarray(center.R, dtype=np.float64)
    t = np.asarray(center.t, dtype=np.float64) + R[:, 0] * float(x_m)
    return Pose(R.copy(), t)


def eye_K(K: Intrinsics, x_m: float, zc: float) -> Intrinsics:
    """Principal-point shifted intrinsics for an eye at ``x_m`` converging at ``zc`` (B)."""
    if math.isinf(zc) or x_m == 0.0:
        return K
    return K.with_principal(K.cx + K.fx * x_m / zc, K.cy)


def overscan_K(K: Intrinsics, overscan: float) -> Intrinsics:
    """Same image size and principal point, focal length x ``(1 + overscan)``."""
    s = 1.0 + float(overscan)
    return Intrinsics(K.fx * s, K.fy * s, K.cx, K.cy, K.width, K.height)


def vr180_eye_K(K: Intrinsics, eye_px: int, size_mult: float = 1.0) -> Intrinsics:
    """Intrinsics (same field of view as ``K``) for rendering one VR180 eye image.

    A 180 degree equirect of ``eye_px`` pixels has ``eye_px / pi`` pixels per radian; at
    the image centre a pinhole image has ``f`` pixels per radian (more toward the edges).
    ``size_mult`` enlarges the photo's field of view, i.e. divides the effective focal
    length by ``k = tan(fov / 2) / tan(fov' / 2)``.  The eye is rendered with
    ``f = VR180_DENSITY_MARGIN * eye_px / (pi * k)`` - never larger than ``K`` itself.
    Pixel-centre exact rescale (``c' = (c + 0.5) * s - 0.5``).
    """
    hfov = 2.0 * math.atan((K.width / 2.0) / K.fx)
    vfov = 2.0 * math.atan((K.height / 2.0) / K.fy)
    big = max(hfov, vfov)
    want = big * float(size_mult)
    if size_mult >= 1.0:
        want = min(want, max(math.radians(_VR180_MAX_FOV_DEG), big))
    k = math.tan(big / 2.0) / math.tan(min(want, math.radians(179.0)) / 2.0)
    need = VR180_DENSITY_MARGIN * eye_px / (math.pi * k)  # px per radian
    s = need / min(K.fx, K.fy)
    if s >= 1.0:
        return K
    w = max(8, int(round(K.width * s)))
    h = max(8, int(round(K.height * s)))
    sx, sy = w / K.width, h / K.height
    return Intrinsics(K.fx * sx, K.fy * sy, (K.cx + 0.5) * sx - 0.5, (K.cy + 0.5) * sy - 0.5, w, h)


# --------------------------------------------------------------------------- validation
def _check_inputs(li: LayeredImage, path: CameraPath) -> None:
    if not isinstance(li, LayeredImage) or not li.layers:
        raise InputError("Nothing to render: the layered image has no layers.")
    if not isinstance(path, CameraPath) or not path.poses:
        raise InputError("The camera path has no frames.")
    for i, pose in enumerate(path.poses):
        R, t = np.asarray(pose.R, dtype=np.float64), np.asarray(pose.t, dtype=np.float64)
        if R.shape != (3, 3) or t.shape != (3,) or not (np.isfinite(R).all() and np.isfinite(t).all()):
            raise InputError(
                f"Camera path frame {i + 1} has an invalid pose (needs a finite 3x3 R and 3-vector t)."
            )


def _progress(progress: ProgressReporter | None, i: int, n: int) -> None:
    if progress is not None:
        progress(i / max(n, 1), f"Rendering frame {min(i + 1, n)}/{n}")


# --------------------------------------------------------------------------- public API
def render_path_frames(
    li: LayeredImage,
    path: CameraPath,
    *,
    out_K: Intrinsics | None = None,
    stereo: StereoParams | None = None,
    layout: str = "mono",
    overscan: float = 0.04,
    cancel: CancelToken | None = None,
    progress: ProgressReporter | None = None,
) -> Iterator[RGB]:
    """Yield one hole-filled frame per pose of ``path``.

    Args:
        li: layered image (``li.K`` is the source camera).
        path: camera path (poses in the source frame).
        out_K: output camera (size / focal); default ``li.K``.
        stereo: stereo parameters for stereo layouts (default ``StereoParams()``, flat).
        layout: ``"mono"`` or a ``geometry.stereo.compose`` layout (``"sbs"``, ``"tb"``,
            ``"half_sbs"``, ``"anaglyph"``...).
        overscan: zoom-in fraction applied to the output focal length (0 disables).
        cancel: checked before each frame (and inside the warp when supported).
        progress: optional reporter, called once per frame (additive extension).

    Frames are ``HxWx3 uint8``; for ``"sbs"`` ``H x 2W``, ``"tb"`` ``2H x W``.  Arguments
    (and the stereo baseline / convergence) are resolved immediately; frames are rendered
    lazily, one per ``next()``.
    """
    _check_inputs(li, path)
    lay = str(layout).strip().lower()
    if lay not in FLAT_LAYOUTS:
        raise InputError(f"Unknown video layout {layout!r}. Choose one of: {', '.join(FLAT_LAYOUTS)}.")
    ov = float(overscan)
    if not (0.0 <= ov <= MAX_OVERSCAN):
        raise InputError(f"Overscan must be between 0 and {MAX_OVERSCAN} (got {overscan}).")
    K = overscan_K(out_K if out_K is not None else li.K, ov)
    if lay == "mono":
        return _mono_frames(li, path, K, cancel, progress)
    params = stereo if stereo is not None else StereoParams()
    B, zc = resolve_stereo(li, params, K)
    log.debug("Stereo video: baseline %.4f m, convergence %s m, layout %s", B, zc, lay)
    return _stereo_frames(li, path, K, B, zc, bool(params.swap_eyes), lay, cancel, progress)


def _mono_frames(
    li: LayeredImage,
    path: CameraPath,
    K: Intrinsics,
    cancel: CancelToken | None,
    progress: ProgressReporter | None,
) -> Iterator[RGB]:
    n = len(path.poses)
    for i, pose in enumerate(path.poses):
        if cancel is not None:
            cancel.check()
        _progress(progress, i, n)
        yield _render(li, pose, K, cancel)
    _progress(progress, n, n)


def _stereo_frames(
    li: LayeredImage,
    path: CameraPath,
    K: Intrinsics,
    B: float,
    zc: float,
    swap: bool,
    lay: str,
    cancel: CancelToken | None,
    progress: ProgressReporter | None,
) -> Iterator[RGB]:
    h = B / 2.0
    kl, kr = eye_K(K, -h, zc), eye_K(K, h, zc)
    n = len(path.poses)
    if lay in ("left", "right"):  # one eye only: do not render the other one
        x, ke = (-h, kl) if (lay == "left") != swap else (h, kr)
        for i, pose in enumerate(path.poses):
            if cancel is not None:
                cancel.check()
            _progress(progress, i, n)
            yield _render(li, eye_pose(pose, x), ke, cancel)
        _progress(progress, n, n)
        return
    compose = _stereo().compose
    for i, pose in enumerate(path.poses):
        if cancel is not None:
            cancel.check()
        _progress(progress, i, n)
        left = _render(li, eye_pose(pose, -h), kl, cancel)
        if cancel is not None:
            cancel.check()
        right = _render(li, eye_pose(pose, h), kr, cancel)
        if swap:
            left, right = right, left
        yield compose(left, right, lay)
    _progress(progress, n, n)


def render_vr180_frames(
    li: LayeredImage,
    path: CameraPath,
    *,
    eye_px: int,
    size_mult: float = 1.0,
    fill: str | RGB = "blur",
    cancel: CancelToken | None = None,
    stereo: StereoParams | None = None,
    layout: str = "sbs",
    progress: ProgressReporter | None = None,
) -> Iterator[RGB]:
    """Yield one VR180 stereo frame (``eye_px x 2*eye_px`` SBS) per pose of ``path``.

    Each eye is rendered with the source field of view (true angular size; holes filled)
    at the resolution the equirect needs (:func:`vr180_eye_K`), eyes at ``-/+ IPD/2``
    around the path pose with convergence at infinity (VR mode), then mapped with
    ``geometry.projection.vr180_frame`` (``size_mult``, ``fill``).

    Additive extensions: ``stereo`` (only ``ipd_m`` and ``swap_eyes`` are used; default
    IPD 0.063 m), ``layout`` (``"sbs"`` or ``"tb"``) and ``progress``.  Arguments are
    validated immediately; frames are rendered lazily.
    """
    _check_inputs(li, path)
    lay = str(layout).strip().lower()
    if lay not in VR180_LAYOUTS:
        raise InputError(f"Unknown VR180 layout {layout!r}; use 'sbs' or 'tb'.")
    try:
        eye = int(eye_px)
    except (TypeError, ValueError) as exc:
        raise InputError(f"VR180 eye size must be an integer (got {eye_px!r}).") from exc
    if eye < 8:
        raise InputError(f"VR180 eye size must be at least 8 px (got {eye_px}).")
    if not (float(size_mult) > 0 and math.isfinite(float(size_mult))):
        raise InputError(f"VR180 size multiplier must be positive (got {size_mult}).")
    ipd = float(stereo.ipd_m) if stereo is not None else StereoParams().ipd_m
    if not (ipd > 0 and math.isfinite(ipd)):
        raise InputError(f"Eye distance (IPD) must be positive (got {ipd} m).")
    swap = bool(stereo.swap_eyes) if stereo is not None else False
    return _vr180_frames(li, path, eye, float(size_mult), fill, ipd, swap, lay, cancel, progress)


def _vr180_frames(
    li: LayeredImage,
    path: CameraPath,
    eye: int,
    size_mult: float,
    fill: str | RGB,
    ipd: float,
    swap: bool,
    lay: str,
    cancel: CancelToken | None,
    progress: ProgressReporter | None,
) -> Iterator[RGB]:
    h = ipd / 2.0
    K = vr180_eye_K(li.K, eye, size_mult)
    if K is not li.K:
        log.debug("VR180 eyes rendered at %dx%d (photo %dx%d)", K.width, K.height, li.K.width, li.K.height)
    vr180_frame = _projection().vr180_frame
    n = len(path.poses)
    for i, pose in enumerate(path.poses):
        if cancel is not None:
            cancel.check()
        _progress(progress, i, n)
        left = _render(li, eye_pose(pose, -h), K, cancel)
        if cancel is not None:
            cancel.check()
        right = _render(li, eye_pose(pose, h), K, cancel)
        if swap:
            left, right = right, left
        yield vr180_frame(left, right, K, eye, size_mult=size_mult, fill=fill, layout=lay)
    _progress(progress, n, n)


__all__ = [
    "render_path_frames",
    "render_vr180_frames",
    "resolve_stereo",
    "eye_pose",
    "eye_K",
    "overscan_K",
    "vr180_eye_K",
    "FLAT_LAYOUTS",
    "VR180_LAYOUTS",
]
