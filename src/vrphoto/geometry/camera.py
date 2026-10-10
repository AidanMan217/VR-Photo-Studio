"""Pinhole camera helpers: (un)projection, frame conversion and intrinsics estimation.

Conventions (docs/CONTRACTS.md section 0):

* Source camera frame is OpenCV's: x right, y down, z forward, metres.
  ``u = fx*X/Z + cx``, ``v = fy*Y/Z + cy``; pixel (0, 0) is centred at ``u = 0, v = 0``.
* glTF frame: x right, y up, z toward the viewer.  ``cv_to_gltf(P) = P * [1, -1, -1]``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..core.errors import InputError
from ..core.log import get_logger
from ..core.types import DepthResult, FloatMap, Intrinsics

log = get_logger(__name__)

_FLIP_YZ = np.array([1.0, -1.0, -1.0])


def _float_dtype(a: np.ndarray) -> np.dtype:
    """float32 for float32/integer input, float64 for float64 input."""
    return np.dtype(np.float64) if a.dtype == np.float64 else np.dtype(np.float32)


def pixel_grid(K: Intrinsics, dtype: np.dtype | type = np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Return broadcastable pixel-centre coordinates ``(u[1, W], v[H, 1])`` for ``K``."""
    u = np.arange(K.width, dtype=dtype)[None, :]
    v = np.arange(K.height, dtype=dtype)[:, None]
    return u, v


def unproject(K: Intrinsics, depth_z: FloatMap) -> np.ndarray:
    """Back-project a z-depth map (metres along the optical axis) to 3D points.

    Returns an ``HxWx3`` array of source-frame points (x right, y down, z forward).
    ``depth_z`` must be ``K.height x K.width``.  Output dtype is float32 unless
    ``depth_z`` is float64.
    """
    depth_z = np.asarray(depth_z)
    if depth_z.ndim != 2:
        raise InputError(f"Depth map must be a 2-D array, got shape {depth_z.shape}.")
    if depth_z.shape != (K.height, K.width):
        raise InputError(
            f"Depth map size {depth_z.shape[1]}x{depth_z.shape[0]} does not match the camera "
            f"intrinsics {K.width}x{K.height}."
        )
    dt = _float_dtype(depth_z)
    z = depth_z.astype(dt, copy=False)
    u, v = pixel_grid(K, dt)
    xn = (u - dt.type(K.cx)) / dt.type(K.fx)
    yn = (v - dt.type(K.cy)) / dt.type(K.fy)
    out = np.empty(z.shape + (3,), dtype=dt)
    np.multiply(xn, z, out=out[..., 0])
    np.multiply(yn, z, out=out[..., 1])
    out[..., 2] = z
    return out


def project(K: Intrinsics, P: np.ndarray) -> np.ndarray:
    """Project ``...x3`` camera-frame points to pixel coordinates ``...x2`` (u, v).

    Points with ``z <= 1e-6`` (on or behind the camera plane) give NaN.
    """
    P = np.asarray(P)
    if P.shape[-1] != 3:
        raise InputError(f"Points must have a trailing dimension of 3, got shape {P.shape}.")
    dt = _float_dtype(P)
    P = P.astype(dt, copy=False)
    z = P[..., 2]
    valid = z > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(valid, 1.0 / np.where(valid, z, 1.0), np.nan).astype(dt, copy=False)
    out = np.empty(P.shape[:-1] + (2,), dtype=dt)
    np.multiply(P[..., 0], inv, out=out[..., 0])
    out[..., 0] *= dt.type(K.fx)
    out[..., 0] += dt.type(K.cx)
    np.multiply(P[..., 1], inv, out=out[..., 1])
    out[..., 1] *= dt.type(K.fy)
    out[..., 1] += dt.type(K.cy)
    return out


def cv_to_gltf(P: np.ndarray) -> np.ndarray:
    """Convert ``...x3`` points (or directions) from the OpenCV frame to the glTF frame."""
    P = np.asarray(P)
    if P.shape[-1] != 3:
        raise InputError(f"Points must have a trailing dimension of 3, got shape {P.shape}.")
    dt = P.dtype if np.issubdtype(P.dtype, np.floating) else np.dtype(np.float32)
    return (P * _FLIP_YZ.astype(dt)).astype(dt, copy=False)


def gltf_to_cv(P: np.ndarray) -> np.ndarray:
    """Convert ``...x3`` points from the glTF frame to the OpenCV frame (self-inverse)."""
    return cv_to_gltf(P)


def _valid_hfov(h: float | None) -> bool:
    return h is not None and math.isfinite(h) and 1.0 <= h <= 179.0


def _as_float(value: object, what: str) -> float:
    """``float(value)`` or :class:`InputError` naming ``what``."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise InputError(f"{what} must be a number of degrees (got {value!r}).") from e


def estimate_intrinsics(
    img_shape: Sequence[int],
    *,
    hfov_deg: float | None = None,
    exif_hfov: float | None = None,
    depth: DepthResult | None = None,
    default_hfov: float = 65.0,
) -> Intrinsics:
    """Choose pinhole intrinsics for an image of shape ``(H, W[, C])``.

    Priority: explicit ``hfov_deg`` > ``depth.intrinsics`` (models that estimate the
    camera, e.g. MoGe; rescaled to the image size) > ``exif_hfov`` > ``default_hfov``.
    ``hfov_deg`` and ``exif_hfov`` are horizontal fields of view of the displayed image;
    ``default_hfov`` applies to the LONG side (a typical camera is ~65 deg across its
    long side whichever way the photo is held), so an EXIF-less portrait photo gets 65 deg
    vertically, not horizontally.  An explicit but invalid ``hfov_deg`` raises
    :class:`InputError`; an implausible EXIF value is ignored with a warning.
    """
    try:
        if len(img_shape) < 2:
            raise ValueError
        h, w = int(img_shape[0]), int(img_shape[1])
    except (TypeError, ValueError) as e:
        raise InputError(f"Image shape must be (H, W[, C]), got {img_shape!r}.") from e
    if h <= 0 or w <= 0:
        raise InputError(f"Invalid image size {w}x{h}.")
    if hfov_deg is not None:
        hf = _as_float(hfov_deg, "Field of view")
        if not _valid_hfov(hf):
            raise InputError(
                f"Field of view {hfov_deg} deg is not valid; use a value between 1 and 179 degrees."
            )
        return Intrinsics.from_hfov(w, h, hf)
    if depth is not None and depth.intrinsics is not None:
        k = depth.intrinsics
        if k.fx > 0 and k.fy > 0 and k.width > 0 and k.height > 0:
            if (k.width, k.height) != (w, h):
                k = k.resized(w, h)
            if _valid_hfov(k.hfov_deg):
                return k
        log.warning("Ignoring implausible model-estimated intrinsics %s", depth.intrinsics)
    if exif_hfov is not None:
        try:
            ef = float(exif_hfov)
        except (TypeError, ValueError):
            ef = math.nan
        if _valid_hfov(ef):
            return Intrinsics.from_hfov(w, h, ef)
        log.warning("Ignoring implausible EXIF field of view %r deg", exif_hfov)
    df = _as_float(default_hfov, "Default field of view")
    if not _valid_hfov(df):
        raise InputError(f"Default field of view {default_hfov} deg is not valid.")
    if w >= h:
        return Intrinsics.from_hfov(w, h, df)
    # portrait: the default spans the long (vertical) side; square pixels, centred
    f = (h / 2.0) / math.tan(math.radians(df) / 2.0)
    return Intrinsics(f, f, w / 2.0, h / 2.0, w, h)


__all__ = ["unproject", "project", "cv_to_gltf", "gltf_to_cv", "estimate_intrinsics", "pixel_grid"]
