"""Spherical projections: equirect <-> perspective / cubemap, VR180 frames and mono 360 panos.

Conventions (docs/CONTRACTS.md section 0 and algorithm E):

* World / source frame is the OpenCV camera frame: x right, **y down**, z forward.
* Equirect pixel ``(i, j)`` of a ``W x H`` image covering ``lon_range`` x ``lat_range``
  (degrees) has longitude ``lambda = lon0 + (i + .5) / W * (lon1 - lon0)`` and latitude
  ``phi = lat1 - (j + .5) / H * (lat1 - lat0)`` (top row = highest latitude), and looks
  along ``d = (cos(phi) sin(lambda), -sin(phi), cos(phi) cos(lambda))``.  Longitude
  grows to the RIGHT, latitude grows UP.  Pixel centres are integer coordinates, so the
  outer pixel *edges* (``i = -0.5`` and ``i = W - 0.5``) sit exactly on the range bounds.
* Perspective images follow :class:`~vrphoto.core.types.Intrinsics`:
  ``u = fx * X / Z + cx``.  A camera turned by ``yaw_deg`` / ``pitch_deg`` uses the
  rotation of :meth:`~vrphoto.core.types.Pose.look` (yaw > 0 turns right, pitch > 0 looks up).

All samplers work in tiles (bounded memory for 24 MP inputs / 16K equirects), wrap
longitude across the +-180 deg seam and continue over the poles for full spheres, and
pre-filter the source with area averaging when it is much denser than the output
(prevents moire when e.g. a 24 MP photo becomes part of a 4K VR180 frame).
Any dtype accepted by ``cv2.remap`` (uint8, uint16, int16, float32, float64) and any
channel count (HxW or HxWxC, C <= 4 recommended) is supported unless noted otherwise.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence

import cv2
import numpy as np

from ..core.errors import InputError
from ..core.fill import push_pull_fill
from ..core.log import get_logger
from ..core.types import RGB, Intrinsics, Pose

log = get_logger(__name__)

__all__ = [
    "CUBE_FACES",
    "clear_map_cache",
    "cube_face_rotation",
    "equirect_dirs",
    "persp_to_equirect",
    "equirect_to_persp",
    "equirect_to_cubemap",
    "cubemap_to_equirect",
    "push_pull_fill",
    "vr180_frame",
    "pano_from_photo",
]

#: Cube face keys in the order used by this module (cv frame: +y points DOWN).
CUBE_FACES: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")

# Columns are the face camera's (right, down, forward) axes expressed in the world cv frame,
# i.e. the same convention as ``Pose.R``.  A face pixel (u, v) looks along
# R @ ((u - c) / f, (v - c) / f, 1) with f = face / 2 and c = (face - 1) / 2.
_FACE_AXES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "+x": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),  # looking right: image right = back (-z)
    "-x": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),  # looking left: image right = front (+z)
    "+y": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),  # looking down: image top = front (+z)
    "-y": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),  # looking up: image bottom = front (+z)
    "+z": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),  # front: identity
    "-z": ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),  # back: image right = left (-x)
}
_FACE_R: dict[str, np.ndarray] = {
    k: np.array(axes, dtype=np.float64).T.copy() for k, axes in _FACE_AXES.items()
}

_SHRT = 32767  # cv2.remap requires src and dst sizes < SHRT_MAX
_MAX_SRC = _SHRT - 16
_TILE_PX = 1 << 20  # output pixels per remap tile (bounds temporary memory)
_TILE_W = 8192
_PAD = 2  # source padding (wrap / over-the-pole / replicate) for bilinear sampling
_AA_RATIO = 1.25  # pre-filter the source when it is this much denser than the output
_MAX_VR180_FOV = 150.0
_FILL_WORK_LONG = 768  # working resolution of the blurred surround fill
_FILL_SIGMA_FRAC = 0.02
_FILL_DARKEN = 0.40

_FAR = -1.0e5  # remap coordinate used for "no source" pixels (constant-0 border)
_MAP_CACHE_MAX_PX = 4096 * 4096  # cache photo->equirect maps up to this output size

MapFn = Callable[[int, int, int, int], tuple[np.ndarray, np.ndarray, "np.ndarray | None"]]


# --------------------------------------------------------------------------- helpers
def cube_face_rotation(key: str) -> np.ndarray:
    """Rotation (3x3, columns = right/down/forward axes in the cv world frame) of cube face ``key``.

    Same convention as :class:`~vrphoto.core.types.Pose` ``R``: a face-camera ray ``r``
    points along ``R @ r`` in the world.  See :func:`equirect_to_cubemap` for the layout.
    """
    try:
        return _FACE_R[key].copy()
    except KeyError:
        raise InputError(f"Unknown cube face {key!r}; expected one of {', '.join(CUBE_FACES)}") from None


def _as_image(img: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3) or img.shape[0] < 1 or img.shape[1] < 1:
        raise InputError(f"{name} must be an HxW or HxWxC image array")
    if img.ndim == 3 and not 1 <= img.shape[2] <= 4:
        raise InputError(f"{name} must have 1-4 channels (got {img.shape[2]})")
    if img.dtype == np.bool_:
        img = img.astype(np.uint8)
    elif img.dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
        img = img.astype(np.float32)
    return img


def _range(r: Sequence[float], name: str) -> tuple[float, float]:
    try:
        a, b = float(r[0]), float(r[1])
    except (TypeError, ValueError, IndexError):
        raise InputError(f"{name} must be a (min, max) pair of degrees") from None
    if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
        raise InputError(f"{name} must satisfy min < max (got {a}, {b})")
    return math.radians(a), math.radians(b)


def _as_int(value: object, name: str) -> int:
    """``int(value)`` for numbers / numeric strings, InputError (not ValueError) otherwise."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a whole number (got {value!r})") from None
    if not math.isfinite(f):
        raise InputError(f"{name} must be a finite number (got {value!r})")
    return int(f)


def _as_float(value: object, name: str) -> float:
    """``float(value)`` with a human-readable InputError for non-numbers (NaN passes through)."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a number (got {value!r})") from None


def _check_size(w: int, h: int, what: str) -> tuple[int, int]:
    w, h = _as_int(w, f"{what} width"), _as_int(h, f"{what} height")
    if w < 1 or h < 1:
        raise InputError(f"{what} size must be positive (got {w}x{h})")
    return w, h


def _tiles(h: int, w: int) -> Iterator[tuple[int, int, int, int]]:
    tw = min(w, _TILE_W)
    th = max(1, min(h, _TILE_PX // max(1, tw)))
    for j0 in range(0, h, th):
        for i0 in range(0, w, tw):
            yield j0, min(h, j0 + th), i0, min(w, i0 + tw)


def _remap_tiled(srcs: Sequence[np.ndarray], out_h: int, out_w: int, map_fn: MapFn, *,
                 border: int = cv2.BORDER_CONSTANT, want_aux: bool = False
                 ) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Tiled bilinear ``cv2.remap`` of one or more same-size sources with shared maps.

    ``map_fn(j0, j1, i0, i1)`` returns float32 ``(mapx, mapy, aux)`` for the output tile.
    With the default constant border, coordinates far outside the source give 0 (callers
    map invalid pixels to ``_FAR`` for that).  ``aux`` (float32 or None) is collected into
    a full-size array when ``want_aux``."""
    outs = [np.empty((out_h, out_w) + src.shape[2:], dtype=src.dtype) for src in srcs]
    aux_out = np.zeros((out_h, out_w), np.float32) if want_aux else None
    for j0, j1, i0, i1 in _tiles(out_h, out_w):
        mx, my, aux = map_fn(j0, j1, i0, i1)
        if len(srcs) > 1:  # convert once, reuse for every source
            mx, my = cv2.convertMaps(mx, my, cv2.CV_16SC2)
        for src, out in zip(srcs, outs):
            tile = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=border, borderValue=0)
            out[j0:j1, i0:i1] = tile.reshape(out[j0:j1, i0:i1].shape)
        if aux_out is not None and aux is not None:
            aux_out[j0:j1, i0:i1] = aux
    return outs, aux_out


def _lon_axis(w: int, lon0: float, lon1: float, i0: int, i1: int) -> np.ndarray:
    return lon0 + (np.arange(i0, i1, dtype=np.float64) + 0.5) / w * (lon1 - lon0)


def _lat_axis(h: int, lat0: float, lat1: float, j0: int, j1: int) -> np.ndarray:
    return lat1 - (np.arange(j0, j1, dtype=np.float64) + 0.5) / h * (lat1 - lat0)


def _eq_dirs_tile(w: int, h: int, lon: tuple[float, float], lat: tuple[float, float],
                  j0: int, j1: int, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam = _lon_axis(w, lon[0], lon[1], i0, i1)[None, :]
    phi = _lat_axis(h, lat[0], lat[1], j0, j1)[:, None]
    cphi = np.cos(phi)
    x = cphi * np.sin(lam)
    y = np.broadcast_to(-np.sin(phi), (j1 - j0, i1 - i0))
    z = cphi * np.cos(lam)
    return x, y, z


def _rotate(x: np.ndarray, y: np.ndarray, z: np.ndarray, M: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``M @ (x, y, z)`` component-wise."""
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2] * z,
            M[1, 0] * x + M[1, 1] * y + M[1, 2] * z,
            M[2, 0] * x + M[2, 1] * y + M[2, 2] * z)


def _is_full_lon(lon: tuple[float, float]) -> bool:
    return abs((lon[1] - lon[0]) - 2.0 * math.pi) < 1e-9


def _is_full_lat(lat: tuple[float, float]) -> bool:
    return abs(lat[0] + math.pi / 2) < 1e-9 and abs(lat[1] - math.pi / 2) < 1e-9


def _pad_equirect(eq: np.ndarray, p: int, full_lon: bool, full_lat: bool) -> np.ndarray:
    """Pad by ``p`` pixels: longitude wraps (full 360), rows continue over the poles
    (full sphere: row -1-k is row k turned by 180 deg), otherwise edges replicate."""
    h, w = eq.shape[:2]
    out = np.empty((h + 2 * p, w + 2 * p) + eq.shape[2:], dtype=eq.dtype)
    out[p:p + h, p:p + w] = eq
    if full_lon and full_lat and h >= p:
        half = w // 2
        out[:p, p:p + w] = np.roll(eq[:p], -half, axis=1)[::-1]
        out[p + h:, p:p + w] = np.roll(eq[h - p:], -half, axis=1)[::-1]
    else:
        out[:p, p:p + w] = eq[:1]
        out[p + h:, p:p + w] = eq[h - 1:]
    if full_lon and w >= p:
        out[:, :p] = out[:, w:w + p]
        out[:, w + p:] = out[:, p:2 * p]
    else:
        out[:, :p] = out[:, p:p + 1]
        out[:, w + p:] = out[:, w + p - 1:w + p]
    return out


def _area_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    out = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    if out.ndim < img.ndim:
        out = out[..., None]
    return out


def _fit_equirect_source(eq: np.ndarray, max_ppr: float | None, span: float) -> np.ndarray:
    """Area-downscale an equirect when it is much denser (px/rad) than needed or too large
    for ``cv2.remap``.  Resizing keeps the algorithm-E mapping exact (edges stay on bounds)."""
    h, w = eq.shape[:2]
    s = 1.0
    if max_ppr is not None and max_ppr > 0:
        ppr = w / span
        if ppr > _AA_RATIO * max_ppr:
            s = max_ppr / ppr
    s = min(s, _MAX_SRC / (max(h, w) + 2 * _PAD))
    if s >= 1.0:
        return eq
    nw, nh = max(2, int(round(w * s))), max(1, int(round(h * s)))
    if nw >= w and nh >= h:
        return eq
    return _area_resize(eq, nw, nh)


def _fit_persp_source(img: np.ndarray, K: Intrinsics, max_ppr: float | None) -> tuple[np.ndarray, Intrinsics]:
    """Area-downscale a perspective image that is much denser than the output; returns the
    image and its exactly rescaled intrinsics (pixel-centre convention)."""
    h, w = img.shape[:2]
    if (K.width, K.height) != (w, h):
        K = _scale_K(K, w / K.width, h / K.height, w, h)
    s = 1.0
    if max_ppr is not None and max_ppr > 0:
        ppr = max(K.fx, K.fy)
        if ppr > _AA_RATIO * max_ppr:
            s = max_ppr / ppr
    s = min(s, _MAX_SRC / (max(h, w) + 2 * _PAD))
    if s >= 1.0:
        return img, K
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if nw >= w and nh >= h:
        return img, K
    return _area_resize(img, nw, nh), _scale_K(K, nw / w, nh / h, nw, nh)


def _scale_K(K: Intrinsics, sx: float, sy: float, w: int, h: int) -> Intrinsics:
    return Intrinsics(K.fx * sx, K.fy * sy, (K.cx + 0.5) * sx - 0.5, (K.cy + 0.5) * sy - 0.5, int(w), int(h))


def _sample_equirect(eq: np.ndarray, out_h: int, out_w: int,
                     dirs_fn: Callable[[int, int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
                     lon_range: Sequence[float] = (-180.0, 180.0), lat_range: Sequence[float] = (-90.0, 90.0),
                     *, max_ppr: float | None = None) -> np.ndarray:
    """Sample ``eq`` along world directions ``dirs_fn(j0, j1, i0, i1) -> (x, y, z)``.

    Directions outside a partial ``lon_range`` / ``lat_range`` give 0 (black)."""
    lon = _range(lon_range, "lon_range")
    lat = _range(lat_range, "lat_range")
    eq = _fit_equirect_source(eq, max_ppr, lon[1] - lon[0])
    h, w = eq.shape[:2]
    full_lon, full_lat = _is_full_lon(lon), _is_full_lat(lat)
    src = _pad_equirect(eq, _PAD, full_lon, full_lat)
    lon_c = 0.5 * (lon[0] + lon[1])
    partial = not (full_lon and full_lat)

    def map_fn(j0: int, j1: int, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        x, y, z = dirs_fn(j0, j1, i0, i1)
        lam = np.arctan2(x, z)
        phi = np.arctan2(-y, np.hypot(x, z))
        if full_lon:
            lam = lon[0] + np.mod(lam - lon[0], 2.0 * math.pi)
        else:
            lam = lon_c + np.mod(lam - lon_c + math.pi, 2.0 * math.pi) - math.pi
        px = (lam - lon[0]) / (lon[1] - lon[0]) * w - 0.5
        py = (lat[1] - phi) / (lat[1] - lat[0]) * h - 0.5
        mx = np.clip(px + _PAD, 0.0, w + 2 * _PAD - 1.0)
        my = np.clip(py + _PAD, 0.0, h + 2 * _PAD - 1.0)
        if partial:
            ok = (px >= -0.5) & (px <= w - 0.5) & (py >= -0.5) & (py <= h - 0.5)
            mx = np.where(ok, mx, _FAR)
            my = np.where(ok, my, _FAR)
        return mx.astype(np.float32), my.astype(np.float32), None

    return _remap_tiled([src], out_h, out_w, map_fn)[0][0]


def _persp_dirs_fn(K: Intrinsics, R: np.ndarray
                   ) -> Callable[[int, int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    def fn(j0: int, j1: int, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = ((np.arange(i0, i1, dtype=np.float64) - K.cx) / K.fx)[None, :]
        b = ((np.arange(j0, j1, dtype=np.float64) - K.cy) / K.fy)[:, None]
        shape = (j1 - j0, i1 - i0)
        return _rotate(np.broadcast_to(a, shape), np.broadcast_to(b, shape), np.ones(shape), R)

    return fn


def _equirect_to_persp_R(eq: np.ndarray, K: Intrinsics, R: np.ndarray,
                         lon_range: Sequence[float] = (-180.0, 180.0)) -> np.ndarray:
    """Render a pinhole view ``K`` with camera rotation ``R`` (Pose convention) from ``eq``."""
    eq = _as_image(eq, "eq")
    _check_K(K)
    w, h = _check_size(K.width, K.height, "Output")
    return _sample_equirect(eq, h, w, _persp_dirs_fn(K, R), lon_range, max_ppr=min(K.fx, K.fy))


_map_cache_lock = threading.Lock()
_map_cache: list[tuple[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []  # at most one entry


def _persp_eq_maps(Ks: Intrinsics, out_w: int, out_h: int, lon: tuple[float, float],
                   lat: tuple[float, float], R: np.ndarray) -> MapFn:
    """Map function (equirect tile -> padded photo coordinates, coverage) for a pinhole photo."""
    ws, hs = Ks.width, Ks.height
    out_ppr = max(out_w / (lon[1] - lon[0]), out_h / (lat[1] - lat[0]))
    scale = out_ppr / max(Ks.fx, Ks.fy)  # output px per source px (at the optical axis)
    Rt = np.asarray(R, dtype=np.float64).T

    def map_fn(j0: int, j1: int, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = _eq_dirs_tile(out_w, out_h, lon, lat, j0, j1, i0, i1)
        xc, yc, zc = _rotate(x, y, z, Rt)
        front = zc > 1e-6
        inv = 1.0 / np.where(front, zc, 1.0)
        u = Ks.fx * xc * inv + Ks.cx
        v = Ks.fy * yc * inv + Ks.cy
        edge = np.minimum(np.minimum(u + 0.5, ws - 0.5 - u), np.minimum(v + 0.5, hs - 0.5 - v))
        cov = np.where(front, np.clip(edge * scale + 0.5, 0.0, 1.0), 0.0)
        has = cov > 0.0
        mx = np.where(has, u + _PAD, _FAR).astype(np.float32)
        my = np.where(has, v + _PAD, _FAR).astype(np.float32)
        return mx, my, cov.astype(np.float32)

    return map_fn


def _cached_maps(key: tuple, out_w: int, out_h: int, map_fn: MapFn
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-size fixed-point remap maps + coverage, memoised for the last parameter set
    (VR180 / 360 video renders every frame with the same camera)."""
    with _map_cache_lock:
        for k, v in _map_cache:
            if k == key:
                return v
    m1 = np.empty((out_h, out_w, 2), np.int16)
    m2 = np.empty((out_h, out_w), np.uint16)
    cov = np.empty((out_h, out_w), np.float32)
    for j0, j1, i0, i1 in _tiles(out_h, out_w):
        mx, my, c = map_fn(j0, j1, i0, i1)
        a, b = cv2.convertMaps(mx, my, cv2.CV_16SC2)
        m1[j0:j1, i0:i1] = a
        m2[j0:j1, i0:i1] = b
        cov[j0:j1, i0:i1] = c
    for arr in (m1, m2, cov):
        arr.setflags(write=False)  # shared between calls / threads
    val = (m1, m2, cov)
    with _map_cache_lock:
        _map_cache[:] = [(key, val)]
    return val


def clear_map_cache() -> None:
    """Release the memoised photo->equirect remap maps."""
    with _map_cache_lock:
        _map_cache.clear()


def _check_K(K: Intrinsics) -> None:
    try:
        vals = [float(v) for v in (K.fx, K.fy, K.cx, K.cy)]
        size = (int(K.width), int(K.height))
    except (AttributeError, TypeError, ValueError):
        raise InputError(f"Invalid camera intrinsics: {K!r}") from None
    if not all(math.isfinite(v) for v in vals) or vals[0] <= 0 or vals[1] <= 0 or min(size) < 1:
        raise InputError(f"Invalid camera intrinsics: {K}")


def _persp_to_equirect_multi(imgs: Sequence[np.ndarray], K: Intrinsics, out_w: int, out_h: int,
                             R: np.ndarray, lon_range: Sequence[float],
                             lat_range: Sequence[float] = (-90.0, 90.0)) -> tuple[list[np.ndarray], np.ndarray]:
    """Map same-size pinhole images (sharing ``K`` and rotation ``R``) onto an equirect.

    Returns (images, coverage float32 in [0,1]).  Coverage is an anti-aliased footprint of
    the photo (>= 0.5 exactly inside the photo's pixel area); pixels with zero coverage are
    0, pixels on the rim keep the edge colour."""
    _check_K(K)
    imgs = [_as_image(im, "img") for im in imgs]
    if any(im.shape != imgs[0].shape or im.dtype != imgs[0].dtype for im in imgs):
        raise InputError("Images mapped together must have the same size and type")
    out_w, out_h = _check_size(out_w, out_h, "Output")
    lon = _range(lon_range, "lon_range")
    lat = _range(lat_range, "lat_range")
    out_ppr = max(out_w / (lon[1] - lon[0]), out_h / (lat[1] - lat[0]))
    padded = []
    Ks = K
    for im in imgs:
        src, Ks = _fit_persp_source(im, K, out_ppr)
        pad = cv2.copyMakeBorder(src, _PAD, _PAD, _PAD, _PAD, cv2.BORDER_REPLICATE)
        padded.append(pad.reshape(pad.shape[:2] + src.shape[2:]))
    map_fn = _persp_eq_maps(Ks, out_w, out_h, lon, lat, R)
    # the cached path remaps the whole canvas at once: cv2.remap needs both sides < SHRT_MAX
    if out_w * out_h <= _MAP_CACHE_MAX_PX and max(out_w, out_h) <= _TILE_W:
        key = (tuple(float(v) for v in (Ks.fx, Ks.fy, Ks.cx, Ks.cy)), Ks.width, Ks.height, out_w, out_h,
               lon, lat, np.asarray(R, dtype=np.float64).tobytes())
        m1, m2, cov = _cached_maps(key, out_w, out_h, map_fn)
        outs = []
        for src in padded:
            o = cv2.remap(src, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            outs.append(o.reshape((out_h, out_w) + src.shape[2:]))
        return outs, cov
    outs, cov_t = _remap_tiled(padded, out_h, out_w, map_fn, want_aux=True)
    assert cov_t is not None
    return outs, cov_t


def _persp_to_equirect_cov(img: np.ndarray, K: Intrinsics, out_w: int, out_h: int, R: np.ndarray,
                           lon_range: Sequence[float], lat_range: Sequence[float] = (-90.0, 90.0)
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Single-image :func:`_persp_to_equirect_multi`."""
    outs, cov = _persp_to_equirect_multi([img], K, out_w, out_h, R, lon_range, lat_range)
    return outs[0], cov


# --------------------------------------------------------------------------- public API
def equirect_dirs(out_w: int, out_h: int, lon_range: Sequence[float] = (-180.0, 180.0),
                  lat_range: Sequence[float] = (-90.0, 90.0)) -> np.ndarray:
    """Unit view directions (cv frame) of every pixel of an equirect, ``HxWx3`` float32.

    Implements algorithm E: pixel ``(i, j)`` -> ``lambda = lon0 + (i+.5)/W*(lon1-lon0)``,
    ``phi = lat1 - (j+.5)/H*(lat1-lat0)``, ``d = (cos phi sin lambda, -sin phi, cos phi cos lambda)``.
    Uses ``12*W*H`` bytes (filled in row tiles, so there are no large float64 temporaries);
    prefer the tiled samplers in this module for very large images.
    """
    out_w, out_h = _check_size(out_w, out_h, "Equirect")
    lon = _range(lon_range, "lon_range")
    lat = _range(lat_range, "lat_range")
    out = np.empty((out_h, out_w, 3), np.float32)
    rows = max(1, _TILE_PX // out_w)
    for j0 in range(0, out_h, rows):
        j1 = min(out_h, j0 + rows)
        x, y, z = _eq_dirs_tile(out_w, out_h, lon, lat, j0, j1, 0, out_w)
        out[j0:j1, :, 0] = x
        out[j0:j1, :, 1] = y
        out[j0:j1, :, 2] = z
    return out


def persp_to_equirect(img: RGB, K: Intrinsics, out_w: int, out_h: int, *,
                      lon_range: Sequence[float] = (-90.0, 90.0), yaw_deg: float = 0.0,
                      pitch_deg: float = 0.0) -> tuple[RGB, np.ndarray]:
    """Place a pinhole photo on an equirect canvas at its true angular size.

    The canvas is ``out_w x out_h`` covering ``lon_range`` (default VR180 hemisphere) and
    latitudes -90..90.  The photo's optical axis points at longitude ``yaw_deg`` (> 0 =
    right) and latitude ``pitch_deg`` (> 0 = up), i.e. the photo is rotated with
    ``Pose.look(yaw_deg, pitch_deg)``.  ``K`` describes ``img`` (rescaled automatically if
    the image size differs).  Returns ``(image, valid)`` where ``valid`` is a bool mask of
    canvas pixels whose centre falls inside the photo; other pixels are 0.
    """
    R = Pose.look(yaw_deg, pitch_deg).R
    out, cov = _persp_to_equirect_cov(img, K, out_w, out_h, R, lon_range)
    valid = cov >= 0.5
    out[~valid] = 0
    return out, valid


def equirect_to_persp(eq: RGB, K: Intrinsics, *, yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                      lon_range: Sequence[float] = (-180.0, 180.0)) -> RGB:
    """Render a ``K.width x K.height`` pinhole view from an equirect covering ``lon_range``
    (latitudes -90..90).  The virtual camera looks toward longitude ``yaw_deg`` / latitude
    ``pitch_deg`` (same rotation as :func:`persp_to_equirect`, so the two are inverses).
    Directions outside a partial ``lon_range`` are black.  Longitude wraps for full 360."""
    return _equirect_to_persp_R(eq, K, Pose.look(yaw_deg, pitch_deg).R, lon_range)


def _face_K(face: int, fov_deg: float = 90.0) -> Intrinsics:
    f = (face / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    c = (face - 1) / 2.0
    return Intrinsics(f, f, c, c, int(face), int(face))


def equirect_to_cubemap(eq: RGB, face: int) -> dict[str, RGB]:
    """Split a full 360x180 equirect into six ``face x face`` cube faces (90 deg FOV each).

    Keys ``"+x" "-x" "+y" "-y" "+z" "-z"`` name the face's viewing direction in the cv
    frame (+x right, +y DOWN, +z forward).  Each face is a pinhole image with
    ``f = face/2`` and principal point ``((face-1)/2, (face-1)/2)`` (so the outer pixel
    edges lie exactly on the cube edges) and this orientation (image right / image down):

    ========  ===========  ==============  ==========================================
    face      looks at     image right     image down
    ========  ===========  ==============  ==========================================
    ``+z``    front        +x (right)      +y (down)
    ``-z``    back         -x (left)       +y (down)
    ``+x``    right        -z (back)       +y (down)
    ``-x``    left         +z (front)      +y (down)
    ``+y``    floor        +x (right)      -z (back)  - top edge touches ``+z``
    ``-y``    sky          +x (right)      +z (front) - bottom edge touches ``+z``
    ========  ===========  ==============  ==========================================

    i.e. the four side faces are upright views after turning (yaw) 0/180/+90/-90 deg,
    ``+y`` is the view after pitching down 90 deg and ``-y`` after pitching up 90 deg.
    """
    eq = _as_image(eq, "eq")
    face = _as_int(face, "Cube face size")
    if face < 1:
        raise InputError(f"Cube face size must be positive (got {face})")
    K = _face_K(face)
    # face pixels near the cube edges are denser than at the centre: keep 1.5x headroom.
    # Prefilter once for all six faces (each sample call would otherwise re-downscale).
    ppr = 1.5 * K.fx
    eq = _fit_equirect_source(eq, ppr, 2.0 * math.pi)
    return {k: _sample_equirect(eq, face, face, _persp_dirs_fn(K, _FACE_R[k]), max_ppr=ppr)
            for k in CUBE_FACES}


def _dominant_face(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Index into CUBE_FACES of the face hit by each direction."""
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    idx = np.where(z >= 0, 4, 5)
    idx = np.where((ay >= az) & (ay > ax), np.where(y >= 0, 2, 3), idx)
    idx = np.where((ax >= ay) & (ax >= az), np.where(x >= 0, 0, 1), idx)
    return idx.astype(np.int8)


def _face_coords(x: np.ndarray, y: np.ndarray, z: np.ndarray, idx: np.ndarray, n: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Face pixel coordinates (u, v) of directions on their face ``idx`` (cube of size n)."""
    f = n / 2.0
    c = (n - 1) / 2.0
    comps = (x, y, z)
    signed: dict[tuple[int, int], np.ndarray] = {}

    def pick(row: int) -> np.ndarray:
        # every face rotation is a signed permutation: camera axis `row` = sign * world axis
        choices = []
        for key in CUBE_FACES:
            r = _FACE_R[key].T[row]
            ax = int(np.argmax(np.abs(r)))
            sg = int(np.sign(r[ax]))
            if (ax, sg) not in signed:
                signed[(ax, sg)] = comps[ax] if sg > 0 else -comps[ax]
            choices.append(signed[(ax, sg)])
        return np.choose(idx, choices)

    xc, yc, zc = pick(0), pick(1), pick(2)
    zc = np.maximum(zc, 1e-12)
    return f * xc / zc + c, f * yc / zc + c


def cubemap_to_equirect(faces: Mapping[str, np.ndarray], out_w: int, out_h: int) -> np.ndarray:
    """Assemble an ``out_w x out_h`` full 360x180 equirect from six cube faces.

    ``faces`` uses the keys / orientation of :func:`equirect_to_cubemap`; all faces must be
    square, the same size and dtype.  Interpolation is seamless across face edges (each face
    is padded with samples from its neighbours before bilinear sampling).
    """
    out_w, out_h = _check_size(out_w, out_h, "Equirect")
    missing = [k for k in CUBE_FACES if k not in faces]
    if missing:
        raise InputError(f"Cubemap is missing face(s): {', '.join(missing)}")
    arrs = [_as_image(np.asarray(faces[k]), f"face {k}") for k in CUBE_FACES]
    n = arrs[0].shape[0]
    for k, a in zip(CUBE_FACES, arrs):
        if a.shape[0] != a.shape[1] or a.shape != arrs[0].shape or a.dtype != arrs[0].dtype:
            raise InputError(f"Cube face {k} must be square and match the other faces "
                             f"(got {a.shape} {a.dtype}, expected {arrs[0].shape} {arrs[0].dtype})")
    # anti-alias / remap size limit: face density at the centre is n/2 px/rad
    target = out_w / (2.0 * math.pi)
    s = 1.0
    if n / 2.0 > _AA_RATIO * target:
        s = target / (n / 2.0)
    s = min(s, (_MAX_SRC / 3.0 - 2 * _PAD - 1) / n)
    if s < 1.0 and int(round(n * s)) < n:
        n2 = max(2, int(round(n * s)))
        arrs = [_area_resize(a, n2, n2) for a in arrs]
        n = n2
    squeeze = arrs[0].ndim == 2
    arrs = [a[..., None] if a.ndim == 2 else a for a in arrs]
    ch = arrs[0].shape[2:]
    # unpadded atlas: 3 columns x 2 rows of n x n tiles
    atlas = np.empty((2 * n, 3 * n) + ch, dtype=arrs[0].dtype)
    for k, a in enumerate(arrs):
        r, c = divmod(k, 3)
        atlas[r * n:(r + 1) * n, c * n:(c + 1) * n] = a
    # padded atlas: each tile extended by _PAD pixels sampled from the neighbouring faces
    m = n + 2 * _PAD
    padded = np.empty((2 * m, 3 * m) + ch, dtype=atlas.dtype)
    Kf = _face_K(n)
    g = np.arange(-_PAD, n + _PAD)
    gu, gv = np.meshgrid(g, g)
    ring = (gu < 0) | (gu >= n) | (gv < 0) | (gv >= n)
    ru, rv = gu[ring].astype(np.float64), gv[ring].astype(np.float64)
    cols = 4096  # remap needs maps < 32767 wide: lay the ring out as a 2-D block
    nr = int(math.ceil(ru.size / cols))
    for k, key in enumerate(CUBE_FACES):
        r, c = divmod(k, 3)
        tile = padded[r * m:(r + 1) * m, c * m:(c + 1) * m]
        tile[_PAD:_PAD + n, _PAD:_PAD + n] = arrs[k]
        x, y, z = _rotate((ru - Kf.cx) / Kf.fx, (rv - Kf.cy) / Kf.fy, np.ones_like(ru), _FACE_R[key])
        idx = _dominant_face(x, y, z)
        u, v = _face_coords(x, y, z, idx, n)
        rr, cc = np.divmod(idx.astype(np.int64), 3)
        mx = np.zeros(nr * cols, np.float32)
        my = np.zeros(nr * cols, np.float32)
        mx[:ru.size] = np.clip(u, 0.0, n - 1.0) + cc * n
        my[:ru.size] = np.clip(v, 0.0, n - 1.0) + rr * n
        vals = cv2.remap(atlas, mx.reshape(nr, cols), my.reshape(nr, cols), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        tile[ring] = vals.reshape((nr * cols,) + ch)[:ru.size]
    lon = (-math.pi, math.pi)
    lat = (-math.pi / 2, math.pi / 2)

    def map_fn(j0: int, j1: int, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray, None]:
        x, y, z = _eq_dirs_tile(out_w, out_h, lon, lat, j0, j1, i0, i1)
        y = np.ascontiguousarray(y)
        idx = _dominant_face(x, y, z)
        u, v = _face_coords(x, y, z, idx, n)
        rr, cc = np.divmod(idx.astype(np.int64), 3)
        mx = (np.clip(u, -0.5, n - 0.5) + _PAD + cc * m).astype(np.float32)
        my = (np.clip(v, -0.5, n - 0.5) + _PAD + rr * m).astype(np.float32)
        return mx, my, None

    out = _remap_tiled([padded], out_h, out_w, map_fn, border=cv2.BORDER_REPLICATE)[0][0]
    return out[..., 0] if squeeze else out


# --------------------------------------------------------------------------- surround fill
def _parse_fill(fill: object) -> tuple[str, np.ndarray | None]:
    """Normalise a fill spec -> (kind, payload): blur | black | color (uint8[3]) | image."""
    if isinstance(fill, str):
        f = fill.strip().lower()
        if f in ("blur", "black"):
            return f, None
        raise InputError(f"Unknown fill {fill!r}: use 'blur', 'black', an (R, G, B) colour or an image",
                         hint="Outpainting is provided by an add-on; pass its result as an image.")
    arr = np.asarray(fill)
    if arr.ndim == 1 and arr.shape[0] == 3:
        if not np.all(np.isfinite(arr.astype(np.float64))):
            raise InputError("Fill colour must be finite")
        return "color", np.clip(np.rint(arr.astype(np.float64)), 0, 255).astype(np.uint8)
    if (arr.ndim == 3 and arr.shape[2] in (1, 3, 4) and min(arr.shape[:2]) >= 1) or \
            (arr.ndim == 2 and min(arr.shape) >= 2):
        return "image", _to_rgb_u8(arr, "Fill image")  # gray / RGBA / float [0,1] accepted
    raise InputError("Fill must be 'blur', 'black', an (R, G, B) tuple or an HxWx3 image")


def _resample_equirect(img: np.ndarray, out_w: int, out_h: int, lon_range: Sequence[float]) -> np.ndarray:
    """Resize a full-latitude equirect covering ``lon_range`` to ``out_w x out_h``.

    Pure scaling in equirect space (separable); upsampling wraps longitude for full 360
    and continues over the poles, downsampling uses area averaging."""
    h, w = img.shape[:2]
    if (h, w) == (out_h, out_w):
        return img
    if w >= out_w and h >= out_h:
        return _area_resize(img, out_w, out_h).reshape((out_h, out_w) + img.shape[2:])
    lon = _range(lon_range, "lon_range")
    src = _pad_equirect(img, _PAD, _is_full_lon(lon), True)
    xs = ((np.arange(out_w, dtype=np.float64) + 0.5) * (w / out_w) - 0.5 + _PAD).astype(np.float32)
    ys = ((np.arange(out_h, dtype=np.float64) + 0.5) * (h / out_h) - 0.5 + _PAD).astype(np.float32)

    def map_fn(j0: int, j1: int, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray, None]:
        shape = (j1 - j0, i1 - i0)
        return (np.ascontiguousarray(np.broadcast_to(xs[None, i0:i1], shape)),
                np.ascontiguousarray(np.broadcast_to(ys[j0:j1, None], shape)), None)

    return _remap_tiled([src], out_h, out_w, map_fn, border=cv2.BORDER_REPLICATE)[0][0]


def _polar_smooth(img: np.ndarray, sigma: float, *, wrap: bool) -> np.ndarray:
    """Extra blur along longitude so a full-latitude equirect has a constant ANGULAR blur.

    A gaussian of ``sigma`` equirect pixels is ``1/cos(lat)`` times too narrow on the
    sphere away from the equator, and every row near a pole maps to (almost) one point:
    an equirect-space fill therefore pinches into a visible star at the zenith / nadir.
    Each row gets an additional longitude gaussian of ``sigma * sqrt(1/cos^2 - 1)`` px
    (total ``sigma / cos(lat)``), so pole rows converge to their mean.  Circular for a
    full 360 (``wrap``), mirror-symmetric otherwise.  ``img``: float32 HxWxC."""
    hs, ws = img.shape[:2]
    phi = _lat_axis(hs, -math.pi / 2, math.pi / 2, 0, hs)
    c = np.maximum(np.cos(phi), 1e-6)
    extra = sigma * np.sqrt(np.maximum(1.0 / (c * c) - 1.0, 0.0))  # per-row sigma (px)
    x = img if wrap else np.concatenate([img, img[:, ::-1]], axis=1)  # even extension = reflect
    n = x.shape[1]
    freq = np.fft.rfftfreq(n)
    gain = np.exp(-2.0 * (math.pi * extra[:, None] * freq[None, :]) ** 2)  # gaussian transfer fn
    spec = np.fft.rfft(x.astype(np.float64), axis=1) * gain[..., None]
    y = np.fft.irfft(spec, n=n, axis=1)[:, :ws]
    return y.astype(np.float32)


def _blur_surround(eqs: Sequence[np.ndarray], cov: np.ndarray, *, wrap: bool) -> np.ndarray:
    """Algorithm E 'blur' fill: push-pull from the photo edge, gaussian sigma = 2% of the
    width, darkened up to 40% with distance from the photo.  Uses the average of ``eqs``
    (zero where the photo has no coverage) and is computed at a small working resolution
    (the result is very smooth); returned full-size uint8.  The fill is smoothed along
    longitude toward the poles (:func:`_polar_smooth`) so it has no zenith/nadir pinch."""
    h, w = cov.shape
    s = min(1.0, _FILL_WORK_LONG / max(h, w))
    ws, hs = max(8, int(round(w * s))), max(4, int(round(h * s)))
    c = cv2.resize((cov > 0.0).astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA)
    acc = np.zeros((hs, ws, 3), np.float32)
    for eq in eqs:  # area means of the photo-covered pixels (others are 0)
        acc += cv2.resize(eq, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32)
    col = acc / (len(eqs) * np.maximum(c, 1e-6))[..., None]
    # longitude wrap: process three copies side by side and keep the middle one
    col_t = np.concatenate([col] * 3, axis=1) if wrap else col
    c_t = np.concatenate([c] * 3, axis=1) if wrap else c
    base = push_pull_fill(col_t, c_t) if c_t.max() > 0.0 else np.zeros_like(col_t)
    sigma = max(0.5, _FILL_SIGMA_FRAC * ws)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    outside = (c_t < 0.5).astype(np.uint8)
    if outside.any() and not outside.all():
        dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
        dmax = float((dist[:, ws:2 * ws] if wrap else dist).max())
        t = dist / dmax if dmax > 0 else np.zeros_like(dist)
        t = cv2.GaussianBlur(t, (0, 0), sigmaX=2.0 * sigma, borderType=cv2.BORDER_REFLECT_101)
        if wrap:
            t = t[:, ws:2 * ws]
        dark = 1.0 - _FILL_DARKEN * np.clip(t, 0.0, 1.0)
    else:
        dark = np.full((hs, ws), 1.0 - _FILL_DARKEN if outside.all() else 1.0, np.float32)
    if wrap:
        base = base[:, ws:2 * ws]
    shaded = _polar_smooth(base * dark[..., None], sigma, wrap=wrap)
    small = np.rint(np.clip(shaded, 0.0, 255.0)).astype(np.uint8)
    return _resample_equirect(small, w, h, (-180.0, 180.0) if wrap else (-90.0, 90.0))


def _make_surround(kind: str, payload: np.ndarray | None, eqs: Sequence[np.ndarray], cov: np.ndarray, *,
                   wrap: bool, lon_range: Sequence[float]) -> np.ndarray:
    h, w = cov.shape
    if kind == "blur":
        return _blur_surround(eqs, cov, wrap=wrap)
    if kind == "black":
        return np.zeros((h, w, 3), np.uint8)
    if kind == "color":
        assert payload is not None
        return np.broadcast_to(payload, (h, w, 3)).copy()
    assert payload is not None
    return _resample_equirect(payload, w, h, lon_range)  # already HxWx3 uint8 (_parse_fill)


def _composite(photo: np.ndarray, cov: np.ndarray, surround: np.ndarray) -> np.ndarray:
    """``photo * cov + surround * (1 - cov)`` in uint8, touching only the rim in float."""
    out = surround.copy()
    np.copyto(out, photo, where=(cov >= 1.0)[..., None])
    rim = (cov > 0.0) & (cov < 1.0)
    if rim.any():
        a = cov[rim][:, None]
        out[rim] = np.clip(np.rint(photo[rim] * a + surround[rim] * (1.0 - a)), 0, 255).astype(np.uint8)
    return out


def _to_rgb_u8(img: np.ndarray, name: str) -> np.ndarray:
    """Coerce an image to HxWx3 uint8 RGB.

    Gray (HxW / HxWx1) is replicated, RGBA drops alpha, uint16 is scaled by 1/257, bool by
    255, other integers are clipped, floats are taken as [0, 1] when their maximum is <= 1
    (else [0, 255]); NaN / inf become 0."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] not in (3, 4)) \
            or img.shape[0] < 1 or img.shape[1] < 1:
        raise InputError(f"{name} must be an HxWx3 RGB image (got shape {img.shape})")
    if img.ndim == 3:
        img = img[..., :3]
    if img.dtype == np.uint8:
        pass
    elif img.dtype == np.bool_:
        img = img.astype(np.uint8) * np.uint8(255)
    elif img.dtype == np.uint16:
        img = ((img.astype(np.uint32) + 128) // 257).astype(np.uint8)
    elif np.issubdtype(img.dtype, np.integer):
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        f = img.astype(np.float32)
        finite = np.isfinite(f)
        if not finite.all():
            f = np.where(finite, f, np.float32(0.0))
        if f.size and float(f.max()) <= 1.0:
            f = f * np.float32(255.0)
        img = np.clip(np.rint(f), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return np.ascontiguousarray(img)


def _virtual_K(K: Intrinsics, w: int, h: int, size_mult: float) -> Intrinsics:
    """Intrinsics of the photo enlarged by ``size_mult`` in field of view (<= 150 deg)."""
    _check_K(K)
    if (K.width, K.height) != (w, h):
        K = _scale_K(K, w / K.width, h / K.height, w, h)
    if not (math.isfinite(size_mult) and size_mult > 0):
        raise InputError(f"size_mult must be a positive number (got {size_mult})")
    hfov = math.degrees(2.0 * math.atan((w / 2.0) / K.fx))
    vfov = math.degrees(2.0 * math.atan((h / 2.0) / K.fy))
    big = max(hfov, vfov)
    want = big * size_mult
    limit = max(_MAX_VR180_FOV, big) if size_mult >= 1.0 else want
    if want > limit + 1e-9:
        log.warning("VR180 size x%.2f would need a %.0f deg field of view; clamped to %.0f deg",
                    size_mult, want, limit)
        want = limit
    if abs(want - big) < 1e-9:
        return K
    # scale both focal lengths so the larger FOV becomes `want` (pixel aspect is preserved)
    half = (w / 2.0) if hfov >= vfov else (h / 2.0)
    f_big = K.fx if hfov >= vfov else K.fy
    k = (half / math.tan(math.radians(want) / 2.0)) / f_big
    return Intrinsics(K.fx * k, K.fy * k, K.cx, K.cy, w, h)


def vr180_frame(left: RGB, right: RGB, K: Intrinsics, eye_px: int, *, size_mult: float = 1.0,
                fill: str | RGB | tuple[int, int, int] = "blur", layout: str = "sbs") -> RGB:
    """Build a VR180 stereo frame from two perspective eye images.

    Each eye is mapped into a 180x180 deg equirect square of ``eye_px`` pixels (longitude
    -90..90, latitude -90..90) with the photo's optical axis straight ahead.  The photo keeps
    its TRUE angular size (from ``K``, which describes the eye images); ``size_mult`` scales
    the virtual field of view (the larger of h/v FOV is capped at 150 deg, with a logged
    warning).  Outside the photo:

    * ``"blur"``: push-pull from the photo edge + gaussian (sigma = 2% of the eye width),
      darkened up to 40% toward the edges; computed once from both eyes and used
      identically in both, so it sits at infinity; the blur widens along longitude toward
      the zenith / nadir (constant angular radius) so the poles show no pinch;
    * ``"black"``; an ``(R, G, B)`` tuple; or an ``HxWx3`` image (e.g. an outpainted
      180 equirect, resampled to ``eye_px``) used for both eyes.

    ``layout`` ``"sbs"`` returns ``eye_px x 2*eye_px`` (left eye on the left); ``"tb"``
    returns ``2*eye_px x eye_px`` (left eye on top).
    """
    eye_px = _as_int(eye_px, "VR180 eye size")
    if eye_px < 8:
        raise InputError(f"VR180 eye size must be at least 8 px (got {eye_px})")
    lay = str(layout).strip().lower()
    if lay not in ("sbs", "tb"):
        raise InputError(f"Unknown VR180 layout {layout!r}: use 'sbs' or 'tb'")
    left = _to_rgb_u8(left, "Left eye image")
    right = _to_rgb_u8(right, "Right eye image")
    if left.shape != right.shape:
        raise InputError(f"Left and right eye images differ in size ({left.shape} vs {right.shape})")
    kind, payload = _parse_fill(fill)
    h, w = left.shape[:2]
    Kv = _virtual_K(K, w, h, _as_float(size_mult, "size_mult"))
    lon_range = (-90.0, 90.0)
    (eq_l, eq_r), cov = _persp_to_equirect_multi([left, right], Kv, eye_px, eye_px, np.eye(3), lon_range)
    # one surround for both eyes (from both eyes' colours) -> it sits at infinity
    surround = _make_surround(kind, payload, [eq_l, eq_r], cov, wrap=False, lon_range=lon_range)
    out_l = _composite(eq_l, cov, surround)
    del eq_l
    out_r = _composite(eq_r, cov, surround)
    del eq_r, surround
    axis = 1 if lay == "sbs" else 0
    return np.concatenate([out_l, out_r], axis=axis)


def pano_from_photo(img: RGB, K: Intrinsics, out_w: int, *, fill: str | RGB | tuple[int, int, int] = "blur"
                    ) -> tuple[RGB, np.ndarray]:
    """Mono 360 equirect (``out_w x out_w//2``) with the photo straight ahead at its true
    angular size and filled surroundings (see :func:`vr180_frame` for ``fill``; the blur
    fill wraps seamlessly around the back).  Returns ``(pano, valid)`` with ``valid`` the
    bool mask of pixels covered by the photo."""
    out_w = _as_int(out_w, "Panorama width")
    if out_w < 8:
        raise InputError(f"Panorama width must be at least 8 px (got {out_w})")
    out_h = out_w // 2
    img = _to_rgb_u8(img, "Photo")
    kind, payload = _parse_fill(fill)
    lon_range = (-180.0, 180.0)
    eq, cov = _persp_to_equirect_cov(img, K, out_w, out_h, np.eye(3), lon_range)
    surround = _make_surround(kind, payload, [eq], cov, wrap=True, lon_range=lon_range)
    out = _composite(eq, cov, surround)
    return out, cov >= 0.5
