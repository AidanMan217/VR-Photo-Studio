"""Novel-view rendering of layered depth images by fixed-point backward warping.

Algorithm D (docs/CONTRACTS.md).  For every layer the forward map ``q`` (source pixel ->
destination pixel) is computed from the layer's inverse depth and the new camera pose
(``P = unproject(K, 1/inv_z)``, ``P' = R.T (P - t)``, ``q = project(dst_K, P')``, invalid
where ``z' < 0.05``).  For each destination pixel ``u`` we solve ``q(p) = u`` for the
source position ``p`` with the fixed-point iteration ``p <- p + (u - q(p)) / s``
(``s = dst_f / src_f``), sampling ``q`` with bilinear ``cv2.remap``.  Two initialisations
are run:

* ``p0 = u`` (zero-displacement guess), and
* ``p0 = 2u - q(u)`` where ``q`` is evaluated with the *locally nearest* depth (the
  maximum inverse depth within the layer's parallax range around the first iterate).
  This seed lands on foreground surfaces that slide over the background, so occlusions
  are resolved inside a single layer.  (Evaluated with the layer's own depth the formula
  equals the first iterate of ``p0 = u`` and would add nothing.)

Both are iterated ``iters`` times; the converged solution with the larger inverse depth
(nearer surface) wins.  The final residual ``r`` (destination pixels) becomes the
convergence alpha ``clip(1.5 - r, 0, 1)``: where depth discontinuities tear the surface
open the iteration cannot converge and the layer turns transparent instead of showing
stretched "rubber" pixels.  Colour and alpha are sampled premultiplied and layers are
composited back to front.

Two robustness extensions keep the iteration contractive for any camera:

* **Rotation preconditioning** (``R != I`` only): ``u`` is replaced by the exact
  solution for points at infinity (where the destination ray hits the source image) and
  each step is multiplied by the inverse Jacobian of the rotation-only map there.
  Without it the iteration converges slowly beyond a few degrees of rotation and
  diverges beyond ~20 deg of yaw/pitch (the rotation alone stretches the periphery by
  ``sec^2``).  Stretch caused by depth discontinuities is not compensated, so tears
  still turn transparent.  For ``R = I`` the algorithm is exactly the contract's.
* **Box clamp**: iterates are clamped to one pixel around the source image, where the
  replicated forward map would otherwise be flat and stall the iteration.

Coordinates are handled as "source-equivalent" pixels: destination pixel ``u`` maps to
``a = (u - c_dst) / s + c_src``, so the destination intrinsics may have any size, focal
length or principal point (convergence shifts, overscan, VR eye rendering).

Performance (4 CPU cores, no GPU, unloaded): ~0.2 s for a 1920x1080 two-layer render
(~0.35 s with a rotated camera), a few seconds at 24 MP.  Strips of output rows are
solved in parallel on a small shared thread pool.  Memory: ~20 bytes per source pixel
for the layer being rendered (forward map, local-max depth, premultiplied RGBA) plus
bounded per-strip temporaries.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

import cv2
import numpy as np

from ..core.errors import InputError
from ..core.fill import push_pull_fill
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, Intrinsics, Layer, LayeredImage, Pose, RenderResult

log = get_logger(__name__)

#: Points closer than this to the new camera plane (metres) are treated as invalid.
MIN_Z = 0.05
#: Residual (destination pixels) below which the fixed point counts as converged.
CONVERGED_PX = 1.0
_SENTINEL = np.float32(1.0e5)  # forward-map value for invalid (behind camera) points
_MAX_STRIP_PX = 1_048_576  # destination pixels per strip (bounds temporaries, ~60 B/px)
_QMAP_STRIP_PX = 1_048_576  # source pixels per strip when building the forward map
_PTS_CHUNK = 8192  # point-list remaps are reshaped to (rows, _PTS_CHUNK)
_SHRT_MAX = 32767  # cv2.remap size limit (exclusive)
_FULLRES_DILATE_MAX = 16  # larger dilation radii are computed on a pooled grid
_PARALLEL_MIN_PX = 250_000  # smaller outputs are rendered on the calling thread

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _get_pool() -> ThreadPoolExecutor:
    """Shared worker pool for strip-parallel rendering (numpy/cv2 release the GIL)."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=_workers(), thread_name_prefix="vrphoto-warp")
        return _pool


# --------------------------------------------------------------------------- small helpers
def _check(cancel: CancelToken | None) -> None:
    if cancel is not None:
        cancel.check()


def _block_reduce(a: np.ndarray, f: int, op: str) -> np.ndarray:
    """Reduce a 2-D array over non-overlapping ``f x f`` blocks (``op`` = "max" | "min").

    The array is edge-padded to a multiple of ``f``; output is ``ceil(H/f) x ceil(W/f)``.
    """
    if f <= 1:
        return a.copy()
    h, w = a.shape[:2]
    hp, wp = -(-h // f) * f, -(-w // f) * f
    if (hp, wp) != (h, w):
        a = np.pad(a, ((0, hp - h), (0, wp - w)), mode="edge")
    fn = np.maximum if op == "max" else np.minimum
    cols = a[:, 0::f].copy()
    for i in range(1, f):
        fn(cols, a[:, i::f], out=cols)
    out = cols[0::f].copy()
    for i in range(1, f):
        fn(out, cols[i::f], out=out)
    return out


def _upsample_blocks(small: np.ndarray, f: int, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour inverse of :func:`_block_reduce`, cropped to ``shape``."""
    h, w = shape
    sh, sw = small.shape[:2]
    up = cv2.resize(small, (sw * f, sh * f), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(up[:h, :w])


def _grey_max(w: np.ndarray, radius: int) -> np.ndarray:
    """Local maximum over a ``(2r+1)^2`` square window (pooled grid for large radii)."""
    if radius <= 0:
        return w.copy()
    if radius <= _FULLRES_DILATE_MAX:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(w, ker, borderType=cv2.BORDER_REPLICATE)
    f = int(math.ceil(radius / 12.0))
    small = _block_reduce(w, f, "max")
    rs = int(math.ceil(radius / f)) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * rs + 1, 2 * rs + 1))
    small = cv2.dilate(small, ker, borderType=cv2.BORDER_REPLICATE)
    return _upsample_blocks(small, f, w.shape[:2])


def _remap(
    src: np.ndarray, pts: np.ndarray, border: int, dst: np.ndarray | None = None, value: float = 0.0
) -> np.ndarray:
    """Bilinear sample ``src`` at a float32 ``HxWx2`` map of (x, y) positions."""
    return cv2.remap(src, pts, None, cv2.INTER_LINEAR, dst=dst, borderMode=border, borderValue=value)


def _remap_pts(src: np.ndarray, pts: np.ndarray, border: int, value: float = 0.0) -> np.ndarray:
    """Bilinear sample ``src`` at an ``Nx2`` float32 point list -> ``(N,)`` or ``(N, C)``."""
    n = pts.shape[0]
    tail = src.shape[2:]
    if n == 0:
        return np.empty((0, *tail), dtype=src.dtype)
    width = min(n, _PTS_CHUNK)
    rows = -(-n // width)
    buf = np.zeros((rows * width, 2), np.float32)
    buf[:n] = pts
    out = _remap(src, buf.reshape(rows, width, 2), border, value=value)
    return out.reshape(rows * width, *tail)[:n]


def sanitize_inv_z(inv_z: np.ndarray) -> np.ndarray:
    """float32 copy-or-view of ``inv_z`` with non-finite / non-positive values made safe."""
    w = np.asarray(inv_z, dtype=np.float32)
    if not np.isfinite(w).all():
        w = np.nan_to_num(w, nan=1e-6, posinf=1e6, neginf=1e-6)
    lo, hi = float(w.min()), float(w.max())
    if lo < 1e-6 or hi > 1e6:
        w = np.clip(w, 1e-6, 1e6)
    return np.ascontiguousarray(w, dtype=np.float32)


# --------------------------------------------------------------------------- projection core
@dataclass(frozen=True)
class _Motion:
    """Pose + source intrinsics, ready for vectorised source -> new-camera projection."""

    R: np.ndarray  # 3x3 float64, columns = new camera axes in the source frame
    t: np.ndarray  # 3 float64, new camera centre in the source frame
    identity_R: bool
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def make(K: Intrinsics, pose: Pose) -> _Motion:
        R = np.asarray(pose.R, dtype=np.float64)
        t = np.asarray(pose.t, dtype=np.float64).reshape(-1)
        if R.shape != (3, 3) or t.shape != (3,):
            raise InputError("Camera pose must have a 3x3 rotation and a 3-vector translation.")
        if not (np.isfinite(R).all() and np.isfinite(t).all()):
            raise InputError("Camera pose contains non-finite values.")
        ident = bool(np.allclose(R, np.eye(3), atol=1e-12))
        return _Motion(R, t, ident, float(K.fx), float(K.fy), float(K.cx), float(K.cy))

    def project(
        self,
        xn: np.ndarray,
        yn: np.ndarray,
        w: np.ndarray,
        outx: np.ndarray | None = None,
        outy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map normalised source rays ``(xn, yn)`` with inverse depth ``w`` into the new camera.

        ``xn``/``yn`` broadcast against ``w``.  Returns target coordinates in SOURCE
        intrinsics (source-equivalent pixels); points whose new-camera depth is below
        ``MIN_Z`` get ``_SENTINEL``.  Works with the point scaled by ``w``
        (``P*w = (xn, yn, 1)``), so points at infinity (``w = 0``) are exact.
        """
        f32 = np.float32
        shape = w.shape
        if outx is None:
            outx = np.empty(shape, f32)
        if outy is None:
            outy = np.empty(shape, f32)
        tx, ty, tz = (f32(v) for v in self.t)
        if self.identity_R:
            # X = xn - tx*w, Y = yn - ty*w, Z = 1 - tz*w
            if tx:
                np.multiply(w, -tx, out=outx)
                outx += xn
            else:
                outx[...] = xn
            if ty:
                np.multiply(w, -ty, out=outy)
                outy += yn
            else:
                outy[...] = yn
            Z = (f32(1.0) - tz * w) if tz else None
        else:
            R = self.R.astype(f32)
            hx = xn - tx * w if tx else np.broadcast_to(xn, shape)
            hy = yn - ty * w if ty else np.broadcast_to(yn, shape)
            hz = f32(1.0) - tz * w if tz else None
            Z = R[0, 2] * hx + R[1, 2] * hy
            Z += R[2, 2] if hz is None else R[2, 2] * hz
            np.multiply(hx, R[0, 0], out=outx)
            outx += R[1, 0] * hy
            outx += R[2, 0] if hz is None else R[2, 0] * hz
            np.multiply(hx, R[0, 1], out=outy)
            outy += R[1, 1] * hy
            outy += R[2, 1] if hz is None else R[2, 1] * hz
        if Z is None:  # Z == 1 (scaled): valid iff the true depth 1/w >= MIN_Z
            valid = w < f32(1.0 / MIN_Z)
        else:
            valid = Z > f32(MIN_Z) * w
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                inv = np.divide(f32(1.0), Z, where=valid, out=np.ones(shape, f32))
            outx *= inv
            outy *= inv
        outx *= f32(self.fx)
        outx += f32(self.cx)
        outy *= f32(self.fy)
        outy += f32(self.cy)
        if not valid.all():
            bad = ~valid
            outx[bad] = _SENTINEL
            outy[bad] = _SENTINEL
        return outx, outy

    def project_px(self, pts: np.ndarray, w: np.ndarray) -> np.ndarray:
        """:meth:`project` for an ``Nx2`` list of source pixel positions -> ``Nx2``."""
        xn = ((pts[:, 0] - np.float32(self.cx)) / np.float32(self.fx)).astype(np.float32)
        yn = ((pts[:, 1] - np.float32(self.cy)) / np.float32(self.fy)).astype(np.float32)
        out = np.empty((pts.shape[0], 2), np.float32)
        self.project(xn, yn, np.asarray(w, np.float32), out[:, 0], out[:, 1])
        return out


@dataclass
class _PreparedLayer:
    q: np.ndarray  # HxWx2 float32 forward map (source-equivalent target coordinates)
    w: np.ndarray  # HxW float32 inverse depth
    rgba: np.ndarray  # HxWx4 uint8 premultiplied colour + alpha
    near: np.ndarray | None  # HxW float32 local max of w (None: no occlusion possible)
    par_px: float  # parallax (source px) per unit of inverse depth, for seed decisions


def _forward_map(w: np.ndarray, K: Intrinsics, mot: _Motion, cancel: CancelToken | None) -> np.ndarray:
    h, wd = w.shape
    q = np.empty((h, wd, 2), np.float32)
    xn = ((np.arange(wd, dtype=np.float64) - K.cx) / K.fx).astype(np.float32)[None, :]
    ys = ((np.arange(h, dtype=np.float64) - K.cy) / K.fy).astype(np.float32)
    rows = max(1, _QMAP_STRIP_PX // max(1, wd))
    for r0 in range(0, h, rows):
        _check(cancel)
        r1 = min(h, r0 + rows)
        qx, qy = mot.project(xn, ys[r0:r1, None], w[r0:r1])
        q[r0:r1] = cv2.merge([qx, qy])
    return q


def _premultiplied_rgba(layer: Layer) -> np.ndarray:
    """Premultiplied ``HxWx4`` uint8 colour + alpha of a layer (NaN alpha -> transparent)."""
    rgb = np.ascontiguousarray(layer.rgb)
    a = np.asarray(layer.alpha, dtype=np.float32)
    if float(a.min()) >= 1.0:  # opaque (a NaN makes min() NaN -> general path)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    a8 = np.empty(a.shape, np.float32)
    np.multiply(np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0), np.float32(255.0), out=a8)
    np.clip(a8, 0.0, 255.0, out=a8)
    a8 += np.float32(0.5)
    a8 = a8.astype(np.uint8)
    prem = cv2.multiply(rgb, cv2.merge([a8, a8, a8]), scale=1.0 / 255.0)
    return cv2.merge([prem, a8])


def _near_map(w: np.ndarray, q: np.ndarray, K: Intrinsics, mot: _Motion) -> tuple[np.ndarray | None, float]:
    """Local-max inverse depth around depth discontinuities (for the near seed).

    Occlusions (fold-overs) can only happen where the parallax changes by about a pixel
    or more between neighbouring pixels.  Returns a map holding the maximum inverse depth
    within the layer's parallax range of such discontinuities and 0 elsewhere, plus the
    parallax in source pixels per unit of inverse depth.  ``(None, 0)`` when no occlusion
    is possible.
    """
    h, wd = w.shape
    st = max(1, int(math.sqrt(h * wd / 65536.0)))
    ws = np.ascontiguousarray(w[::st, ::st])
    qs = q[::st, ::st]
    xn = ((np.arange(0, wd, st, dtype=np.float64) - K.cx) / K.fx).astype(np.float32)[None, :]
    yn = ((np.arange(0, h, st, dtype=np.float64) - K.cy) / K.fy).astype(np.float32)[:, None]
    qix, qiy = mot.project(xn, yn, np.zeros_like(ws))  # same rays at infinity
    valid = (qs[..., 0] != _SENTINEL) & (qix != _SENTINEL)
    if int(np.count_nonzero(valid)) < 2:
        return None, 0.0
    dx = (qs[..., 0] - qix)[valid]
    dy = (qs[..., 1] - qiy)[valid]
    ptp = max(float(dx.max() - dx.min()), float(dy.max() - dy.min()))
    wv = ws[valid]
    wr_s = float(wv.max() - wv.min())
    wr = float(w.max() - w.min())
    if wr_s > 1e-12:
        ptp *= max(1.0, wr / wr_s)  # the subsample may miss thin extremes
    if ptp < 1.0 or wr <= 1e-12:
        return None, 0.0
    par = ptp / wr
    grad = cv2.morphologyEx(w, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    edges = (grad > np.float32(1.0 / par)).astype(np.uint8)
    if not edges.any():
        return None, 0.0
    rho = min(int(math.ceil(ptp)) + 2, max(h, wd) // 2 + 1)
    near = _grey_max(w, rho)
    near[_grey_max(edges, rho) == 0] = 0.0
    return near, par


def _prepare_layer(layer: Layer, K: Intrinsics, mot: _Motion, cancel: CancelToken | None) -> _PreparedLayer:
    w = sanitize_inv_z(layer.inv_z)
    q = _forward_map(w, K, mot, cancel)
    near, par = _near_map(w, q, K, mot)
    return _PreparedLayer(q=q, w=w, rgba=_premultiplied_rgba(layer), near=near, par_px=par)


# --------------------------------------------------------------------------- solver
def _residual(d: np.ndarray, kx: float, ky: float) -> np.ndarray:
    """Length in destination pixels of source-equivalent error vectors ``...x2``."""
    dx = d[..., 0]
    dy = d[..., 1]
    if kx != 1.0 or ky != 1.0:
        dx = dx * np.float32(kx)
        dy = dy * np.float32(ky)
    r = dx * dx
    r += dy * dy
    return np.sqrt(r, out=r)


class _Precond:
    """Rotation-aware start point and per-pixel step matrix for one strip of targets.

    For a rotated camera the plain step ``p <- p + (u - q(p))`` converges slowly (and
    diverges beyond ~20 deg of yaw/pitch or ~60 deg of roll) because the rotation alone
    stretches the image by up to ``sec^2`` toward the periphery.  ``p0`` is the exact
    solution for points at infinity (where the destination ray hits the source image)
    and ``minv`` the inverse Jacobian of the rotation-only map at ``p0``, so the
    preconditioned iteration ``p <- p + M^-1 (u - q(p))`` started at ``p0`` only has to
    resolve parallax, exactly like the unrotated case (both reduce to ``(u, I)`` for
    ``R = I``, which is why unrotated renders skip this).  Stretch caused by depth
    discontinuities is NOT compensated, so tears still fail to converge and turn
    transparent.  Rays pointing away from the source (or far outside it) keep
    ``(u, I)``.
    """

    __slots__ = ("p0", "minv", "_t")

    def __init__(self, A: np.ndarray, mot: _Motion) -> None:
        f32 = np.float32
        R = mot.R.astype(np.float32)
        rx = A[..., 0] - f32(mot.cx)
        rx *= f32(1.0 / mot.fx)
        ry = A[..., 1] - f32(mot.cy)
        ry *= f32(1.0 / mot.fy)

        def lin(a: float, b: float, c: float) -> np.ndarray:
            out = rx * f32(a)
            out += ry * f32(b)
            out += f32(c)
            return out

        dx = lin(R[0, 0], R[0, 1], R[0, 2])
        dy = lin(R[1, 0], R[1, 1], R[1, 2])
        dz = lin(R[2, 0], R[2, 1], R[2, 2])
        ok = dz > f32(1e-2)
        inv = np.divide(f32(1.0), dz, where=ok, out=np.ones_like(dz))
        p0 = np.empty(A.shape, f32)
        dx *= inv
        dy *= inv
        np.multiply(dx, f32(mot.fx), out=p0[..., 0])
        p0[..., 0] += f32(mot.cx)
        np.multiply(dy, f32(mot.fy), out=p0[..., 1])
        p0[..., 1] += f32(mot.cy)
        # rays that hit far outside the source cannot see anything useful: keep (u, I)
        span = f32(4.0 * max(mot.cx, mot.cy, 1.0) + 64.0)
        ok &= np.abs(p0[..., 0] - f32(mot.cx)) < span
        ok &= np.abs(p0[..., 1] - f32(mot.cy)) < span
        # Jacobian of the rotation-only map at p0 (source-equivalent pixels)
        rfx = f32(mot.fx / mot.fy)
        m00 = rx * -R[0, 2]
        m00 += R[0, 0]
        m00 *= dz
        m01 = rx * -R[1, 2]
        m01 += R[1, 0]
        m01 *= dz
        m01 *= rfx
        m10 = ry * -R[0, 2]
        m10 += R[0, 1]
        m10 *= dz
        m10 /= rfx
        m11 = ry * -R[1, 2]
        m11 += R[1, 1]
        m11 *= dz
        det = m00 * m11
        det -= m01 * m10
        ok &= np.abs(det) > f32(1e-6)
        det[~ok] = 1.0
        np.reciprocal(det, out=det)
        minv = np.empty((4,) + A.shape[:-1], f32)
        np.multiply(m11, det, out=minv[0])
        np.multiply(m01, det, out=minv[1])
        np.negative(minv[1], out=minv[1])
        np.multiply(m10, det, out=minv[2])
        np.negative(minv[2], out=minv[2])
        np.multiply(m00, det, out=minv[3])
        bad = ~ok
        if bad.any():
            p0[bad] = A[bad]
            for i, v in enumerate((1.0, 0.0, 0.0, 1.0)):
                minv[i][bad] = v
        self.p0 = p0
        self.minv = minv
        self._t: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def subset(self, idx: np.ndarray) -> np.ndarray:
        """Step matrices of the flat pixel indices ``idx`` -> ``(4, N)``."""
        return self.minv.reshape(4, -1)[:, idx]

    def buffers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Three reusable float32 scratch arrays of the strip's ``HxW`` shape."""
        if self._t is None:
            shape = self.minv.shape[1:]
            self._t = (np.empty(shape, np.float32), np.empty(shape, np.float32), np.empty(shape, np.float32))
        return self._t


def _step(
    p: np.ndarray,
    e: np.ndarray,
    hi: tuple[float, float],
    minv: np.ndarray | None = None,
    bufs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """In place ``p = clamp(p + minv @ e)`` for ``...x2`` arrays (``minv`` None = identity).

    ``minv`` is planar ``(4, ...)`` = ``[m00, m01, m10, m11]``.  ``p`` is clamped to
    ``[-1, W] x [-1, H]`` (``hi = (W, H)``): beyond one pixel outside the source the
    replicated forward map is flat, so an iterate that left the image would crawl back
    at the speed of the solution's distance from the border (or run away when there is
    no solution).  Clamping keeps the fixed point fast for solutions next to the border;
    a target with no solution stays at the border with a large residual and becomes
    transparent.
    """
    if minv is None:
        p += e
    else:
        if bufs is None:
            t0, t1, t2 = (np.empty(e.shape[:-1], np.float32) for _ in range(3))
        else:
            t0, t1, t2 = bufs
        ex = e[..., 0]
        ey = e[..., 1]
        np.multiply(minv[0], ex, out=t0)
        np.multiply(minv[1], ey, out=t1)
        t0 += t1
        np.multiply(minv[2], ex, out=t1)
        np.multiply(minv[3], ey, out=t2)
        t1 += t2
        px = p[..., 0]
        py = p[..., 1]
        px += t0
        py += t1
    if p.ndim == 3:
        cv2.max(p, (-1.0, -1.0, 0.0, 0.0), dst=p)  # per-channel bounds, in place
        cv2.min(p, (hi[0], hi[1], 0.0, 0.0), dst=p)
    else:  # Nx2 point lists
        np.clip(p[:, 0], -1.0, hi[0], out=p[:, 0])
        np.clip(p[:, 1], -1.0, hi[1], out=p[:, 1])


def _solve_strip(
    lay: _PreparedLayer,
    A: np.ndarray,
    mot: _Motion,
    iters: int,
    kxy: tuple[float, float],
    pre: _Precond | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-point solve ``q(p) = A`` for one strip -> (p ``HxWx2``, residual ``HxW``).

    ``pre`` (rotated cameras) supplies the start point and step matrices; without it
    ``p0 = A`` with unit steps (algorithm D as written).
    """
    q = lay.q
    rep = cv2.BORDER_REPLICATE
    hi = (float(q.shape[1]), float(q.shape[0]))
    if pre is None:
        p0, minv, bufs = A, None, None
    else:
        p0, minv, bufs = pre.p0, pre.minv, pre.buffers()
    # init 1: p0 = u; its first iterate is 2u - q(u)
    s = _remap(q, p0, rep)
    np.subtract(A, s, out=s)
    p = p0.copy()
    _step(p, s, hi, minv, bufs)
    x1 = p.copy() if lay.near is not None else None
    for _ in range(iters - 1):
        _remap(q, p, rep, dst=s)
        np.subtract(A, s, out=s)
        _step(p, s, hi, minv, bufs)
    _remap(q, p, rep, dst=s)
    np.subtract(A, s, out=s)
    r = _residual(s, *kxy)
    if lay.near is None or x1 is None:
        return p, r

    # init 2: p0 = 2u - q_near(u), the displacement of the nearest surface around x1
    w_a = _remap(lay.w, p, rep)
    w_n = _remap(lay.near, x1, rep)
    # candidates: a nearer surface is within reach (> 0.5 px more parallax), or seed a
    # failed next to a depth edge.  Unconverged pixels with no edge nearby (outside the
    # source, behind the camera) cannot be rescued by the near seed and are skipped.
    near_edge = w_n > 0
    w_n -= w_a
    w_n *= np.float32(lay.par_px)
    cand = w_n > 0.5
    cand |= (r >= CONVERGED_PX) & near_edge
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return p, r
    pf = p.reshape(-1, 2)
    rf = r.reshape(-1)
    a_c = A.reshape(-1, 2)[idx]
    x1_c = x1.reshape(-1, 2)[idx]
    m_c = None if pre is None else pre.subset(idx)
    wn_c = _remap_pts(lay.near, x1_c, rep)
    pb = x1_c.copy()
    _step(pb, a_c - mot.project_px(x1_c, wn_c), hi, m_c)
    for _ in range(iters):
        sb = _remap_pts(q, pb, rep)
        np.subtract(a_c, sb, out=sb)
        _step(pb, sb, hi, m_c)
    sb = _remap_pts(q, pb, rep)
    np.subtract(a_c, sb, out=sb)
    rb = _residual(sb, *kxy)
    wb = _remap_pts(lay.w, pb, rep)
    ra = rf[idx]
    wa = w_a.reshape(-1)[idx]
    conv_a = ra < CONVERGED_PX
    conv_b = rb < CONVERGED_PX
    use_b = (conv_b & (~conv_a | (wb > wa))) | (~conv_a & ~conv_b & (rb < ra))
    sel = idx[use_b]
    pf[sel] = pb[use_b]
    rf[sel] = rb[use_b]
    return p, r


def _composite_strip(acc: np.ndarray | None, smp: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Premultiplied 'over' of one sampled layer strip onto the accumulator (uint8 RGBA)."""
    conv = np.subtract(np.float32(1.5), r)
    conv *= np.float32(255.0)
    np.clip(conv, 0.0, 255.0, out=conv)
    conv8 = (conv + np.float32(0.5)).astype(np.uint8)
    if int(conv8.min()) < 255:
        smp = cv2.multiply(smp, cv2.merge([conv8, conv8, conv8, conv8]), scale=1.0 / 255.0)
    if acc is None:
        return smp
    inv_a = 255 - smp[..., 3]
    under = cv2.multiply(acc, cv2.merge([inv_a, inv_a, inv_a, inv_a]), scale=1.0 / 255.0)
    return cv2.add(smp, under)


def _check_intrinsics(K: Intrinsics, what: str) -> None:
    """Finite, positive focal lengths and a usable size (cv2.remap limit), else InputError."""
    vals = (K.fx, K.fy, K.cx, K.cy)
    if not all(math.isfinite(float(v)) for v in vals) or K.fx <= 0 or K.fy <= 0:
        raise InputError(f"Invalid {what} intrinsics (focal lengths must be positive and finite): {K}.")
    if int(K.width) <= 0 or int(K.height) <= 0 or int(K.width) != K.width or int(K.height) != K.height:
        raise InputError(f"Invalid {what} size {K.width}x{K.height}.")
    if K.width >= _SHRT_MAX or K.height >= _SHRT_MAX:
        raise InputError(f"The {what} size {K.width}x{K.height} is too large; maximum side is 32766 px.")


def _validate(li: LayeredImage) -> None:
    if not isinstance(li, LayeredImage) or not li.layers:
        raise InputError("Nothing to render: the layered image has no layers.")
    K = li.K
    _check_intrinsics(K, "camera")
    for i, lay in enumerate(li.layers):
        rgb = np.asarray(lay.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise InputError(f"Layer {i}: colour must be an HxWx3 uint8 RGB image.")
        if rgb.shape[:2] != (K.height, K.width):
            raise InputError(
                f"Layer {i}: image size {rgb.shape[1]}x{rgb.shape[0]} does not match the "
                f"camera intrinsics {K.width}x{K.height}."
            )
        if np.shape(lay.inv_z) != (K.height, K.width) or np.shape(lay.alpha) != (K.height, K.width):
            raise InputError(f"Layer {i}: depth and alpha maps must match the image size.")


def render_view(
    li: LayeredImage,
    pose: Pose,
    dst_K: Intrinsics | None = None,
    *,
    iters: int = 6,
    cancel: CancelToken | None = None,
) -> RenderResult:
    """Render ``li`` as seen by a camera at ``pose`` with intrinsics ``dst_K``.

    ``dst_K`` defaults to ``li.K`` and may differ in size, focal length and principal
    point.  Returns unpremultiplied colour (black where nothing is visible) and coverage
    alpha in [0, 1]; ``alpha < 0.5`` marks holes (disocclusions, outside the source).
    ``cancel`` (additive extension of the contract) is checked between layers/strips.
    """
    _validate(li)
    try:
        iters = int(iters)
    except (TypeError, ValueError) as e:
        raise InputError(f"iters must be an integer >= 1 (got {iters!r}).") from e
    if iters < 1:
        raise InputError("iters must be >= 1.")
    K = li.K
    D = dst_K if dst_K is not None else K
    _check_intrinsics(D, "output camera")
    mot = _Motion.make(K, pose)

    hd, wdst = int(D.height), int(D.width)
    sx, sy = K.fx / D.fx, K.fy / D.fy  # source px per destination px
    kxy = (1.0 / sx, 1.0 / sy)
    ax = ((np.arange(wdst, dtype=np.float64) - D.cx) * sx + K.cx).astype(np.float32)
    ay = ((np.arange(hd, dtype=np.float64) - D.cy) * sy + K.cy).astype(np.float32)
    n_par = _workers() if hd * wdst >= _PARALLEL_MIN_PX else 1
    rows = max(1, min(_MAX_STRIP_PX // wdst, -(-hd // n_par)))
    strips = [(r0, min(hd, r0 + rows)) for r0 in range(0, hd, rows)]
    acc: list[np.ndarray | None] = [None] * len(strips)

    for layer in li.layers:
        _check(cancel)
        lay = _prepare_layer(layer, K, mot, cancel)

        def work(si: int, lay: _PreparedLayer = lay) -> None:
            _check(cancel)
            r0, r1 = strips[si]
            A = np.empty((r1 - r0, wdst, 2), np.float32)
            A[..., 0] = ax[None, :]
            A[..., 1] = ay[r0:r1, None]
            pre = None if mot.identity_R else _Precond(A, mot)
            p, r = _solve_strip(lay, A, mot, iters, kxy, pre)
            smp = _remap(lay.rgba, p, cv2.BORDER_CONSTANT)
            acc[si] = _composite_strip(acc[si], smp, r)

        if n_par > 1 and len(strips) > 1:
            futs = [_get_pool().submit(work, si) for si in range(len(strips))]
            try:
                for fut in futs:
                    fut.result()
            except BaseException:
                for fut in futs:  # do not leave strips running after a cancel/failure
                    fut.cancel()
                wait(futs)
                raise
        else:
            for si in range(len(strips)):
                work(si)
        del lay

    full = acc[0] if len(acc) == 1 else np.concatenate(acc, axis=0)  # type: ignore[arg-type]
    assert full is not None
    a8 = np.ascontiguousarray(full[..., 3])
    prem = cv2.cvtColor(full, cv2.COLOR_RGBA2RGB)
    if int(a8.min()) == 255:
        rgb = prem
    else:
        rgb = cv2.divide(prem, cv2.merge([a8, a8, a8]), scale=255.0)  # x/0 -> 0
    alpha = a8.astype(np.float32)
    alpha *= np.float32(1.0 / 255.0)
    return RenderResult(rgb=rgb, alpha=alpha)


# --------------------------------------------------------------------------- hole filling
_PUSH_PULL_MAX_SIDE = 1024  # fill is smooth: ~7x faster than full 1080p, ~56 dB vs exact


def push_pull_fill_fast(
    img: np.ndarray, known: np.ndarray, max_side: int = _PUSH_PULL_MAX_SIDE
) -> np.ndarray:
    """``core.fill.push_pull_fill`` (``HxW`` or ``HxWxC``) evaluated at <= ``max_side`` px.

    The fill of unknown pixels is smooth by construction, so for large images the
    pyramid is built from a weight-aware area-downsampled copy and the result is
    upsampled; known pixels are returned unchanged.  Images whose long side is already
    <= ``max_side`` use ``push_pull_fill`` directly.  Returns the dtype of ``img``.
    """
    known = np.asarray(known, dtype=bool)
    h, w = known.shape
    f = int(math.ceil(max(h, w) / float(max_side)))
    if f <= 1:
        return push_pull_fill(img, known)
    holes = ~known
    if not holes.any():
        return img.copy()
    sw, sh = max(1, -(-w // f)), max(1, -(-h // f))
    masked = img.copy()
    masked[holes] = 0
    num = cv2.resize(masked, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    den = cv2.resize(known.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    safe = np.maximum(den, np.float32(1.0 / 255.0))
    low = num / (safe[..., None] if num.ndim == 3 else safe)
    low = push_pull_fill(low, den)
    # sample the low-res fill only at the hole pixels (cv2.resize-compatible centres)
    ys, xs = np.nonzero(holes)
    pts = np.empty((ys.size, 2), np.float32)
    pts[:, 0] = (xs.astype(np.float32) + 0.5) * np.float32(sw / w) - 0.5
    pts[:, 1] = (ys.astype(np.float32) + 0.5) * np.float32(sh / h) - 0.5
    vals = _remap_pts(np.ascontiguousarray(low, dtype=np.float32), pts, cv2.BORDER_REPLICATE)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        vals = np.clip(np.rint(vals), info.min, info.max)
    out = masked
    out[ys, xs] = vals.astype(img.dtype)
    return out


def hole_fraction(rr: RenderResult) -> float:
    """Fraction of pixels that are holes (alpha < 0.5) in a render."""
    return float(np.count_nonzero(rr.alpha < 0.5)) / max(1, rr.alpha.size)


def fill_holes(rr: RenderResult, *, small_frac: float = 0.001) -> RGB:
    """Fill the holes (alpha < 0.5) of a render and return an RGB image.

    A small total hole area (< ``small_frac`` of the image) is filled with OpenCV Telea
    inpainting, larger areas with push-pull filling (``core.fill.push_pull_fill``).  Hole
    pixels with partial coverage are blended ``alpha*rgb + (1-alpha)*fill`` so tear edges
    stay soft.
    """
    rgb = np.asarray(rr.rgb)
    alpha = np.asarray(rr.alpha, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or alpha.shape != rgb.shape[:2]:
        raise InputError("Render colour must be HxWx3 and alpha HxW of the same size.")
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    holes = alpha < 0.5
    n = int(np.count_nonzero(holes))
    if n == 0:
        return rgb.copy()
    if n == holes.size:
        log.warning("Render has no visible pixels; returning a black image")
        return np.zeros_like(rgb)
    if n < small_frac * holes.size:
        filled = cv2.inpaint(np.ascontiguousarray(rgb), holes.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    else:
        filled = push_pull_fill_fast(rgb, ~holes)
    a = alpha[holes][:, None]
    blend = a * rgb[holes].astype(np.float32) + (1.0 - a) * filled[holes].astype(np.float32)
    filled[holes] = np.clip(np.rint(blend), 0, 255).astype(np.uint8)
    return filled


def render_filled(
    li: LayeredImage,
    pose: Pose,
    dst_K: Intrinsics | None = None,
    *,
    iters: int = 6,
    cancel: CancelToken | None = None,
) -> RGB:
    """:func:`render_view` followed by :func:`fill_holes`."""
    return fill_holes(render_view(li, pose, dst_K, iters=iters, cancel=cancel))


__all__ = [
    "render_view",
    "fill_holes",
    "render_filled",
    "hole_fraction",
    "push_pull_fill_fast",
    "sanitize_inv_z",
    "MIN_Z",
    "CONVERGED_PX",
]
