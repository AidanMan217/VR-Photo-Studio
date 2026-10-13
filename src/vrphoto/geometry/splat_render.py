"""CPU Gaussian-splat renderer (numpy only) and the SHARP 360 turntable preview.

Design: docs/design/SHARP360.md 0.2 item 6 and 3.13.  The renderer is off SHARP 360's
critical path: it draws the turntable preview, later the Brush anchor views (M5) and an
optional yaw check.  It follows the 3D Gaussian Splatting rasteriser (EWA splatting), but
vectorised over (Gaussian, pixel) pairs instead of GPU tiles:

1. **Camera** (:class:`SplatCamera`) - ``P_cam = R @ (P - center)`` with OpenCV axes (x
   right, y down, z forward).  Pixel ``(row i, col j)`` is sampled at its centre ``u = j +
   0.5``, ``v = i + 0.5`` (the SHARP 360 / COLMAP convention of design 3.0, so ``cx = W/2``
   is the middle of the image; ``geometry.camera`` puts centres at integers instead).
   Perspective: ``u = fx*x/z + cx``; orthographic: ``u = x/m + cx`` (``m`` metres per
   pixel).  Gaussians with ``z <= Z_NEAR`` or opacity below 1/255 are culled.
2. **2D covariance** - ``Sigma' = J R Sigma R^T J^T + 0.3 I`` with ``Sigma = Rq S^2 Rq^T``
   (as ``splat.splat_covariances``; its six entries are computed once per cloud by
   :func:`prepare_splats`, which several renders of one cloud can share), the perspective
   Jacobian ``J = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]]`` (``x/z``, ``y/z`` clamped
   to 1.3x the half field of view, as 3DGS does, so Gaussians far outside the frustum do
   not explode) or ``J = diag(1/m)``.  The 0.3 px^2 low-pass keeps sub-pixel Gaussians
   visible.  Colour is the DC colour plus the view-dependent SH bands when the cloud has
   them, clipped to [0, 1].
3. **Pairs** - a Gaussian's footprint is the square of radius ``r = ceil(3 sqrt(lambda_max))``
   (clamped to ``max_radius_px``) around its centre; a pixel takes part iff its centre is
   in that square and ``alpha = min(0.99, o exp(-q/2)) >= 1/255`` with ``q = d^T
   Sigma'^-1 d``.  Instead of enumerating whole squares, each Gaussian is cut into pixel
   rows and every row into the column span where ``q <= 2 ln(255 o)`` (a closed-form
   quadratic, widened by 1e-6 px and then re-tested exactly), using two ragged
   ``np.repeat`` expansions - the same pairs as square buckets by radius with fewer
   candidates (about 2x fewer for pixel-sized Gaussians, far fewer for thin ones) and no
   per-radius loop.
4. **Compositing** - Gaussians are ranked by camera depth once; the pairs are sorted once
   on the int64 key ``pixel * n + rank``.  Front-to-back transmittance is an exclusive
   cumulative sum of ``log(1 - alpha)`` restarted at every pixel; the logs are quantised
   to 2^-32 and summed in int64, so the sums are exact.  As in 3DGS a pixel stops taking
   Gaussians before its transmittance would fall below ``T_MIN = 1e-4`` (the pairs it
   takes are a prefix of its depth order).  Colour, alpha and depth are accumulated with
   ``np.bincount``; the median depth is the z of the pair where the accumulated alpha
   first reaches 0.5 (as SHARP 360 defines the front depth).
5. **Bands and runs** - the image is processed in bands of rows sized from a per-row
   estimate of the candidate pairs (``_PAIR_BUDGET``), and a band's Gaussians in
   depth-ordered runs (``_RUN_PAIRS``) composited one after another into running
   per-pixel sums and transmittance.  Pixels that are done skip the later runs: their
   pairs are dropped before the sort, and Gaussians whose whole footprint is done are not
   expanded at all (a summed-area table), so hidden layers cost little.  The set of
   composited pairs and each pixel's summation order do not depend on the split, so an
   image is bit-identical whatever ``band_rows`` is.  Peak memory is about 250 MB; 300k
   Gaussians at 512 px take a few seconds.  ``cancel`` is checked per band and run.

:func:`orbit_camera` builds the cameras of design 3.0's azimuth contract (``phi = 0`` looks
at the subject's front, 90 = "you walked to your right", 180 = behind) and
:func:`turntable_sheet` renders ``n`` orthographic views of an upright glTF-frame cloud
around its vertical axis into one contact-sheet image.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..core.errors import InputError
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter, SplatCloud
from .mesh import _as_reporter, _check
from .splat import _sh_basis, _validate, quat_to_rotmat

log = get_logger(__name__)

#: Camera models of :class:`SplatCamera`.
CAMERA_KINDS = ("persp", "ortho")
#: Screen-space low-pass added to every 2D covariance (px^2, as in 3DGS).
LOWPASS_PX2 = 0.3
#: Largest alpha of one Gaussian at one pixel (3DGS), so nothing is fully opaque alone.
ALPHA_MAX = 0.99
#: Pairs with a smaller alpha are dropped (3DGS).
ALPHA_MIN = 1.0 / 255.0
#: ``SplatRender.depth`` is NaN where the accumulated alpha is below this.
DEPTH_MIN_ALPHA = 1e-3
#: Gaussians this close to (or behind) the camera plane are culled (metres).
Z_NEAR = 0.01
#: Default footprint radius cap (px).
MAX_RADIUS_PX = 64

_FRUSTUM_SLACK = 0.15  # Jacobian clamp: 15 % of the field of view beyond each side (3DGS's 1.3x)
_PAIR_BUDGET = 4_000_000  # candidate pairs per band (sets the band height)
_RUN_PAIRS = 250_000  # candidate pairs per depth-ordered run: small runs skip more hidden work
_MAX_RADIUS_LIMIT = 1024  # one Gaussian then has at most ~4M candidate pairs
_PREP_CHUNK = 262_144  # Gaussians per block of the projection step
_LOG_Q = float(2**32)  # fixed-point scale of log(1 - alpha): exact, order-free sums
_LOG_HALF = int(round(math.log(0.5) * _LOG_Q))  # accumulated alpha 0.5 in that scale
_LOG_ALPHA_MIN = math.log(ALPHA_MIN)
#: A pixel stops taking Gaussians before its transmittance would drop below this (3DGS).
T_MIN = 1e-4
_LOG_T_MIN = int(round(math.log(T_MIN) * _LOG_Q))
_EPS_PX = 1e-6  # candidate spans are widened by this, then every pair is tested exactly
_SH_MAX = 15  # SH coefficients beyond degree 3 are ignored
_COV_IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))  # PreparedSplats.cov rows
_SHEET_MARGIN = 0.05  # turntable: empty border around the robust box (fraction of the span)
_SHEET_OPACITY = 0.2  # turntable: Gaussians at least this opaque define the box ...
_SHEET_PCT = 0.5  # ... from their 0.5 / 99.5 percentiles per axis


# --------------------------------------------------------------------------- cameras
@dataclass(eq=False)
class SplatCamera:
    """A camera for :func:`render_splats`.

    ``R`` is the world-to-camera rotation: its rows are the camera's x (right), y (down) and
    z (forward) axes in world coordinates (OpenCV axes); ``center`` is the camera centre in
    world coordinates, so ``P_cam = R @ (P - center)``.  ``kind="persp"`` uses ``fx, fy``
    (pixels); ``kind="ortho"`` uses ``m`` (metres per pixel).  ``cx, cy`` (default: the
    image centre ``width/2, height/2``) are the image position of the optical axis, in the
    continuous coordinates where pixel ``(i, j)`` is centred at ``(j + 0.5, i + 0.5)``.
    Values are validated and normalised (float64 arrays, Python numbers) on construction;
    a bad camera raises :class:`InputError`."""

    R: np.ndarray
    center: np.ndarray
    width: int
    height: int
    kind: str = "persp"
    fx: float = 0.0
    fy: float = 0.0
    m: float = 0.0
    cx: float | None = None
    cy: float | None = None

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=np.float64)
        if R.shape != (3, 3) or not np.isfinite(R).all():
            raise InputError(f"The camera rotation must be a finite 3x3 matrix (got shape {R.shape}).")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
            raise InputError("The camera rotation must be orthonormal (its rows are the camera axes).")
        c = np.asarray(self.center, dtype=np.float64).reshape(-1)
        if c.shape != (3,) or not np.isfinite(c).all():
            raise InputError(f"The camera centre must be 3 finite numbers (got {np.shape(self.center)}).")
        try:
            w, h = int(self.width), int(self.height)
        except (TypeError, ValueError) as e:
            raise InputError(f"The image size must be whole numbers ({e}).") from e
        if w < 1 or h < 1 or w != self.width or h != self.height:
            raise InputError(f"The image size must be at least 1x1 pixel (got {self.width}x{self.height}).")
        if self.kind not in CAMERA_KINDS:
            raise InputError(f"Unknown camera kind {self.kind!r}; use one of {', '.join(CAMERA_KINDS)}.")
        fx, fy, m = _num(self.fx, "fx"), _num(self.fy, "fy"), _num(self.m, "m")
        if self.kind == "persp" and not (fx > 0 and fy > 0):
            raise InputError(
                f"A perspective camera needs positive focal lengths fx, fy in pixels (got {fx}, {fy})."
            )
        if self.kind == "ortho" and not m > 0:
            raise InputError(f"An orthographic camera needs a positive m, metres per pixel (got {m}).")
        cx = w / 2.0 if self.cx is None else _num(self.cx, "cx")
        cy = h / 2.0 if self.cy is None else _num(self.cy, "cy")
        self.R, self.center, self.width, self.height = R, c, w, h
        self.fx, self.fy, self.m, self.cx, self.cy = fx, fy, m, cx, cy

    @classmethod
    def pinhole(
        cls,
        f: float,
        width: int,
        height: int,
        *,
        R: np.ndarray | None = None,
        center: Sequence[float] | np.ndarray | None = None,
    ) -> SplatCamera:
        """Perspective camera with ``fx = fy = f`` and the optical axis through the image
        centre; by default at the world origin looking down +z (SHARP's photo camera in its
        frame C is ``SplatCamera.pinhole(cam.f, cam.width, cam.height)``)."""
        return cls(
            R=np.eye(3) if R is None else R,
            center=np.zeros(3) if center is None else center,
            width=width,
            height=height,
            kind="persp",
            fx=f,
            fy=f,
        )

    @property
    def forward(self) -> np.ndarray:
        """Viewing direction (camera +z) in world coordinates."""
        return self.R[2].copy()

    def to_camera(self, P: np.ndarray) -> np.ndarray:
        """``...x3`` world points in the camera frame (``R @ (P - center)``)."""
        return (np.asarray(P, dtype=np.float64) - self.center) @ self.R.T

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continuous pixel coordinates ``(u, v)`` and camera depth ``z`` of ``...x3`` world
        points (perspective: NaN ``u, v`` where ``z <= 0``)."""
        q = self.to_camera(P)
        x, y, z = q[..., 0], q[..., 1], q[..., 2]
        if self.kind == "ortho":
            return x / self.m + self.cx, y / self.m + self.cy, z
        with np.errstate(divide="ignore", invalid="ignore"):
            zs = np.where(z > 0, z, np.nan)
            return self.fx * x / zs + self.cx, self.fy * y / zs + self.cy, z


def _num(x: Any, what: str) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError) as e:
        raise InputError(f"The camera's {what} must be a number (got {x!r}).") from e
    if not math.isfinite(f):
        raise InputError(f"The camera's {what} must be finite (got {x!r}).")
    return f


def _unit(v: Any, what: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(a)) if a.shape == (3,) else 0.0
    if not (math.isfinite(n) and n > 1e-12):
        raise InputError(f"{what} must be a non-zero 3-vector (got {v!r}).")
    return a / n


def orbit_camera(
    target: Sequence[float] | np.ndarray,
    phi_deg: float,
    *,
    up: Sequence[float] | np.ndarray = (0.0, 1.0, 0.0),
    front: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
    distance: float,
    width: int,
    height: int,
    kind: str = "ortho",
    m: float | None = None,
    fov_deg: float | None = None,
    elevation_deg: float = 0.0,
) -> SplatCamera:
    """A camera on a circle around the vertical axis through ``target``, looking at it.

    Azimuth contract (design 3.0): ``phi = 0`` looks at the subject's front, 90 = "you
    walked to your right" (the camera is on the subject's left), 180 = behind (a mirror
    image of the front).  The defaults suit an upright glTF-frame object (``up = +y``, the
    subject faces ``front = +z``); for SHARP's frame C pass ``up = canon.up_C`` and
    ``front = -canon.R[:, 2]`` (the subject faces the photo camera).  ``elevation_deg > 0``
    raises the camera and tilts it down.  The camera centre is ``distance`` metres from
    ``target``.

    ``kind="ortho"`` needs ``m`` (metres per pixel; ``target`` maps to the image centre);
    ``kind="persp"`` needs ``fov_deg`` (horizontal field of view over ``width``, principal
    point at the image centre).  Image axes: ``right = cos(phi) r0 - sin(phi) f0`` with
    ``r0 = up x f0`` (``f0`` = ``front`` made horizontal), ``forward`` toward ``target``,
    ``down = forward x right``."""
    u = _unit(up, "up")
    f0 = _unit(front, "front")
    f0 = f0 - float(f0 @ u) * u
    if float(np.linalg.norm(f0)) < 1e-6:
        raise InputError("The subject's front direction must not be parallel to 'up'.")
    f0 /= np.linalg.norm(f0)
    r0 = np.cross(u, f0)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise InputError(f"The orbit target must be 3 finite numbers (got {target!r}).")
    dist, elev = _num(distance, "orbit distance"), _num(elevation_deg, "elevation")
    if dist <= 0:
        raise InputError(f"The orbit distance must be positive (got {distance}).")
    if abs(elev) >= 89.0:
        raise InputError(f"The elevation must be between -89 and 89 degrees (got {elevation_deg}).")
    p, e = math.radians(_num(phi_deg, "azimuth")), math.radians(elev)
    out = math.cos(e) * (math.cos(p) * f0 + math.sin(p) * r0) + math.sin(e) * u  # target -> camera
    fwd = -out
    right = math.cos(p) * r0 - math.sin(p) * f0
    down = np.cross(fwd, right)
    R = np.stack([right, down / np.linalg.norm(down), fwd])
    center = t + dist * out
    if kind == "ortho":
        if m is None:
            raise InputError("An orthographic orbit camera needs m (metres per pixel).")
        return SplatCamera(R=R, center=center, width=width, height=height, kind="ortho", m=m)
    if kind == "persp":
        fov = None if fov_deg is None else _num(fov_deg, "field of view")
        if fov is None or not 0.0 < fov < 179.0:
            raise InputError(f"A perspective orbit camera needs fov_deg between 0 and 179 (got {fov_deg}).")
        f = (_num(width, "width") / 2.0) / math.tan(math.radians(fov) / 2.0)
        return SplatCamera(R=R, center=center, width=width, height=height, kind="persp", fx=f, fy=f)
    raise InputError(f"Unknown camera kind {kind!r}; use one of {', '.join(CAMERA_KINDS)}.")


# --------------------------------------------------------------------------- rendering
@dataclass
class SplatRender:
    """Output of :func:`render_splats` (all ``height x width``):

    * ``rgb`` uint8 RGB, composited over the background;
    * ``alpha`` float32 accumulated opacity ``1 - prod(1 - alpha_i)`` (at most
      ``1 - T_MIN``);
    * ``depth`` float32 alpha-weighted mean camera z (metres), NaN where alpha < 1e-3;
    * ``depth_median`` float32 z of the Gaussian at which the accumulated alpha first
      reaches 0.5 (front to back), NaN where it never does;
    * ``diag`` - counts and timing for logs."""

    rgb: np.ndarray
    alpha: np.ndarray
    depth: np.ndarray
    depth_median: np.ndarray
    diag: dict = field(default_factory=dict)


@dataclass(eq=False)
class PreparedSplats:
    """A cloud reduced to what rendering needs, independent of the camera (see
    :func:`prepare_splats`): render several views of one cloud without redoing it.  Rows
    that cannot show (non-finite position, scale or orientation, opacity below 1/255) are
    already dropped."""

    means: np.ndarray  # (n, 3) float64
    cov: np.ndarray  # (6, n) float64: Sigma's xx, yy, zz, xy, xz, yz (splat.splat_covariances)
    opacity: np.ndarray  # (n,) float64 in [1/255, 1]
    colors: np.ndarray  # (n, 3) float64 DC colour
    sh_rest: np.ndarray | None  # (n, k, 3) higher SH bands (k <= 15) or None
    n_input: int = 0  # rows of the original cloud

    def __len__(self) -> int:
        return int(self.means.shape[0])


def prepare_splats(sc: SplatCloud, *, cancel: CancelToken | None = None) -> PreparedSplats:
    """The camera-independent part of :func:`render_splats` (the 3D covariances ``Sigma =
    R S^2 R^T`` and validity), done once for several renders of ``sc`` (``render_splats``
    accepts the result in place of the cloud)."""
    n = _validate(sc)
    means = np.asarray(sc.means, dtype=np.float64)
    scales = np.abs(np.asarray(sc.scales, dtype=np.float64))
    quats = np.asarray(sc.quats, dtype=np.float64)
    op = np.asarray(sc.opacity, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        keep = (
            np.isfinite(means).all(axis=1)
            & np.isfinite(scales).all(axis=1)
            & np.isfinite(quats).all(axis=1)
            & (np.linalg.norm(quats, axis=1) > 1e-12)
            & (op >= ALPHA_MIN)
        )
    idx = np.flatnonzero(keep)
    cov = np.empty((6, len(idx)))
    for s in range(0, len(idx), _PREP_CHUNK):
        _check(cancel)
        blk = idx[s : s + _PREP_CHUNK]
        r = quat_to_rotmat(quats[blk])
        s2 = scales[blk] ** 2
        for e, (i, j) in enumerate(_COV_IJ):  # Sigma_ij = sum_k R_ik R_jk s_k^2
            cov[e, s : s + len(blk)] = (r[:, i, :] * r[:, j, :] * s2).sum(axis=1)
    sh = None
    if sc.sh_rest is not None and np.shape(sc.sh_rest)[1] > 0:
        k = min(int(np.shape(sc.sh_rest)[1]), _SH_MAX)
        sh = np.asarray(sc.sh_rest)[idx, :k]  # kept in its dtype (float32: degree 3 is 45 floats)
    return PreparedSplats(
        means=means[idx],
        cov=cov,
        opacity=np.minimum(op[idx], 1.0),
        colors=np.asarray(sc.colors, dtype=np.float64)[idx],
        sh_rest=sh,
        n_input=n,
    )


@dataclass
class _Prep:
    """Projected, visible Gaussians sorted near to far (index = depth rank)."""

    u: np.ndarray  # image position (continuous px)
    v: np.ndarray
    qa: np.ndarray  # conic (inverse 2D covariance) [[qa, qb], [qb, qc]]
    qb: np.ndarray
    qc: np.ndarray
    inv_det: np.ndarray  # qa*qc - qb^2 = 1 / det(Sigma')
    log_o: np.ndarray  # log opacity
    t_cut: np.ndarray  # q <= t_cut  <=>  o exp(-q/2) >= 1/255
    i_lo: np.ndarray  # candidate rows: footprint square and cut-off ellipse, in the image
    i_hi: np.ndarray
    j_lo: np.ndarray  # candidate columns (likewise; for the pair estimates)
    j_hi: np.ndarray
    sj_lo: np.ndarray  # columns of the footprint square, in the image
    sj_hi: np.ndarray
    cz: np.ndarray  # (4, n): colour r, g, b in [0, 1] and camera depth z

    def __len__(self) -> int:
        return int(self.u.shape[0])


def _background(background: Any) -> np.ndarray:
    try:
        bg = np.asarray(background, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as e:
        raise InputError(f"The background must be an RGB triple 0-255 ({e}).") from e
    if bg.shape != (3,) or not np.isfinite(bg).all() or bg.min() < 0 or bg.max() > 255:
        raise InputError(f"The background must be an RGB triple of numbers 0-255 (got {background!r}).")
    return bg / 255.0


def _span(c: np.ndarray, r: np.ndarray, ext: np.ndarray, n: int) -> tuple[np.ndarray, ...]:
    """Pixel index ranges along one image axis for centres ``c``: the footprint square
    ``[ceil(c - r - 0.5), floor(c + r - 0.5)]`` (pixel centres within ``r``) and its
    intersection with the cut-off ellipse's extent ``ext`` (widened by ``_EPS_PX``), both
    clipped to ``[-1, n]`` (in float first: far-off Gaussians) -> four int64 arrays."""
    sq_lo = np.clip(np.ceil(c - r - 0.5), -1, n)
    sq_hi = np.clip(np.floor(c + r - 0.5), -1, n)
    lo = np.maximum(sq_lo, np.clip(np.ceil(c - ext - 0.5 - _EPS_PX), -1, n))
    hi = np.minimum(sq_hi, np.clip(np.floor(c + ext - 0.5 + _EPS_PX), -1, n))
    return tuple(a.astype(np.int64) for a in (sq_lo, sq_hi, lo, hi))


def _project_block(
    src: PreparedSplats, idx: np.ndarray, d: np.ndarray, cam: SplatCamera, max_r: int
) -> dict[str, np.ndarray]:
    """2D footprint of the Gaussians ``idx`` (``d`` = their means minus the camera centre)."""
    R = cam.R
    # explicit 3-term sums: the result of every Gaussian is independent of the block size
    tx = d[:, 0] * R[0, 0] + d[:, 1] * R[0, 1] + d[:, 2] * R[0, 2]
    ty = d[:, 0] * R[1, 0] + d[:, 1] * R[1, 1] + d[:, 2] * R[1, 2]
    tz = d[:, 0] * R[2, 0] + d[:, 1] * R[2, 1] + d[:, 2] * R[2, 2]
    # Sigma' = T Sigma T^T + 0.3 I with the 2x3 T = J R (rows t0, t1)
    if cam.kind == "persp":
        u = cam.fx * tx / tz + cam.cx
        v = cam.fy * ty / tz + cam.cy
        # 3DGS clamps the Jacobian's x/z, y/z just outside the frustum (not the mean)
        lx0, lx1 = -cam.cx / cam.fx, (cam.width - cam.cx) / cam.fx
        ly0, ly1 = -cam.cy / cam.fy, (cam.height - cam.cy) / cam.fy
        sx, sy = _FRUSTUM_SLACK * (lx1 - lx0), _FRUSTUM_SLACK * (ly1 - ly0)
        xz = np.clip(tx / tz, lx0 - sx, lx1 + sx)
        yz = np.clip(ty / tz, ly0 - sy, ly1 + sy)
        t0 = [(cam.fx / tz) * (R[0, k] - xz * R[2, k]) for k in range(3)]
        t1 = [(cam.fy / tz) * (R[1, k] - yz * R[2, k]) for k in range(3)]
    else:
        u = tx / cam.m + cam.cx
        v = ty / cam.m + cam.cy
        t0 = [R[0, k] / cam.m for k in range(3)]
        t1 = [R[1, k] / cam.m for k in range(3)]
    sxx, syy, szz, sxy, sxz, syz = src.cov[:, idx]

    def quad(a: list, b: list) -> np.ndarray:  # a^T Sigma b, Sigma symmetric
        return (
            a[0] * b[0] * sxx
            + a[1] * b[1] * syy
            + a[2] * b[2] * szz
            + (a[0] * b[1] + a[1] * b[0]) * sxy
            + (a[0] * b[2] + a[2] * b[0]) * sxz
            + (a[1] * b[2] + a[2] * b[1]) * syz
        )

    sa = quad(t0, t0) + LOWPASS_PX2
    sb = quad(t0, t1)
    sc_ = quad(t1, t1) + LOWPASS_PX2
    det = sa * sc_ - sb * sb
    op = src.opacity[idx]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(det) & (det > 0)
        u, v = np.where(ok, u, -1e9), np.where(ok, v, -1e9)  # rejected below; keeps the casts quiet
        lam = 0.5 * (sa + sc_) + np.sqrt(np.maximum(0.25 * (sa - sc_) ** 2 + sb * sb, 0.0))
        radius = np.where(ok, np.minimum(np.ceil(3.0 * np.sqrt(lam)), float(max_r)), 0.0)
        t_cut = 2.0 * np.log(255.0 * op)
        ex = np.where(ok, np.sqrt(t_cut * sa), 0.0)
        ey = np.where(ok, np.sqrt(t_cut * sc_), 0.0)
    _, _, i_lo, i_hi = _span(v, radius, ey, cam.height)
    sj_lo, sj_hi, j_lo, j_hi = _span(u, radius, ex, cam.width)
    ok &= (i_hi >= 0) & (i_lo <= cam.height - 1) & (j_hi >= 0) & (j_lo <= cam.width - 1)
    ok &= (i_lo <= i_hi) & (j_lo <= j_hi)

    col = src.colors[idx]
    if src.sh_rest is not None:
        if cam.kind == "ortho":
            dirs = np.broadcast_to(cam.R[2], d.shape)
        else:
            dirs = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        rest = src.sh_rest[idx]
        col = col + np.einsum("nk,nkc->nc", _sh_basis(np.ascontiguousarray(dirs), rest.shape[1]), rest)
    col = np.clip(np.nan_to_num(col, nan=0.0), 0.0, 1.0)

    det_s = np.where(ok, det, 1.0)
    out = {
        "u": u,
        "v": v,
        "qa": sc_ / det_s,
        "qb": -sb / det_s,
        "qc": sa / det_s,
        "inv_det": 1.0 / det_s,
        "log_o": np.log(op),
        "t_cut": t_cut,
        "i_lo": np.maximum(i_lo, 0),
        "i_hi": np.minimum(i_hi, cam.height - 1),
        "j_lo": np.maximum(j_lo, 0),
        "j_hi": np.minimum(j_hi, cam.width - 1),
        "sj_lo": np.maximum(sj_lo, 0),
        "sj_hi": np.minimum(sj_hi, cam.width - 1),
        "cz": np.stack([col[:, 0], col[:, 1], col[:, 2], tz]),
    }
    return {k_: v_[..., ok] for k_, v_ in out.items()}


def _prepare(src: PreparedSplats, cam: SplatCamera, max_r: int, cancel: CancelToken | None) -> _Prep | None:
    """Cull, project and depth-sort the cloud; None when nothing can be visible."""
    d_all = src.means - cam.center
    r2 = cam.R[2]
    tz = d_all[:, 0] * r2[0] + d_all[:, 1] * r2[1] + d_all[:, 2] * r2[2]  # same sums as _project_block
    idx = np.flatnonzero(tz > Z_NEAR)
    if not len(idx):
        return None
    parts: list[dict[str, np.ndarray]] = []
    for s in range(0, len(idx), _PREP_CHUNK):
        _check(cancel)
        blk = idx[s : s + _PREP_CHUNK]
        parts.append(_project_block(src, blk, d_all[blk], cam, max_r))
    data = {k: np.concatenate([p[k] for p in parts], axis=-1) for k in parts[0]}
    if not len(data["u"]):
        return None
    order = np.argsort(data["cz"][3], kind="stable")  # depth rank; ties keep the input order
    return _Prep(**{k: np.ascontiguousarray(v[..., order]) for k, v in data.items()})


def _pairs(
    p: _Prep, sel: np.ndarray, y0: int, y1: int, width: int, done: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(key, alpha)`` of the pairs of the Gaussians ``sel`` (ascending ranks) with the
    pixels of rows ``[y0, y1)`` that are not ``done`` (band pixel flags), sorted by ``key =
    band pixel * len(p) + rank`` (pixel, then near to far)."""
    n = len(p)
    # one entry per (Gaussian, row): the column span where q <= t_cut, i.e. the roots of
    # qa dx^2 + 2 qb dy dx + qc dy^2 = t_cut, within the footprint square
    lo = np.maximum(p.i_lo[sel], y0)
    nrow = np.minimum(p.i_hi[sel], y1 - 1) - lo + 1
    g = np.repeat(sel, nrow)
    rows = np.repeat(lo - (np.cumsum(nrow) - nrow), nrow) + np.arange(len(g))
    dy = rows + 0.5 - p.v[g]
    u, qa, qb = p.u[g], p.qa[g], p.qb[g]
    half = np.sqrt(np.maximum(qa * p.t_cut[g] - dy * dy * p.inv_det[g], 0.0)) / qa
    mid = u - qb * dy / qa
    jl = np.maximum(np.ceil(mid - half - 0.5 - _EPS_PX), p.sj_lo[g]).astype(np.int64)
    jh = np.minimum(np.floor(mid + half - 0.5 + _EPS_PX), p.sj_hi[g]).astype(np.int64)
    cnt = np.maximum(jh - jl + 1, 0)
    # log alpha along the row from its first column: (A dx + B) dx + C
    dx0 = jl + 0.5 - u
    a_ = -0.5 * qa
    b_ = -qb * dy
    c_ = p.log_o[g] - 0.5 * p.qc[g] * dy * dy
    pix0 = (rows - y0) * width + jl
    # one entry per (Gaussian, pixel)
    k = np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    dx = np.repeat(dx0, cnt) + k
    la = (np.repeat(a_, cnt) * dx + np.repeat(b_, cnt)) * dx + np.repeat(c_, cnt)
    pix = np.repeat(pix0, cnt) + k
    ok = la >= _LOG_ALPHA_MIN
    ok &= ~done[pix]
    key = pix[ok] * n + np.repeat(g, cnt)[ok]
    la = la[ok]
    order = np.argsort(key)
    return key[order], np.minimum(ALPHA_MAX, np.exp(la[order]))


class _Band:
    """Accumulators of one band; runs of Gaussians are composited into it near to far.

    A pair is composited iff the transmittance after it stays >= 1e-4 (3DGS's early
    termination); its pixel is ``done`` from the first pair that would cross that limit.
    The set of composited pairs is the same however the Gaussians are split into runs, so
    skipping work at ``done`` pixels never changes the image."""

    def __init__(self, rows: int, width: int) -> None:
        self.rows, self.width, self.npix = rows, width, rows * width
        self.sums = np.zeros((5, self.npix))  # premultiplied r, g, b, alpha-weighted z, alpha
        self.median = np.full(self.npix, np.nan)
        self.log_t = np.zeros(self.npix, dtype=np.int64)  # log transmittance (fixed point)
        self.done = np.zeros(self.npix, dtype=bool)
        self.n_done = 0

    def open_runs(self, p: _Prep, sel: np.ndarray, y0: int) -> np.ndarray:
        """The Gaussians of ``sel`` whose candidate rectangle still has a pixel not done
        (a summed-area table of the open pixels)."""
        if not self.n_done:
            return sel
        sat = cv2.integral((~self.done).reshape(self.rows, self.width).view(np.uint8))
        r0 = np.maximum(p.i_lo[sel] - y0, 0)
        r1 = np.minimum(p.i_hi[sel] - y0 + 1, self.rows)
        c0, c1 = p.j_lo[sel], p.j_hi[sel] + 1
        n_open = sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0]
        return sel[n_open > 0]

    def composite(self, p: _Prep, key: np.ndarray, alpha: np.ndarray) -> None:
        n = len(key)
        pix = key // len(p)
        rank = key - pix * len(p)
        la = np.rint(np.log1p(-alpha) * _LOG_Q).astype(np.int64)
        cs = np.cumsum(la)
        first = np.empty(n, dtype=bool)
        first[0] = True
        np.not_equal(pix[1:], pix[:-1], out=first[1:])
        fi = np.flatnonzero(first)
        seg_pix = pix[fi]
        # exclusive segmented sum: log T before each pair = (cs - la) - (cs - la at the
        # pixel's first pair) + the pixel's log T from earlier runs; int64, exact
        base = np.repeat(cs[fi] - la[fi] - self.log_t[seg_pix], np.diff(np.append(fi, n)))
        incl = cs - base
        excl = incl - la
        live = incl >= _LOG_T_MIN  # a prefix of every pixel's pairs
        n_live = np.add.reduceat(live.view(np.int8), fi, dtype=np.int64)
        w = np.where(live, alpha * np.exp(np.maximum(excl, _LOG_T_MIN) * (1.0 / _LOG_Q)), 0.0)
        for ch in range(4):
            self.sums[ch] += np.bincount(pix, weights=w * p.cz[ch][rank], minlength=self.npix)
        self.sums[4] += np.bincount(pix, weights=w, minlength=self.npix)
        cross = (incl <= _LOG_HALF) & (excl > _LOG_HALF)  # at most one pair per pixel (always live)
        self.median[pix[cross]] = p.cz[3][rank[cross]]
        some = n_live > 0
        self.log_t[seg_pix[some]] = incl[(fi + n_live - 1)[some]]
        ended = seg_pix[n_live < np.diff(np.append(fi, n))]
        self.done[ended] = True
        self.n_done += len(ended)


def _bands(p: _Prep, height: int, band_rows: int | None) -> list[tuple[int, int]]:
    """Row bands: fixed ``band_rows``, else as tall as the candidate-pair budget allows."""
    if band_rows is not None:
        return [(y, min(y + band_rows, height)) for y in range(0, height, band_rows)]
    width_est = (p.j_hi - p.j_lo + 1).astype(np.float64)  # candidates per row, upper bound
    diff = np.bincount(p.i_lo, weights=width_est, minlength=height + 1)
    diff -= np.bincount(p.i_hi + 1, weights=width_est, minlength=height + 1)
    per_row = np.cumsum(diff)[:height]
    cum = np.concatenate([[0.0], np.cumsum(per_row)])
    out: list[tuple[int, int]] = []
    y0 = 0
    while y0 < height:
        y1 = int(np.searchsorted(cum, cum[y0] + _PAIR_BUDGET, side="right")) - 1
        y1 = min(max(y1, y0 + 1), height)
        out.append((y0, y1))
        y0 = y1
    return out


def _runs(p: _Prep, sel: np.ndarray, y0: int, y1: int) -> list[np.ndarray]:
    """Split the (depth-ordered) Gaussians of a band into consecutive runs of at most
    ``_RUN_PAIRS`` candidate pairs (at least one Gaussian each)."""
    rows = np.minimum(p.i_hi[sel], y1 - 1) - np.maximum(p.i_lo[sel], y0) + 1
    est = np.cumsum(rows * (p.j_hi[sel] - p.j_lo[sel] + 1))
    out = []
    s = 0
    while s < len(sel):
        before = est[s - 1] if s else 0
        e = max(int(np.searchsorted(est, before + _RUN_PAIRS, side="right")), s + 1)
        out.append(sel[s:e])
        s = e
    return out


def render_splats(
    sc: SplatCloud | PreparedSplats,
    cam: SplatCamera,
    *,
    background: Sequence[float] = (0, 0, 0),
    max_radius_px: int = MAX_RADIUS_PX,
    band_rows: int | None = None,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> SplatRender:
    """Render a Gaussian splat cloud with ``cam`` (see the module docstring).

    ``sc`` is in the camera's world frame (any frame: glTF, or SHARP's C); pass
    :func:`prepare_splats` ``(sc)`` instead to render several views faster.  ``background``
    is an RGB triple 0-255 behind everything.  ``max_radius_px`` caps a footprint's
    half-size (huge Gaussians are cut off there rather than costing ``r^2`` pixels each).
    ``band_rows`` forces bands of that many rows (default: sized by the memory budget);
    the image does not depend on it.  Raises :class:`InputError` for a malformed cloud or
    arguments and ``Cancelled`` when ``cancel`` fires."""
    if not isinstance(cam, SplatCamera):
        raise InputError("render_splats needs a SplatCamera (see orbit_camera / SplatCamera.pinhole).")
    bg = _background(background)
    if int(max_radius_px) != max_radius_px or not 1 <= int(max_radius_px) <= _MAX_RADIUS_LIMIT:
        raise InputError(
            f"max_radius_px must be a whole number of pixels from 1 to {_MAX_RADIUS_LIMIT} (got {max_radius_px})."
        )
    if band_rows is not None and (int(band_rows) != band_rows or int(band_rows) < 1):
        raise InputError(f"band_rows must be a whole number >= 1 (got {band_rows}).")
    rep = _as_reporter(progress)
    t0 = time.perf_counter()
    H, W = cam.height, cam.width
    _check(cancel)
    rep(0.0, "Rendering the splat")
    src = sc if isinstance(sc, PreparedSplats) else prepare_splats(sc, cancel=cancel)
    p = _prepare(src, cam, int(max_radius_px), cancel)

    sums = np.zeros((5, H * W))  # premultiplied r, g, b, alpha-weighted z, alpha
    median = np.full(H * W, np.nan)
    diag: dict[str, Any] = {"n": src.n_input, "visible": 0, "pairs": 0, "bands": 0, "runs": 0}
    if p is not None:
        bands = _bands(p, H, None if band_rows is None else int(band_rows))
        diag.update(visible=len(p), bands=len(bands))
        for b, (y0, y1) in enumerate(bands):
            _check(cancel)
            sel = np.flatnonzero((p.i_lo <= y1 - 1) & (p.i_hi >= y0))
            if len(sel):
                acc = _Band(y1 - y0, W)
                for run in _runs(p, sel, y0, y1):
                    _check(cancel)
                    run = acc.open_runs(p, run, y0)
                    if not len(run):
                        continue
                    key, a = _pairs(p, run, y0, y1, W, acc.done)
                    diag["runs"] += 1
                    if len(key):
                        diag["pairs"] += len(key)
                        acc.composite(p, key, a)
                sums[:, y0 * W : y1 * W] = acc.sums
                median[y0 * W : y1 * W] = acc.median
            rep((b + 1) / len(bands), "Rendering the splat")
    alpha = np.minimum(sums[4], 1.0)
    out_rgb = sums[:3].T + (1.0 - alpha)[:, None] * bg
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(alpha >= DEPTH_MIN_ALPHA, sums[3] / np.maximum(sums[4], 1e-300), np.nan)
    diag["seconds"] = round(time.perf_counter() - t0, 3)
    log.debug(
        "render_splats %dx%d %s: %d of %d Gaussians visible, %d pairs, %d bands, %.2f s",
        W,
        H,
        cam.kind,
        diag["visible"],
        diag["n"],
        diag["pairs"],
        diag["bands"],
        diag["seconds"],
    )
    rep(1.0, "Rendering the splat")
    return SplatRender(
        rgb=np.clip(np.rint(out_rgb * 255.0), 0, 255).astype(np.uint8).reshape(H, W, 3),
        alpha=alpha.astype(np.float32).reshape(H, W),
        depth=depth.astype(np.float32).reshape(H, W),
        depth_median=median.astype(np.float32).reshape(H, W),
        diag=diag,
    )


# --------------------------------------------------------------------------- turntable
def _robust_box(sc: SplatCloud) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` of the cloud from per-axis percentiles of its fairly opaque Gaussians
    (stray floaters do not shrink the subject in the preview)."""
    means = np.asarray(sc.means, dtype=np.float64)
    fin = np.isfinite(means).all(axis=1)
    pts = means[fin & (np.asarray(sc.opacity, dtype=np.float64) >= _SHEET_OPACITY)]
    if len(pts) < 16:
        pts = means[fin]
    if not len(pts):
        raise InputError("Nothing to preview: the splat has no Gaussians with a finite position.")
    lo = np.percentile(pts, _SHEET_PCT, axis=0)
    hi = np.percentile(pts, 100.0 - _SHEET_PCT, axis=0)
    return lo, hi


def _label(tile: np.ndarray, text: str) -> None:
    """Draw ``text`` in the top-left corner (light text with a dark outline, in place)."""
    k = tile.shape[0] / 384.0
    scale = max(0.3, 0.6 * k)
    th = max(1, int(round(k)))
    (_, th_px), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    org = (max(2, int(6 * k)), max(2, int(6 * k)) + th_px)
    cv2.putText(tile, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(tile, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), th, cv2.LINE_AA)


def turntable_sheet(
    sc: SplatCloud,
    *,
    n: int = 8,
    size: int = 384,
    cols: int = 4,
    background: Sequence[float] = (32, 32, 32),
    labels: bool = True,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> np.ndarray:
    """Contact sheet of ``n`` orthographic views around the vertical (y) axis of an upright
    glTF-frame cloud (y up, the subject facing +z): ``phi = 0, 360/n, ...`` (0 = the front,
    90 = seen from the subject's left, 180 = the back), ``size x size`` pixels each, ``cols``
    per row, labelled with their azimuth.  All views share one scale and are framed on the
    robust bounding box (0.5 / 99.5 percentiles of the Gaussians with opacity >= 0.2, plus
    a 5 % margin): the orbit axis is the box's vertical centre line.  Returns HxWx3 uint8
    RGB; raises :class:`InputError` for an empty cloud or bad arguments."""
    for name, val, low in (("n", n, 1), ("size", size, 16), ("cols", cols, 1)):
        if int(val) != val or int(val) < low:
            raise InputError(f"turntable_sheet: {name} must be a whole number >= {low} (got {val}).")
    n, size, cols = int(n), int(size), min(int(cols), int(n))
    if _validate(sc) == 0:
        raise InputError("Nothing to preview: the splat is empty.")
    bg = np.rint(_background(background) * 255.0).astype(np.uint8)
    rep = _as_reporter(progress)
    lo, hi = _robust_box(sc)
    centre = 0.5 * (lo + hi)
    radius = 0.5 * math.hypot(hi[0] - lo[0], hi[2] - lo[2])  # the box turned about y fits in it
    tall = float(hi[1] - lo[1])
    span = max(2.0 * radius, tall, 1e-3) * (1.0 + 2.0 * _SHEET_MARGIN)
    m = span / size
    dist = 2.0 * (radius + tall) + 1.0  # outside the box: only far-off floaters fall behind the camera
    src = prepare_splats(sc, cancel=cancel)
    rows_n = math.ceil(n / cols)
    sheet = np.empty((rows_n * size, cols * size, 3), np.uint8)
    sheet[:] = bg
    for k in range(n):
        _check(cancel)
        phi = 360.0 * k / n
        cam = orbit_camera(centre, phi, distance=dist, width=size, height=size, kind="ortho", m=m)
        tile = render_splats(
            src, cam, background=background, progress=rep.sub(k / n, (k + 1) / n), cancel=cancel
        )
        img = np.ascontiguousarray(tile.rgb)
        if labels:
            _label(img, f"{phi:g} deg")
        r, c = divmod(k, cols)
        sheet[r * size : (r + 1) * size, c * size : (c + 1) * size] = img
    log.debug("turntable_sheet: %d views at %d px, %.4f m/px, axis at %s", n, size, m, np.round(centre, 3))
    return sheet


__all__ = [
    "ALPHA_MAX",
    "ALPHA_MIN",
    "CAMERA_KINDS",
    "DEPTH_MIN_ALPHA",
    "LOWPASS_PX2",
    "MAX_RADIUS_PX",
    "Z_NEAR",
    "T_MIN",
    "PreparedSplats",
    "SplatCamera",
    "SplatRender",
    "orbit_camera",
    "prepare_splats",
    "render_splats",
    "turntable_sheet",
]
