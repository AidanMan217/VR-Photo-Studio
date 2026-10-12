"""SHARP 360: clean SHARP's photographed front and find the subject's upright frame.

Part of the SHARP 360 fusion core (docs/design/SHARP360.md sections 3.0-3.3).  Pure
numpy / OpenCV, CPU only.

Frames and grids (CONTRACTS section 13):

* **C** - SHARP's camera frame: OpenCV axes (x right, y down, z forward), metres, the
  photo's camera at the origin.  Continuous pixel coordinates ``u = f*X/Z + W/2``,
  ``v = f*Y/Z + H/2``; the pixel index is ``floor(u)`` (== ``round(u - 0.5)`` as in
  ``run._subject_splats``).  ``geometry.camera`` puts pixel *centres* at integers, so
  subtract 0.5 before calling its ``project`` / ``unproject``.
* **SHARP grid** ``G x G`` (anisotropic over the photo): SHARP predicts about one Gaussian
  per cell and layer.  Used only to build the front z-buffer.
* **Mould grid** ``Hm x Wm`` (isotropic, :class:`MouldGrid`): long side 512 / 768 / 1024
  cells, cell ``k`` covers photo pixels ``[k*cell_px, (k+1)*cell_px)``.  Distance
  transforms, normals, the back height field and back Gaussians live on this grid.
* **S** - upright subject frame (:class:`Canon`): axes ``x^`` (right), ``y^`` (down, i.e.
  ``-up``), ``z^`` (horizontal viewing direction, away from the camera), expressed in C;
  ``P_S = R.T @ (P_C - c)`` with ``R = [x^ y^ z^]`` (columns).

Front clean-up (:func:`sharp_front`).  SHARP reconstructs the whole photo with (usually)
two layers of Gaussians per grid cell.  Inside the subject's outline the second layer is
mostly the wall / floor *behind* the subject ("wall blobs" that show from the side), and
at silhouette depth jumps SHARP stretches "card" Gaussians along the camera ray from the
subject edge to the background (streaks seen from the side).  The front depth per SHARP
cell is the depth where the accumulated opacity of the (non-card) Gaussians sorted near
to far reaches 0.5, computed for all cells at once: ``lexsort`` by (cell, z), a cumulative
sum of ``log1p(-alpha)`` restarted per cell, and one ``searchsorted`` on the globally
non-decreasing key ``2 * rank(cell) + accumulated_alpha``.  A translucent floater cannot
define the surface that way.  Gaussians clearly behind that surface are dropped unless
they sit next to an internal depth edge (disocclusion fill, e.g. the torso behind an arm);
the 1 % ring around the matte keeps only Gaussians at the subject's depth (hair wisps);
cards are slid along their own ray onto the subject (sizes rescaled so they look the same
from the photo's camera).  The per-cell maps the back builders need are then resampled
onto the isotropic mould grid.

Canonical frame (:func:`canonical_frame`): "up" from a floor plane under the feet, else
the elongated axis of the front, else the camera's up; then the S axes, robust sizes, the
orbit-axis depth prior and whether the subject is cut off by the photo border.

Every long function takes ``progress=`` / ``cancel=`` and checks ``cancel`` regularly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from ..core.errors import InputError, VRPhotoError
from ..core.fill import push_pull_fill
from ..core.log import get_logger
from ..core.types import CancelToken, FloatMap, ProgressReporter, SplatCloud
from .mesh import _as_reporter, _check
from .splat import _validate, fit_plane_ransac, quat_to_rotmat, rotmat_to_quat, subset_splats

log = get_logger(__name__)

#: Mould grid long side per quality (cells).
MOULD_LONG = {"fast": 512, "balanced": 768, "best": 1024}
#: SHARP's default prediction grid (1536 px input, stride 2) and layer count.
SHARP_GRID = 768

_GRID_MATCH = 0.02  # N_raw within 2 % of L * 768**2 counts as SHARP's own grid
_MIN_Z = 0.05  # candidates must be at least this far in front of the camera (m)
_RING_FRAC = 0.01  # matte dilation (fraction of the photo's long side), as _subject_splats
_CARD_RATIO = 4.0  # s1 > 4 * s2 ...
_CARD_COS = 0.8  # ... |a1 . ray| > 0.8 ...
_CARD_MIN_M = 0.01  # ... and s1 > 1 cm
_CARD_S1 = 1.5  # collapsed card: s1' = 1.5 * s2'
_ALPHA_MAX = 0.999  # keeps log1p(-alpha) finite
_FRONT_ACC = 0.5  # accumulated alpha that defines the front surface
_DISOCC_M = 0.05  # D_far - D_f above this marks an internal depth edge
# a cell more than max(3 tau, this share of the subject's size) behind its 5x5 median depth is
# background seen through a gap, not subject (~25 cm for an adult, a few cm for a small object)
_OUTLIER_FRAC = 0.15
_FALLBACK_MIN = 500  # fewer kept Gaussians than max(this, 25 % of the subject's SHARP cells) ...
_FALLBACK_FRAC = 0.25  # ... -> plain matte filter (the clean-up keeps about one per cell)
_RIM_CELLS = 2.0  # protected outline band: dt <= max(2 cells, 0.4 % of the long side)
_RIM_FRAC = 0.004
_FLOOR_TOL_M = 0.01
_FLOOR_MAX_DEG = 35.0
_FLOOR_MIN_INLIERS = 0.25
_FLOOR_FEET_M = 0.10
_FEET_PCT = 0.1  # "lowest front points": this height percentile of the front
_FLOOR_MAX_PTS = 20_000
_PCA_MAX_DEG = 25.0
_PCA_MIN_ELONG = 1.8  # ratio of the two largest standard deviations
_PCA_MIN_DEG = 3.0  # a PCA axis closer than this to the camera's vertical is noise
_PCA_ITERS = 8  # Tukey-reweighted line fits ...
_PCA_TUKEY = 2.5  # ... with the cut-off at 2.5 x the median distance from the axis
_PITCH_WARN_DEG = 30.0
_CUT_BAND = 0.01  # cut-off test: matte within 1 % of the border ...
_CUT_SPAN = 0.15  # ... spanning more than 15 % of the subject width
_CAM_UP = np.array([0.0, -1.0, 0.0])
_ROT_CHUNK = 262_144  # rotation matrices built per block (float64 N x 3 x 3)


def depth_tol(z: np.ndarray | float) -> np.ndarray | float:
    """Depth tolerance ``tau(z) = max(2 cm, 1 % of z)`` (metres)."""
    return np.maximum(0.02, 0.01 * np.asarray(z)) if isinstance(z, np.ndarray) else max(0.02, 0.01 * float(z))


def _positive_or(value: Any, default: float) -> float:
    """``float(value)`` when it is a finite positive number, else ``default``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) and v > 0 else default


def _detect_grid(n_raw: int) -> tuple[int, int, bool]:
    """``(G, layers, matched)``: SHARP's 768 grid when ``n_raw ~= L * 768**2`` (L in 1..3),
    else ``round(sqrt(n_raw / 2))`` with 2 layers."""
    n = int(n_raw)
    if n <= 0:
        return SHARP_GRID, 2, False
    for layers in (2, 1, 3):
        if abs(n - layers * SHARP_GRID**2) <= _GRID_MATCH * layers * SHARP_GRID**2:
            return SHARP_GRID, layers, True
    return max(1, round(math.sqrt(n / 2.0))), 2, False


@dataclass(frozen=True)
class SharpCam:
    """SHARP's pinhole camera: focal length ``f`` (pixels of the ``width x height`` photo),
    principal point at the image centre, plus the detected SHARP grid.

    ``n_raw`` is the number of Gaussians SHARP returned; ``grid`` = 768 when
    ``n_raw ~= L * 768**2`` for an ``L`` in 1..3 (``layers = L``), otherwise
    ``round(sqrt(n_raw / 2))`` with ``layers = 2`` (logged)."""

    f: float
    width: int
    height: int
    n_raw: int = 0
    grid: int = SHARP_GRID
    layers: int = 2

    @staticmethod
    def from_outputs(
        outs: dict[str, Any], params: dict[str, Any], *, n_raw: int, shape: tuple[int, int]
    ) -> SharpCam:
        """Camera from the SHARP worker outputs (``f_px``, ``width``, ``height``), falling back
        to ``params["hfov_deg"]`` (default 65 deg) and the photo ``shape`` ``(H, W)``
        exactly like ``run._subject_splats``.  Missing, non-numeric or non-positive values
        fall back one by one."""
        outs = outs or {}
        params = params or {}
        h, w = int(shape[0]), int(shape[1])
        width = max(1, int(round(_positive_or(outs.get("width"), float(w)))))
        height = max(1, int(round(_positive_or(outs.get("height"), float(h)))))
        f = _positive_or(outs.get("f_px"), 0.0)
        source = "f_px"
        if f <= 0.0:
            hfov = _positive_or(params.get("hfov_deg"), 65.0)
            if hfov >= 179.0:
                hfov = 65.0
            f = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)
            source = f"{hfov:g} deg horizontal FOV"
        grid, layers, matched = _detect_grid(n_raw)
        log.info(
            "SHARP camera: %dx%d, f = %.1f px (%s); %d Gaussians -> grid %d x %d, %d layer(s)%s",
            width,
            height,
            f,
            source,
            int(n_raw),
            grid,
            grid,
            layers,
            "" if matched else " (not SHARP's 768 grid: estimated)",
        )
        return SharpCam(f=float(f), width=width, height=height, n_raw=int(n_raw), grid=grid, layers=layers)

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continuous pixel coordinates ``(u, v, z)`` of ``...x3`` points in C (NaN where
        ``z <= 1e-6``)."""
        P = np.asarray(P, dtype=np.float64)
        z = P[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            zs = np.where(z > 1e-6, z, np.nan)
            u = self.f * P[..., 0] / zs + self.width / 2.0
            v = self.f * P[..., 1] / zs + self.height / 2.0
        return u, v, z

    def rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Ray directions with ``z = 1``: ``((u - W/2)/f, (v - H/2)/f, 1)`` (``...x3``)."""
        uu, vv = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
        return np.stack(
            [(uu - self.width / 2.0) / self.f, (vv - self.height / 2.0) / self.f, np.ones_like(uu)], axis=-1
        )

    def sharp_cell(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Flat SHARP-grid cell ``row * grid + col`` (``row = floor(v * G / H)``, ``col =
        floor(u * G / W)``) of continuous pixel coordinates, ``-1`` outside the photo."""
        uu, vv = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
        G = int(self.grid)
        with np.errstate(invalid="ignore"):
            ok = (
                np.isfinite(uu)
                & np.isfinite(vv)
                & (uu >= 0)
                & (uu < self.width)
                & (vv >= 0)
                & (vv < self.height)
            )
        out = np.full(uu.shape, -1, np.int64)
        col = np.clip(np.floor(uu[ok] * G / self.width), 0, G - 1).astype(np.int64)
        row = np.clip(np.floor(vv[ok] * G / self.height), 0, G - 1).astype(np.int64)
        out[ok] = row * G + col
        return out


@dataclass(frozen=True)
class MouldGrid:
    """Isotropic grid over the photo: ``Wm x Hm`` square cells of ``cell_px`` photo pixels."""

    Wm: int
    Hm: int
    cell_px: float

    @staticmethod
    def for_photo(width: int, height: int, long_side: int) -> MouldGrid:
        """Grid with ``max(Wm, Hm) == min(long_side, max(width, height))``; the short side
        is ``round(short / cell_px)`` cells (at least 1)."""
        try:
            w, h, n = int(width), int(height), int(long_side)
        except (TypeError, ValueError):
            w = h = n = 0
        if w < 1 or h < 1 or n < 1:
            raise InputError(f"Invalid mould grid: photo {width} x {height}, long side {long_side} cells.")
        big = max(w, h)
        cell = big / min(n, big)
        return MouldGrid(Wm=max(1, round(w / cell)), Hm=max(1, round(h / cell)), cell_px=float(cell))

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Photo pixel coordinates ``(u, v)`` (each ``Hm x Wm``) of the cell centres:
        ``u = (col + 0.5) * cell_px``, ``v = (row + 0.5) * cell_px``."""
        rows, cols = np.mgrid[0 : self.Hm, 0 : self.Wm]
        return (cols + 0.5) * self.cell_px, (rows + 0.5) * self.cell_px

    def cell_of(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Flat cell index ``row * Wm + col`` of continuous photo coordinates, ``-1`` outside."""
        uu, vv = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
        with np.errstate(invalid="ignore"):
            col = np.floor(uu / self.cell_px)
            row = np.floor(vv / self.cell_px)
            ok = (
                np.isfinite(col)
                & np.isfinite(row)
                & (col >= 0)
                & (col < self.Wm)
                & (row >= 0)
                & (row < self.Hm)
            )
        out = np.full(uu.shape, -1, np.int64)
        out[ok] = row[ok].astype(np.int64) * self.Wm + col[ok].astype(np.int64)
        return out

    def resize(self, img: np.ndarray, *, interpolation: int | None = None) -> np.ndarray:
        """Resample a photo-sized map (``H x W`` or ``H x W x C``) onto the grid
        (``INTER_AREA`` by default).  Bool maps come back as float32 coverage."""
        a = np.asarray(img)
        if a.dtype == bool or a.dtype.kind not in "uf" or a.dtype == np.float16:
            a = a.astype(np.float32)
        interp = cv2.INTER_AREA if interpolation is None else int(interpolation)
        if a.shape[:2] == (self.Hm, self.Wm):
            return a.copy()
        out = cv2.resize(np.ascontiguousarray(a), (self.Wm, self.Hm), interpolation=interp)
        if a.ndim == 3 and out.ndim == 2:  # OpenCV drops a single channel axis
            out = out[..., None]
        return out


@dataclass
class FrontModel:
    """SHARP's cleaned subject (the photographed front) plus the per-cell maps the back
    builders and the merge need.  All maps are on ``grid``; depths are ``z`` in C (metres).

    Guarantee: ``sc`` contains only Gaussians that were in SHARP's output (some "card"
    Gaussians collapsed along their own ray, see :func:`sharp_front`), in C.
    """

    sc: SplatCloud  # cleaned subject Gaussians, frame C
    cam: SharpCam
    grid: MouldGrid
    matte: np.ndarray  # (Hm, Wm) bool: subject (alpha >= 0.5)
    ring: np.ndarray  # (Hm, Wm) bool: 1 %-dilated matte minus matte (soft edge / hair)
    D_f: FloatMap  # (Hm, Wm) front depth (transmittance 0.5), filled inside matte | ring, NaN elsewhere
    D_near: FloatMap  # (Hm, Wm) eroded (5x5 min) D_f
    D_far: FloatMap  # (Hm, Wm) dilated (9x9 max) D_f
    D_ext: FloatMap  # (Hm, Wm) D_near push-pull extrapolated into ring and the whole grid
    A_f: FloatMap  # (Hm, Wm) accumulated alpha of the front in that cell, [0, 1]
    normals: np.ndarray  # (Hm, Wm, 3) unit front normals in C, facing the camera (NaN outside)
    cos_theta: FloatMap  # (Hm, Wm) n . (-ray) (1 = faces the camera), NaN outside
    dt: FloatMap  # (Hm, Wm) distance to the matte boundary in cells (0 outside)
    rgb: np.ndarray  # (Hm, Wm, 3) uint8 photo colours on the grid
    cell: np.ndarray  # (N,) int64 mould cell of each Gaussian of ``sc`` (-1 = outside grid)
    theta_deg: np.ndarray  # (N,) float32 view angle of each Gaussian's cell (90 outside matte)
    rim: np.ndarray  # (N,) bool: protected outline band (dist to boundary <= max(2 cells, 0.4 % long))
    stats: dict[str, Any] = field(default_factory=dict)
    # (N,) int64 row of SHARP's cloud each Gaussian of ``sc`` came from (None when unknown,
    # e.g. a synthetic model); for diagnostics and the "exact F rows" bookkeeping
    src: np.ndarray | None = None


@dataclass
class Canon:
    """Upright subject frame S (see module docstring) and the subject's size.

    ``c = R @ [x0, y0, 0] + z_axis * z^`` ... concretely ``c`` is the point with S
    coordinates ``(0, 0, 0)``: the orbit axis (vertical through the subject) at the
    bounding-box centre height.  ``z_axis`` is the orbit axis' depth along ``z^``.
    """

    R: np.ndarray  # (3, 3) columns x^, y^, z^ in C
    c: np.ndarray  # (3,) origin of S in C
    up_C: np.ndarray  # (3,) unit "up" in C (= -y^)
    up_source: str  # "floor" | "pca" | "camera"
    height_s: float  # subject height (m, SHARP's metric scale): P99.5 - P0.5 of y_S
    width_s: float  # subject width along x^ (m)
    T_prior: float  # expected median body thickness (m) = thickness * height_s
    z_axis: float  # depth of the orbit axis along z^ (C units) - prior: median front + T/2
    pitch_deg: float  # camera pitch relative to the floor (+ = looking down)
    cut_top: bool = False  # subject touches the top photo border
    cut_bottom: bool = False  # ... the bottom border (feet cut off)
    # how up was found (floor / PCA tests) and plain-language ``warnings`` for the job
    # (e.g. a steep camera pitch); JSON-serialisable, for debug.json
    diag: dict[str, Any] = field(default_factory=dict)

    def to_S(self, P: np.ndarray) -> np.ndarray:
        """``...x3`` points from C to S."""
        return (np.asarray(P, dtype=np.float64) - self.c) @ self.R

    def to_C(self, P: np.ndarray) -> np.ndarray:
        """``...x3`` points from S to C."""
        return np.asarray(P, dtype=np.float64) @ self.R.T + self.c


# --------------------------------------------------------------------------- front clean-up helpers
def _photo_inputs(alpha: Any, rgb: Any, cam: SharpCam) -> tuple[np.ndarray, np.ndarray]:
    """The matte (float32) and photo (uint8 RGB) at SHARP's ``width x height``."""
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim != 2 or a.size == 0:
        raise InputError(f"The cut-out matte must be an H x W map (got shape {a.shape}).")
    a = np.nan_to_num(a, nan=0.0)
    if rgb is None:
        img = np.full((*a.shape, 3), 128, np.uint8)
    else:
        img = np.asarray(rgb)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.ndim != 3 or img.shape[2] < 3 or img.shape[:2] != a.shape:
            raise InputError(
                f"The photo ({np.shape(rgb)}) must be H x W x 3 and match the matte ({a.shape})."
            )
        img = img[..., :3]
        if img.dtype != np.uint8:
            scale = 255.0 if img.dtype.kind == "f" and float(np.nanmax(img, initial=0.0)) <= 1.0 else 1.0
            img = np.clip(np.nan_to_num(img.astype(np.float64)) * scale, 0, 255).round().astype(np.uint8)
    W, H = cam.width, cam.height
    if a.shape != (H, W):
        # SHARP saw a resized photo: work in its pixel grid (same field of view)
        log.info("Matte %dx%d resampled to SHARP's %dx%d", a.shape[1], a.shape[0], W, H)
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(np.ascontiguousarray(img), (W, H), interpolation=cv2.INTER_AREA)
    return a, np.ascontiguousarray(img)


def _ring_kernel(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _ring_px(width: int, height: int) -> int:
    """Matte dilation radius in photo pixels (``run._subject_splats``)."""
    return max(2, int(round(_RING_FRAC * max(width, height))))


def _card_axes(
    P: np.ndarray, scales: np.ndarray, quats: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(is_card, i1, i2, i3, a1z)``: card test of design 3.2 step 2, the axis indices of
    the largest / middle / smallest scale and the z component of the long axis (unit)."""
    s = np.abs(np.asarray(scales, dtype=np.float64))
    order = np.argsort(s, axis=1)
    i3, i2, i1 = order[:, 0], order[:, 1], order[:, 2]
    rows = np.arange(len(s))
    s1, s2 = s[rows, i1], s[rows, i2]
    is_card = np.zeros(len(s), bool)
    a1z = np.zeros(len(s))
    pre = np.flatnonzero((s1 > _CARD_RATIO * s2) & (s1 > _CARD_MIN_M))
    for a in range(0, len(pre), _ROT_CHUNK):  # only the elongated ones need their rotation
        idx = pre[a : a + _ROT_CHUNK]
        Rm = quat_to_rotmat(np.asarray(quats[idx], dtype=np.float64))
        a1 = Rm[np.arange(len(idx)), :, i1[idx]]
        p = P[idx]
        rhat = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        is_card[idx] = np.abs((a1 * rhat).sum(axis=1)) > _CARD_COS
        a1z[idx] = a1[:, 2]
    return is_card, i1, i2, i3, a1z


def _collapse_cards(
    P: np.ndarray, scales: np.ndarray, quats: np.ndarray, i2: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(scales, quats)`` (float32) of cards at ``P`` slid along their rays by ``k`` (``P' =
    k P``) and squashed along the ray.

    The covariance becomes ``M Sigma M`` with ``M = k (I - r r^T) + c r r^T`` (``r`` = the
    unit ray): the photo camera's projection Jacobian at ``k P`` is ``1/k`` times the one at
    ``P`` and annihilates ``r``, so the projected footprint is exactly unchanged whatever the
    card's tilt, while ``c`` shrinks the extent along the ray to ``1.5 k s2`` (for a card
    lying exactly on its ray: ``s1' = 1.5 s2'`` and ``s2, s3`` times ``k``)."""
    n = len(P)
    out_s = np.empty((n, 3), np.float32)
    out_q = np.empty((n, 4), np.float32)
    eye = np.eye(3)
    for a in range(0, n, _ROT_CHUNK):
        sl = slice(a, a + _ROT_CHUNK)
        p = np.asarray(P[sl], dtype=np.float64)
        r = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        s = np.asarray(scales[sl], dtype=np.float64)
        Rm = quat_to_rotmat(np.asarray(quats[sl], dtype=np.float64))
        cov = (Rm * (s * s)[:, None, :]) @ np.swapaxes(Rm, 1, 2)
        kk = np.asarray(k[sl], dtype=np.float64)
        along = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", r, cov, r), 1e-30))
        s2 = s[np.arange(len(s)), i2[sl]]
        c = _CARD_S1 * kk * s2 / along
        rr = r[:, :, None] * r[:, None, :]
        M = kk[:, None, None] * (eye - rr) + c[:, None, None] * rr  # symmetric
        evals, evecs = np.linalg.eigh(M @ cov @ M)
        flip = np.linalg.det(evecs) < 0  # a proper rotation for the quaternion
        evecs[flip, :, 0] *= -1.0
        out_s[sl] = np.sqrt(np.maximum(evals, 1e-20))
        out_q[sl] = rotmat_to_quat(evecs)
    return out_s, out_q


def _transmittance_depth(
    cell: np.ndarray, z: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Front depth per occupied cell by transmittance: ``(cells, depth, acc)``.

    Sorted near to far within each cell, ``acc = 1 - prod(1 - alpha)``; the depth is that of
    the first Gaussian where ``acc >= 0.5`` (else of the one with the largest ``alpha * T``),
    ``acc`` the cell's final accumulated alpha."""
    if len(cell) == 0:
        e = np.zeros(0)
        return np.zeros(0, np.int64), e, e
    order = np.lexsort((z, cell))
    cs, zs = cell[order], z[order]
    al = np.clip(alpha[order], 0.0, _ALPHA_MAX)
    start = np.ones(len(cs), bool)
    start[1:] = cs[1:] != cs[:-1]
    seg = np.cumsum(start) - 1
    starts = np.flatnonzero(start)
    ends = np.append(starts[1:], len(cs))
    lg = np.log1p(-al)
    cum = np.cumsum(lg)
    local = cum - (cum - lg)[starts][seg]  # inclusive log-transmittance, restarted per cell
    acc = -np.expm1(local)
    # acc never decreases within a cell and stays < 2, so 2*rank + acc is sorted globally
    key = 2.0 * seg + acc
    first = np.searchsorted(key, 2.0 * np.arange(len(starts)) + _FRONT_ACC, side="left")
    weight = al * np.exp(local - lg)  # alpha * transmittance in front of it
    strongest = np.lexsort((weight, seg))[ends - 1]
    pick = np.where(first < ends, first, strongest)
    return cs[starts], zs[pick], acc[ends - 1]


def _subject_size(hard: np.ndarray, depth: np.ndarray, f: float) -> float:
    """Rough metric size of the subject (m): the longer side of the matte's bounding box at
    the median front depth (0 when unknown)."""
    rows = np.flatnonzero(hard.any(axis=1))
    cols = np.flatnonzero(hard.any(axis=0))
    d = depth[np.isfinite(depth) & (depth > 0)]
    if not len(rows) or not len(d):
        return 0.0
    extent_px = max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1)
    return float(extent_px) * float(np.median(d)) / float(f)


def _grid_depth(
    G: int,
    cells: np.ndarray,
    depth: np.ndarray,
    matte_G: np.ndarray,
    card_z: np.ndarray | None,
    gap_m: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """SHARP-grid depth maps (each ``G x G`` float32, no NaN).

    ``med``: holes push-pull filled + 3x3 median (the map resampled onto the mould grid);
    ``cls``: the cell's own transmittance depth where it is trusted, else ``med`` (used to
    classify Gaussians); ``near`` / ``far``: 5x5 erode / 9x9 dilate of ``cls``; ``ext``:
    ``near`` push-pull extrapolated from the matte cells.  A cell is not trusted when a card
    lies in front of its depth (the front is the card, the depth is the background behind
    it) or when it is more than ``max(3 tau, gap_m)`` behind its 5x5 median (background
    showing through a gap: it would otherwise define a far "surface"; ``gap_m`` scales with
    the subject, so a wall close behind a small subject is caught too).  Before the 3x3
    median the map is continued outside the subject with the nearest subject value, so the
    median does not pull the steep outline toward the interior."""
    n = G * G
    D = np.full(n, np.nan, np.float32)
    D[cells] = depth
    valid = np.isfinite(D)
    n_card = 0
    if card_z is not None:
        bad = valid & (D > card_z)
        n_card = int(bad.sum())
        valid &= ~bad
    D2 = np.where(valid, D, 0.0).astype(np.float32).reshape(G, G)
    known = valid.reshape(G, G)
    if not known.any():
        raise _NoDepth()
    # push-pull (not nearest) continuation here: a background cell on the outline must not
    # spread outward and become its own median
    ref = _median(push_pull_fill(D2, known), 5)
    outlier = known & (D2 - ref > np.maximum(3.0 * depth_tol(ref), gap_m))
    known &= ~outlier
    if not known.any():
        raise _NoDepth()
    med = _median(_subject_fill(D2, known, matte_G), 3)
    cls = np.where(known, D2, med).astype(np.float32)
    near = cv2.erode(cls, np.ones((5, 5), np.uint8))
    far = cv2.dilate(cls, np.ones((9, 9), np.uint8))
    src = matte_G | known
    ext = push_pull_fill(np.where(src, near, 0.0).astype(np.float32), src)
    maps = {"med": med, "cls": cls, "near": near, "far": far, "ext": ext}
    return maps, {
        "card_cells_refilled": n_card,
        "outlier_cells": int(outlier.sum()),
        "depth_cells": int(known.sum()),
    }


class _NoDepth(Exception):
    """No Gaussian inside the matte to take the front depth from."""


def _median(img: np.ndarray, k: int) -> np.ndarray:
    return cv2.medianBlur(img, k) if min(img.shape) >= k else img


def _nearest_fill(img: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Unknown cells take the value of the nearest known cell (``known`` must not be empty)."""
    if known.all():
        return img
    _, lab = cv2.distanceTransformWithLabels(
        np.where(known, 0, 1).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    src = np.flatnonzero(known.ravel())  # labels number the known cells in raster order, from 1
    return img.ravel()[src[lab.ravel() - 1]].reshape(img.shape)


def _subject_fill(D: np.ndarray, known: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """Holes inside the subject (``matte`` or ``known``) push-pull filled from ``known``,
    everything outside continued with the nearest subject value."""
    inside = known | matte
    filled = push_pull_fill(D, known)
    return _nearest_fill(np.where(inside, filled, 0.0).astype(np.float32), inside)


def _dist_cells(mask: np.ndarray) -> np.ndarray:
    """Distance (cells) of inside cells to the nearest outside cell; borders count as
    outside (like ``billboard._dist``)."""
    pad = np.pad(mask.astype(np.uint8), 1)
    return cv2.distanceTransform(pad, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1].astype(np.float32)


def _tangent(
    fwd: np.ndarray, bwd: np.ndarray, ok_f: np.ndarray, ok_b: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """Per-cell surface step along one grid axis: the central difference, but the one-sided
    one toward a matte neighbour at the outline, and the flatter one-sided difference across
    an internal depth edge (so the arm and the torso behind it both get their own normal)."""
    t = 0.5 * (fwd + bwd)
    jump = ok_f & ok_b & (np.abs(fwd[..., 2] - bwd[..., 2]) > tau)
    flat_f = np.abs(fwd[..., 2]) <= np.abs(bwd[..., 2])
    t = np.where((jump & flat_f)[..., None], fwd, t)
    t = np.where((jump & ~flat_f)[..., None], bwd, t)
    t = np.where((ok_f & ~ok_b)[..., None], fwd, t)
    return np.where((ok_b & ~ok_f)[..., None], bwd, t)


def _front_normals(
    grid: MouldGrid, cam: SharpCam, Z: np.ndarray, matte: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals (facing the camera) and ``cos_theta = n . (-ray)`` of the depth map
    ``Z`` (no NaN) from its unprojected 4-neighbours; NaN outside ``matte``."""
    u, v = grid.centers()
    d = cam.rays(u, v)
    P = d * Z[..., None].astype(np.float64)
    Pp = np.pad(P, ((1, 1), (1, 1), (0, 0)), mode="edge")
    mp = np.pad(matte, 1)
    tau = depth_tol(Z.astype(np.float64))
    t_u = _tangent(Pp[1:-1, 2:] - P, P - Pp[1:-1, :-2], mp[1:-1, 2:], mp[1:-1, :-2], tau)
    t_v = _tangent(Pp[2:, 1:-1] - P, P - Pp[:-2, 1:-1], mp[2:, 1:-1], mp[:-2, 1:-1], tau)
    n = np.cross(t_v, t_u)  # x right, y down: t_v x t_u points toward the camera
    rhat = d / np.linalg.norm(d, axis=-1, keepdims=True)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), -rhat)
    n = np.where(((n * rhat).sum(-1) > 0)[..., None], -n, n)
    cos_t = -(n * rhat).sum(-1)
    nan = np.float32(np.nan)
    normals = np.where(matte[..., None], n, nan).astype(np.float32)
    return normals, np.where(matte, cos_t, nan).astype(np.float32)


def _mould_maps(
    grid: MouldGrid,
    cam: SharpCam,
    med_G: np.ndarray,
    acc_G: np.ndarray,
    alpha: np.ndarray,
    rgb: np.ndarray,
) -> dict[str, np.ndarray]:
    """Design 3.2 step 7: the FrontModel maps on the mould grid from the SHARP-grid front
    depth ``med_G`` (filled, no NaN) and accumulated alpha ``acc_G``."""
    G = med_G.shape[0]
    W, H = cam.width, cam.height
    u, v = grid.centers()
    mx = (u * G / W - 0.5).astype(np.float32)  # SHARP cell centres sit at (k + 0.5) * W / G
    my = (v * G / H - 0.5).astype(np.float32)
    Z = cv2.remap(med_G, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    A = cv2.remap(acc_G, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    matte = grid.resize(alpha) >= 0.5
    r_cells = max(1, int(round(_ring_px(W, H) / grid.cell_px)))
    ring = (cv2.dilate(matte.astype(np.uint8), _ring_kernel(r_cells)) > 0) & ~matte
    band = matte | ring
    Z_fill = push_pull_fill(np.where(matte, Z, 0.0).astype(np.float32), matte) if matte.any() else Z
    nan = np.float32(np.nan)
    near_full = cv2.erode(Z_fill, np.ones((5, 5), np.uint8))
    far_full = cv2.dilate(Z_fill, np.ones((9, 9), np.uint8))
    D_ext = (
        push_pull_fill(np.where(matte, near_full, 0.0).astype(np.float32), matte)
        if matte.any()
        else near_full
    )
    normals, cos_t = _front_normals(grid, cam, Z_fill, matte)
    return {
        "matte": matte,
        "ring": ring,
        "D_f": np.where(band, Z_fill, nan).astype(np.float32),
        "D_near": np.where(band, near_full, nan).astype(np.float32),
        "D_far": np.where(band, far_full, nan).astype(np.float32),
        "D_ext": D_ext.astype(np.float32),
        "A_f": np.clip(A, 0.0, 1.0).astype(np.float32),
        "normals": normals,
        "cos_theta": cos_t,
        "dt": _dist_cells(matte),
        "rgb": np.asarray(grid.resize(rgb), dtype=np.uint8),
    }


def _mould_cells(grid: MouldGrid, u: np.ndarray, v: np.ndarray, W: int, H: int) -> np.ndarray:
    """Mould cell of photo coordinates; points inside the photo always get a cell (the grid
    can end a fraction of a cell short of the photo edge after rounding), others -1."""
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uc = np.where(inside, np.minimum(u, (grid.Wm - 0.5) * grid.cell_px), np.nan)
        vc = np.where(inside, np.minimum(v, (grid.Hm - 0.5) * grid.cell_px), np.nan)
    return grid.cell_of(uc, vc)


def _per_gaussian(
    grid: MouldGrid, maps: dict[str, np.ndarray], u: np.ndarray, v: np.ndarray, W: int, H: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(cell, theta_deg, rim)`` of Gaussians at photo coordinates ``(u, v)``."""
    cell = _mould_cells(grid, u, v, W, H)
    ok = cell >= 0
    cos_t = np.full(len(cell), np.nan, np.float32)
    cos_t[ok] = maps["cos_theta"].ravel()[cell[ok]]
    dt = np.zeros(len(cell), np.float32)
    dt[ok] = maps["dt"].ravel()[cell[ok]]
    rim = dt <= max(_RIM_CELLS, _RIM_FRAC * max(grid.Hm, grid.Wm))
    return cell, angle_deg(cos_t), rim


def _subject_filter(z: np.ndarray, base: np.ndarray, in_dil: np.ndarray) -> np.ndarray:
    """``run._subject_splats``' rule: centre on the 1 %-dilated matte and not far behind
    the near depths (``z <= 1.5 * p25 + 2 * (p25 - p5)``)."""
    keep = base & in_dil
    if int(keep.sum()) >= 50:
        p5, p25 = (float(q) for q in np.percentile(z[keep], [5.0, 25.0]))
        keep &= z <= 1.5 * p25 + 2.0 * (p25 - p5)
    return keep


# --------------------------------------------------------------------------- front clean-up
def sharp_front(
    sc_full: SplatCloud,
    alpha: FloatMap,
    cam: SharpCam,
    rgb: np.ndarray | None = None,
    *,
    mould_long: int = 768,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> FrontModel:
    """Clean SHARP's whole-photo reconstruction ``sc_full`` (frame C) down to the subject's
    photographed front (design section 3.2):

    1. candidates: ``z > 0.05`` and centre inside the matte dilated by 1 % of the long side;
       the ring = dilation minus hard matte (``alpha >= 0.5``);
    2. "cards" (``s1 > 4*s2``, ``|a1 . ray| > 0.8``, ``s1 > 1 cm``) are excluded from the
       depth statistics;
    3. front depth per SHARP cell by transmittance (first Gaussian where the accumulated
       alpha reaches 0.5, else the largest ``alpha*T``), holes filled (push-pull) + 3x3
       median, then ``D_near`` / ``D_far`` / ``D_ext``;
    4. hidden layer / background: matte-interior Gaussians with ``z > D_f + tau`` are kept
       only near internal depth edges (``D_far - D_f > 5 cm`` and ``z <= D_far + tau``),
       front floaters ``z < D_f - 3 tau`` are dropped;
    5. ring Gaussians kept iff ``|z - D_ext| <= 3 tau``;
    6. cards collapsed along their own ray onto ``D_near`` (``D_ext`` in the ring) with
       ``s1' = 1.5 s2`` and ``s2, s3`` scaled by ``D_near/z`` (same look from the photo;
       a card tilted off its ray gets its covariance collapsed instead, see below);
    7. mould-grid maps, normals, ``cos_theta``, per-Gaussian ``theta_deg`` and ``rim``.

    Details beyond the design: steps 4-6 compare against the cell's own transmittance depth
    (the 3x3 median is used for the maps and for cells that are not trusted: a card cell
    whose remaining Gaussians lie behind the card, i.e. the background seen past the
    subject's edge, or a cell more than ``max(3 tau, 15 % of the subject's size)`` behind
    its 5x5 median - the size is the longer side of the matte's bounding box at the median
    front depth, so a wall close behind a small subject is caught too).  A card is kept only
    when its extent along the ray (``+-2 s1``) reaches the subject depth (it starts no
    farther than ``D_far + tau``, ``D_ext + 3 tau`` in the ring, and ends no nearer than
    ``target - 3 tau``); cards lying wholly behind or in front are dropped.  A collapsed
    card gets the covariance ``M Sigma M`` with ``M = k (I - r r^T) + c r r^T`` (``k =
    target / z``, ``r`` the unit ray, ``c`` making the extent along the ray ``1.5 k s2``),
    re-diagonalised: for a card lying on its ray this is exactly the design's scale rule,
    and it also keeps the photo-view footprint of a tilted card (the design's rule would
    lose the part of ``s1`` across the ray).  Kept Gaussians stay in SHARP's row order
    (``FrontModel.src``).

    Falls back (``stats["fallback"] = True``) to the plain matte filter of
    ``run._subject_splats`` when fewer than ``max(500, 25 % of the SHARP-grid cells inside
    the matte)`` Gaussians would remain (the design's ``1 %`` of SHARP's whole-photo cloud
    made small full-body figures fall back: the clean-up keeps only about one Gaussian per
    subject cell); to the whole cloud, like ``_subject_splats``, if that is still too few; with
    plain-language ``stats["warnings"]`` for the job.  ``rgb`` is the photo (``H x W x 3``
    uint8, same size as ``alpha``; float in [0, 1] is converted; None gives a grey
    ``FrontModel.rgb``).  A matte at another resolution than SHARP's ``width x height`` is
    resampled to it (same field of view).  A camera with ``n_raw <= 0`` gets its grid detected
    from ``len(sc_full)``.  Raises :class:`InputError` for bad inputs or an empty matte and
    :class:`VRPhotoError` for an empty SHARP result.
    """
    rep = _as_reporter(progress)
    _check(cancel)
    n_raw = _validate(sc_full)
    if n_raw == 0:
        raise VRPhotoError(
            "SHARP returned an empty 3D result.", hint="Try again; if it keeps happening, reinstall SHARP."
        )
    try:
        long_cells = int(mould_long)
    except (TypeError, ValueError):
        long_cells = 0
    if long_cells < 8:
        raise InputError(f"The mould grid needs at least 8 cells on its long side (got {mould_long!r}).")
    if not (math.isfinite(float(cam.f)) and cam.f > 0 and cam.width >= 1 and cam.height >= 1):
        raise InputError(f"Invalid SHARP camera: f = {cam.f}, {cam.width} x {cam.height} px.")
    if cam.n_raw <= 0:
        G0, L0, _ = _detect_grid(n_raw)
        cam = replace(cam, n_raw=n_raw, grid=G0, layers=L0)
    rep(0.0, "Cleaning the SHARP front (removing background layers and edge streaks)")
    a_c, rgb_c = _photo_inputs(alpha, rgb, cam)
    W, H, G = cam.width, cam.height, max(1, int(cam.grid))
    hard = a_c >= 0.5
    if not hard.any():
        raise InputError("No subject was found in the cut-out.", hint="Check the subject cut-out (matte).")
    grid = MouldGrid.for_photo(W, H, long_cells)
    stats: dict[str, Any] = {
        "n_raw": n_raw,
        "grid": G,
        "layers": int(cam.layers),
        "n_expected": int(cam.layers) * G * G,
        "sharp_grid_match": bool(_detect_grid(n_raw)[2]),
        "mould": [grid.Hm, grid.Wm],
        "cell_px": round(grid.cell_px, 4),
        "fallback": False,
        "warnings": [],
    }

    # ---- 1. candidates
    P = np.asarray(sc_full.means, dtype=np.float64)
    scales = np.asarray(sc_full.scales, dtype=np.float64)
    op = np.clip(np.nan_to_num(np.asarray(sc_full.opacity, dtype=np.float64), nan=0.0), 0.0, 1.0)
    finite = np.isfinite(P).all(axis=1) & np.isfinite(scales).all(axis=1)
    u, v, z = cam.project(np.where(finite[:, None], P, 0.0))
    with np.errstate(invalid="ignore"):
        on_photo = finite & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui = np.clip(np.floor(np.nan_to_num(u)), 0, W - 1).astype(np.int64)
    vi = np.clip(np.floor(np.nan_to_num(v)), 0, H - 1).astype(np.int64)
    dil = cv2.dilate(hard.astype(np.uint8), _ring_kernel(_ring_px(W, H))) > 0
    in_dil = on_photo & dil[vi, ui]
    cand = np.flatnonzero(in_dil & (z > _MIN_Z))
    in_matte = hard[vi[cand], ui[cand]]
    zc, Pc = z[cand], P[cand]
    gcell = cam.sharp_cell(u[cand], v[cand])
    stats.update(candidates=int(len(cand)), ring_candidates=int((~in_matte).sum()))
    _check(cancel)
    rep(0.15, "Finding SHARP's edge streaks")

    # ---- 2. cards
    is_card, i1, i2, i3, a1z = _card_axes(Pc, scales[cand], np.asarray(sc_full.quats)[cand])
    s_c = np.abs(scales[cand])
    rows_c = np.arange(len(cand))
    s1 = s_c[rows_c, i1]
    stats["cards"] = int(is_card.sum())
    _check(cancel)
    rep(0.3, "Measuring the front surface")

    # ---- 3. front depth by transmittance on the SHARP grid
    sel = ~is_card & in_matte
    cells, depth, acc = _transmittance_depth(gcell[sel], zc[sel], op[cand][sel])
    card_z = None
    if is_card.any():
        card_z = np.full(G * G, np.inf)
        np.minimum.at(card_z, gcell[is_card], zc[is_card])
    matte_G = cv2.resize(hard.astype(np.float32), (G, G), interpolation=cv2.INTER_AREA) >= 0.5
    acc_G = np.zeros(G * G, np.float32)
    acc_G[cells] = acc
    acc_G = acc_G.reshape(G, G)
    _check(cancel)

    size_m = _subject_size(hard, depth, cam.f)
    gap_m = _OUTLIER_FRAC * size_m
    stats.update(subject_size_m=round(size_m, 4), gap_test_m=round(gap_m, 4))
    keep = np.zeros(len(cand), bool)
    target = np.full(len(cand), np.nan)
    gmaps: dict[str, np.ndarray] | None = None
    try:
        gmaps, gstats = _grid_depth(G, cells, depth, matte_G, card_z, gap_m)
        stats.update(gstats)
    except _NoDepth:
        log.warning("SHARP front: no Gaussian inside the matte to measure the depth from")
    _check(cancel)
    rep(0.45, "Removing background layers")

    if gmaps is not None:
        # ---- 4. matte interior: hidden layer, background, front floaters
        Dc = gmaps["cls"].ravel()[gcell]
        Dfar = gmaps["far"].ravel()[gcell]
        Dnear = gmaps["near"].ravel()[gcell]
        Dext = gmaps["ext"].ravel()[gcell]
        tau = depth_tol(Dc)
        m = in_matte & ~is_card
        behind = m & (zc > Dc + tau)
        fill_ok = behind & (Dfar - Dc > _DISOCC_M) & (zc <= Dfar + tau)
        floater = m & (zc < Dc - 3.0 * tau)
        keep |= m & ~floater & (~behind | fill_ok)
        # ---- 5. ring: soft edge / hair at the subject's depth only
        r = ~in_matte & ~is_card
        tau_x = depth_tol(Dext)
        ring_ok = r & (np.abs(zc - Dext) <= 3.0 * tau_x)
        keep |= ring_ok
        # ---- 6. cards: keep those reaching the subject, slid along their ray onto it
        target = np.where(in_matte, Dnear, Dext)
        reach = 2.0 * s1 * np.abs(a1z)
        start_lim = np.where(in_matte, Dfar + tau, Dext + 3.0 * tau_x)
        card_ok = is_card & (zc - reach <= start_lim) & (zc + reach >= target - 3.0 * depth_tol(target))
        keep |= card_ok
        stats.update(
            hidden_dropped=int((behind & ~fill_ok).sum()),
            hidden_kept=int(fill_ok.sum()),
            floaters_dropped=int(floater.sum()),
            ring_kept=int(ring_ok.sum()),
            ring_dropped=int((r & ~ring_ok).sum()),
            cards_collapsed=int(card_ok.sum()),
            cards_dropped=int((is_card & ~card_ok).sum()),
        )
    _check(cancel)
    rep(0.6, "Building the front maps")

    n_keep = int(keep.sum())
    # what the clean-up can keep is about one Gaussian per SHARP cell of the subject (it drops
    # the hidden layer), so the bar is a share of those cells, not of the whole-photo cloud:
    # a small full-body figure in a landscape photo must not fall back
    need = max(_FALLBACK_MIN, int(math.ceil(_FALLBACK_FRAC * int(matte_G.sum()))))
    stats["fallback_below"] = need
    if gmaps is not None and n_keep >= need:
        rows = cand[keep]  # ascending: SHARP's row order
        sc = subset_splats(sc_full, rows)
        kc = keep & is_card
        if kc.any():
            pos = np.flatnonzero(kc[keep])
            k = target[kc] / zc[kc]
            sc.means[pos] = (Pc[kc] * k[:, None]).astype(np.float32)
            sk, qk = _collapse_cards(Pc[kc], s_c[kc], np.asarray(sc_full.quats)[cand[kc]], i2[kc], k)
            sc.scales[pos] = sk
            sc.quats[pos] = qk
        maps = _mould_maps(grid, cam, gmaps["med"], acc_G, a_c, rgb_c)
        uu, vv = u[rows], v[rows]  # cards moved along their own ray: (u, v) unchanged
    else:
        # ---- 8. fallback: run._subject_splats' filter, maps from its own z-buffer
        stats["fallback"] = True
        log.warning(
            "SHARP front clean-up kept %d Gaussians (< %d); using the plain matte filter", n_keep, need
        )
        sub = _subject_filter(z, finite & (z > 1e-3), in_dil)
        if int(sub.sum()) >= need:
            stats["fallback_kind"] = "subject_filter"
            msg = (
                "The SHARP front could not be cleaned up as usual, so a simpler cut-out of the subject "
                "was used; the edges may show streaks from the side."
            )
        else:
            sub = finite.copy()
            stats["fallback_kind"] = "whole_scene"
            msg = (
                "The subject could not be separated from the background in SHARP's 3D result, so the "
                "whole scene was kept."
            )
        stats["warnings"].append(msg)
        rows = np.flatnonzero(sub)
        sc = subset_splats(sc_full, rows)
        uu, vv = u[rows], v[rows]
        zz = z[rows]
        with np.errstate(invalid="ignore"):
            on = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (zz > _MIN_Z)
        idx = np.flatnonzero(on)
        idx = idx[hard[vi[rows[idx]], ui[rows[idx]]]]
        fcells, fdepth, facc = _transmittance_depth(cam.sharp_cell(uu[idx], vv[idx]), zz[idx], op[rows[idx]])
        acc_G = np.zeros(G * G, np.float32)
        acc_G[fcells] = facc
        acc_G = acc_G.reshape(G, G)
        try:
            med = _grid_depth(G, fcells, fdepth, matte_G, None, gap_m)[0]["med"]
        except _NoDepth:  # nothing usable in the matte: a flat front at the typical depth
            zf = zz[np.isfinite(zz) & (zz > _MIN_Z)]
            med = np.full((G, G), float(np.median(zf)) if len(zf) else 1.0, np.float32)
        maps = _mould_maps(grid, cam, med, acc_G, a_c, rgb_c)
    _check(cancel)
    rep(0.9, "Finding the outline band")

    cell, theta, rim = _per_gaussian(grid, maps, uu, vv, W, H)
    stats.update(kept=len(sc), rim=int(rim.sum()), dropped=n_raw - len(sc))
    log.info(
        "SHARP front: kept %d of %d Gaussians (hidden dropped %s, ring kept %s / dropped %s, cards %s "
        "collapsed / %s dropped; grid %d%s)%s",
        len(sc),
        n_raw,
        stats.get("hidden_dropped", "-"),
        stats.get("ring_kept", "-"),
        stats.get("ring_dropped", "-"),
        stats.get("cards_collapsed", "-"),
        stats.get("cards_dropped", "-"),
        G,
        "" if stats["sharp_grid_match"] else " estimated",
        " [fallback: " + stats["fallback_kind"] + "]" if stats["fallback"] else "",
    )
    rep(1.0, "SHARP front cleaned")
    return FrontModel(
        sc=sc,
        cam=cam,
        grid=grid,
        matte=maps["matte"],
        ring=maps["ring"],
        D_f=maps["D_f"],
        D_near=maps["D_near"],
        D_far=maps["D_far"],
        D_ext=maps["D_ext"],
        A_f=maps["A_f"],
        normals=maps["normals"],
        cos_theta=maps["cos_theta"],
        dt=maps["dt"],
        rgb=maps["rgb"],
        cell=cell,
        theta_deg=theta,
        rim=rim,
        stats=stats,
        src=rows.astype(np.int64),
    )


# --------------------------------------------------------------------------- canonical frame
def _unit(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), 1e-300)


def _angle_to(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(_unit(a), _unit(b)))))))


def _floor_up(
    cam: SharpCam, sc_full: SplatCloud, hard: np.ndarray, front_pts: np.ndarray
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Up from a floor plane under the subject (design 3.3 step 1.1), or None with the
    reason in the diagnostics."""
    diag: dict[str, Any] = {"used": False}
    h, w = hard.shape
    rows = np.flatnonzero(hard.any(axis=1))
    cols = np.flatnonzero(hard.any(axis=0))
    v_low = float(rows[-1] + 1)  # below the lowest matte row
    c_mid, width = 0.5 * (cols[0] + cols[-1] + 1), float(cols[-1] + 1 - cols[0])
    P = np.asarray(sc_full.means, dtype=np.float64)
    op = np.asarray(sc_full.opacity, dtype=np.float64)
    finite = np.isfinite(P).all(axis=1) & np.isfinite(op)
    u, v, z = cam.project(np.where(finite[:, None], P, 0.0))
    gcell = cam.sharp_cell(u, v)
    u = u * (w / cam.width)  # matte pixels (the matte may be at another resolution)
    v = v * (h / cam.height)
    z_ref = float(np.median(front_pts[:, 2]))
    with np.errstate(invalid="ignore"):
        ok = (
            finite
            & (z > _MIN_Z)
            & (v >= v_low)
            & (v < h)
            & (u >= 0)
            & (u < w)
            & (np.abs(u - c_mid) <= 0.75 * width)
            & (op > 0.3)
            & (z >= 0.5 * z_ref)
            & (z <= 2.0 * z_ref)
        )
    idx = np.flatnonzero(ok & (gcell >= 0))
    dil = cv2.dilate(hard.astype(np.uint8), _ring_kernel(_ring_px(w, h))) > 0
    idx = idx[~dil[np.floor(v[idx]).astype(np.int64), np.floor(u[idx]).astype(np.int64)]]
    # only the visible floor: SHARP's second layer sits a little behind it along the ray,
    # and at grazing angles enough of it falls within the 1 cm tolerance to tilt the refit
    order = idx[np.lexsort((z[idx], gcell[idx]))]
    first = np.ones(len(order), bool)
    first[1:] = gcell[order[1:]] != gcell[order[:-1]]
    idx = np.sort(order[first])
    diag["points"] = int(len(idx))
    if len(idx) < 50:
        diag["reason"] = "too few points below the subject"
        return None, diag
    if len(idx) > _FLOOR_MAX_PTS:
        idx = np.sort(np.random.default_rng(0).choice(idx, _FLOOR_MAX_PTS, replace=False))
    normal, offset, frac = fit_plane_ransac(P[idx], _FLOOR_TOL_M, seed=0)
    if np.dot(normal, _CAM_UP) < 0:
        normal, offset = -normal, -offset
    ang = _angle_to(normal, _CAM_UP)
    feet = float(np.percentile(front_pts @ normal - offset, _FEET_PCT))  # height of the lowest front points
    diag.update(angle_deg=round(ang, 2), inliers=round(frac, 3), feet_m=round(feet, 3))
    if ang > _FLOOR_MAX_DEG:
        diag["reason"] = "plane too steep"
    elif frac < _FLOOR_MIN_INLIERS:
        diag["reason"] = "not planar enough"
    elif abs(feet) > _FLOOR_FEET_M:
        diag["reason"] = "the subject does not stand on it"
    else:
        diag["used"] = True
        return _unit(normal), diag
    return None, diag


def _weighted_pca(pts: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean, eigenvalues ascending, eigenvectors as columns)`` of weighted points."""
    mu = (pts * w[:, None]).sum(0) / w.sum()
    X = pts - mu
    evals, evecs = np.linalg.eigh((X * w[:, None]).T @ X / w.sum())
    return mu, evals, evecs


def _pca_up(pts: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Up from the opacity-weighted principal axis of the front (design 3.3 step 1.2).

    The elongation test uses the plain weighted PCA.  The axis itself is refined by a few
    Tukey-reweighted line fits: a raised arm, a bag or a bent leg tilts the plain major axis
    sideways by several degrees, while the torso / legs line is what is vertical.  The
    front surface cannot tell up more precisely than a few degrees, so an axis within 3 deg
    of the camera's vertical is not used (the camera's up is then the better guess)."""
    w = np.maximum(np.nan_to_num(weights), 0.0)
    if len(pts) < 3 or w.sum() <= 0:
        return None, {"used": False, "reason": "too few points"}
    _, evals, evecs = _weighted_pca(pts, w)
    elong = math.sqrt(max(evals[-1], 0.0) / max(evals[-2], 1e-18))
    axis = evecs[:, -1]
    wt = w
    for _ in range(_PCA_ITERS):
        mu, _, ev = _weighted_pca(pts, wt)
        axis = ev[:, -1]
        X = pts - mu
        dist = np.linalg.norm(X - np.outer(X @ axis, axis), axis=1)
        scale = _PCA_TUKEY * max(float(np.median(dist[wt > 0])), 1e-9)
        r = np.minimum(dist / scale, 1.0)
        wt_new = w * (1.0 - r * r) ** 2
        if wt_new.sum() <= 0:
            break
        wt = wt_new
    if np.dot(axis, _CAM_UP) < 0:
        axis = -axis
    ang = _angle_to(axis, _CAM_UP)
    diag: dict[str, Any] = {"used": False, "angle_deg": round(ang, 2), "elongation": round(elong, 2)}
    if ang > _PCA_MAX_DEG:
        diag["reason"] = "axis too far from the camera's vertical"
    elif elong <= _PCA_MIN_ELONG:
        diag["reason"] = "subject not elongated enough"
    elif ang < _PCA_MIN_DEG:
        diag["reason"] = "within the accuracy of the camera's vertical"
    else:
        diag["used"] = True
        return _unit(axis), diag
    return None, diag


def _cut_flags(hard: np.ndarray) -> tuple[bool, bool]:
    """``(cut_top, cut_bottom)``: matte pixels within 1 % of that border span more than 15 %
    of the subject's width."""
    h = hard.shape[0]
    cols = np.flatnonzero(hard.any(axis=0))
    if not len(cols):
        return False, False
    width = float(cols[-1] - cols[0] + 1)
    k = max(1, math.ceil(_CUT_BAND * h))
    out = []
    for band in (hard[:k], hard[-k:]):
        bc = np.flatnonzero(band.any(axis=0))
        out.append(bool(len(bc) and (bc[-1] - bc[0] + 1) > _CUT_SPAN * width))
    return out[0], out[1]


def canonical_frame(
    front: FrontModel,
    sc_full: SplatCloud,
    alpha: FloatMap,
    *,
    thickness: float = 0.11,
) -> Canon:
    """Upright subject frame (design section 3.3): up from a floor plane fitted to
    non-subject Gaussians below the feet (``fit_plane_ransac``, accepted within 35 deg of
    ``-y_C`` with >= 25 % inliers and the feet within 10 cm), else the opacity-weighted PCA
    axis of the front (within 25 deg, elongation > 1.8), else the camera's ``-y``.
    Then axes, ``height_s`` / ``width_s`` / ``T_prior``, the ``z_axis`` prior, pitch and
    the cut-off flags (matte pixels within 1 % of the top/bottom border spanning > 15 % of
    the subject width).

    Floor points: opacity > 0.3, outside the 1 %-dilated matte, below its lowest row,
    within a band 1.5 subject widths wide centred on it, at 0.5-2x the front's median
    depth, and only the nearest one per SHARP cell (the visible floor, not SHARP's second
    layer behind it); "feet" = the 0.1th height percentile of the front.  PCA: the axis
    comes from a robust (Tukey-reweighted) line fit and is ignored within 3 deg of the
    camera's vertical, see :func:`_pca_up`; elongation is the ratio of the two largest
    standard deviations of the plain weighted PCA.  Sizes and the axes use front Gaussians
    with opacity > 0.5 (the centroid ray also leaves out the outline band).  ``diag``
    records both up tests and a plain-language warning when ``|pitch|`` exceeds 30 deg.
    Raises :class:`InputError` for an empty front or matte or a bad ``thickness``."""
    try:
        thick = float(thickness)
    except (TypeError, ValueError):
        thick = float("nan")
    if not (math.isfinite(thick) and thick >= 0):
        raise InputError(f"The thickness must be a number >= 0 (got {thickness!r}).")
    P = np.asarray(front.sc.means, dtype=np.float64)
    op = np.asarray(front.sc.opacity, dtype=np.float64)
    ok = np.isfinite(P).all(axis=1) & np.isfinite(op)
    rel = ok & (op > 0.5)
    if int(rel.sum()) < 3:
        rel = ok
    if int(rel.sum()) < 3:
        raise InputError("The subject's front is empty, so its size and orientation cannot be measured.")
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    hard = np.nan_to_num(a, nan=0.0) >= 0.5
    if a.ndim != 2 or not hard.any():
        raise InputError("No subject was found in the cut-out.", hint="Check the subject cut-out (matte).")
    pts = P[rel]

    # ---- 1. up: floor, else PCA of the front, else the camera
    up, floor_diag = _floor_up(front.cam, sc_full, hard, pts)
    source = "floor"
    pca_diag: dict[str, Any] = {"used": False, "reason": "not needed"}
    if up is None:
        up, pca_diag = _pca_up(pts, op[rel])
        source = "pca"
    if up is None:
        up, source = _CAM_UP.copy(), "camera"
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(up[2])))))
    warnings: list[str] = []
    if abs(pitch) > _PITCH_WARN_DEG:
        way = "down" if pitch > 0 else "up"
        warnings.append(f"The camera looks {way} steeply ({abs(pitch):.0f}°); the back may be less accurate.")
        log.warning("Camera pitch %.1f deg (up from %s)", pitch, source)

    # ---- 2. axes
    y_hat = -up
    core = rel & ~np.asarray(front.rim, dtype=bool) if len(front.rim) == len(P) else rel
    d = _unit(P[core].mean(axis=0) if int(core.sum()) >= 3 else pts.mean(axis=0))
    z_vec = d - np.dot(d, y_hat) * y_hat
    if np.linalg.norm(z_vec) < 1e-6:  # looking straight along the vertical: keep the camera's x
        fwd = np.array([0.0, 0.0, 1.0])
        z_vec = fwd - np.dot(fwd, y_hat) * y_hat
        if np.linalg.norm(z_vec) < 1e-6:
            z_vec = np.cross(np.array([1.0, 0.0, 0.0]), y_hat)
    z_hat = _unit(z_vec)
    x_hat = np.cross(y_hat, z_hat)
    R = np.column_stack([x_hat, y_hat, z_hat])

    # ---- 3. sizes and the orbit-axis prior
    q = pts @ R
    x_lo, x_hi = (float(t) for t in np.percentile(q[:, 0], [0.5, 99.5]))
    y_lo, y_hi = (float(t) for t in np.percentile(q[:, 1], [0.5, 99.5]))
    height_s = max(y_hi - y_lo, 1e-3)
    T_prior = thick * height_s
    x0, y0 = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    z_axis = float(np.median(q[:, 2])) + 0.5 * T_prior

    # ---- 4. cut off by the photo border
    cut_top, cut_bottom = _cut_flags(hard)
    log.info(
        "Subject frame: up from %s (pitch %.1f deg), height %.3f m, width %.3f m, axis depth %.3f m%s%s",
        source,
        pitch,
        height_s,
        x_hi - x_lo,
        z_axis,
        ", cut off at the top" if cut_top else "",
        ", cut off at the bottom" if cut_bottom else "",
    )
    return Canon(
        R=R,
        c=R @ np.array([x0, y0, z_axis]),
        up_C=_unit(up),
        up_source=source,
        height_s=float(height_s),
        width_s=float(x_hi - x_lo),
        T_prior=float(T_prior),
        z_axis=z_axis,
        pitch_deg=float(pitch),
        cut_top=cut_top,
        cut_bottom=cut_bottom,
        diag={"floor": floor_diag, "pca": pca_diag, "warnings": warnings},
    )


def angle_deg(cos_theta: np.ndarray) -> np.ndarray:
    """``theta`` in degrees from ``cos_theta`` (NaN -> 90)."""
    c = np.nan_to_num(np.asarray(cos_theta, dtype=np.float64), nan=0.0)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0))).astype(np.float32)


__all__ = [
    "MOULD_LONG",
    "SHARP_GRID",
    "Canon",
    "FrontModel",
    "MouldGrid",
    "SharpCam",
    "angle_deg",
    "canonical_frame",
    "depth_tol",
    "sharp_front",
]
