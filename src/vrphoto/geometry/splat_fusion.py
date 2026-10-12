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

Every long function takes ``progress=`` / ``cancel=`` and checks ``cancel`` regularly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.types import CancelToken, FloatMap, ProgressReporter, SplatCloud

#: Mould grid long side per quality (cells).
MOULD_LONG = {"fast": 512, "balanced": 768, "best": 1024}
#: SHARP's default prediction grid (1536 px input, stride 2) and layer count.
SHARP_GRID = 768


def depth_tol(z: np.ndarray | float) -> np.ndarray | float:
    """Depth tolerance ``tau(z) = max(2 cm, 1 % of z)`` (metres)."""
    return np.maximum(0.02, 0.01 * np.asarray(z)) if isinstance(z, np.ndarray) else max(0.02, 0.01 * float(z))


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
    def from_outputs(outs: dict[str, Any], params: dict[str, Any], *, n_raw: int, shape: tuple[int, int]) -> SharpCam:
        """Camera from the SHARP worker outputs (``f_px``, ``width``, ``height``), falling back
        to ``params["hfov_deg"]`` (default 65 deg) and the photo ``shape`` ``(H, W)``
        exactly like ``run._subject_splats``."""
        raise NotImplementedError

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continuous pixel coordinates ``(u, v, z)`` of ``...x3`` points in C (NaN where
        ``z <= 1e-6``)."""
        raise NotImplementedError

    def rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Ray directions with ``z = 1``: ``((u - W/2)/f, (v - H/2)/f, 1)`` (``...x3``)."""
        raise NotImplementedError


@dataclass(frozen=True)
class MouldGrid:
    """Isotropic grid over the photo: ``Wm x Hm`` square cells of ``cell_px`` photo pixels."""

    Wm: int
    Hm: int
    cell_px: float

    @staticmethod
    def for_photo(width: int, height: int, long_side: int) -> MouldGrid:
        """Grid with ``max(Wm, Hm) == min(long_side, max(width, height))``."""
        raise NotImplementedError

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Photo pixel coordinates ``(u, v)`` (each ``Hm x Wm``) of the cell centres:
        ``u = (col + 0.5) * cell_px``, ``v = (row + 0.5) * cell_px``."""
        raise NotImplementedError

    def cell_of(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Flat cell index ``row * Wm + col`` of continuous photo coordinates, ``-1`` outside."""
        raise NotImplementedError

    def resize(self, img: np.ndarray, *, interpolation: int | None = None) -> np.ndarray:
        """Resample a photo-sized map (``H x W`` or ``H x W x C``) onto the grid
        (``INTER_AREA`` by default)."""
        raise NotImplementedError


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

    def to_S(self, P: np.ndarray) -> np.ndarray:
        """``...x3`` points from C to S."""
        return (np.asarray(P, dtype=np.float64) - self.c) @ self.R

    def to_C(self, P: np.ndarray) -> np.ndarray:
        """``...x3`` points from S to C."""
        return np.asarray(P, dtype=np.float64) @ self.R.T + self.c


def sharp_front(
    sc_full: SplatCloud,
    alpha: FloatMap,
    cam: SharpCam,
    rgb: np.ndarray,
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
       ``s1' = 1.5 s2`` and ``s2, s3`` scaled by ``D_near/z`` (same look from the photo);
    7. mould-grid maps, normals, ``cos_theta``, per-Gaussian ``theta_deg`` and ``rim``.

    Falls back (``stats["fallback"] = True``) to the plain matte filter of
    ``run._subject_splats`` when fewer than ``max(500, 1 %)`` Gaussians would remain.
    ``rgb`` is the photo (``H x W x 3`` uint8, same size as ``alpha``).
    """
    raise NotImplementedError


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
    the subject width)."""
    raise NotImplementedError


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

_ = math  # used by implementations
