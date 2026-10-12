"""SHARP 360: build the hidden back of the subject as a "mould" behind SHARP's front.

Design sections 3.5.1 and 3.6 (docs/design/SHARP360.md).  The back is a height field
along the photo camera's rays on the isotropic mould grid (``splat_fusion.MouldGrid``):
for every matte cell ``p`` the back surface is at depth ``z_b(p) > D_f(p)``.  It closes
exactly at the photo silhouette because the target thickness is zero there; where the
silhouette edge has depth (``Z_rim > D_ext``), a "rim wall" of Gaussians closes the side.

* :func:`inflation_prior` - the billboard-style inflated thickness behind the front.
* :func:`poisson_heightfield` - screened Poisson solve (matrix-free, Jacobi-preconditioned
  CG, coarse to fine) of the back depth from a target, boundary values and weights.
* :func:`gaussians_from_heightfield` - one flat Gaussian per (strided) cell.
* :func:`rim_wall` - Gaussians closing the side between front edge and back edge.
* :func:`shell_back` - the built-in back source (no AI): prior + colours from the outline.
* :func:`mould_back` - shared back builder for every source that provides a target depth.

All Gaussians are returned in SHARP's camera frame C (OpenCV axes, metres).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.types import CancelToken, FloatMap, ProgressReporter, SplatCloud
from .splat_fusion import Canon, FrontModel

#: Back opacity of mould Gaussians.
BACK_OPACITY = 0.95
#: The back stays at least this far (m) behind the front surface.
MIN_GAP_M = 0.005
#: Max relief detail (m) added from a depth map's high-pass when there are no side views.
DETAIL_M = 0.03


@dataclass
class HullIntervals:
    """Per mould-grid ray, the first contiguous ``[z_in, z_out]`` interval inside the visual
    hull of the accepted generated views (filled in by view-based sources, milestone M4).
    ``valid`` marks rays where the interval is known."""

    z_in: FloatMap
    z_out: FloatMap
    valid: np.ndarray


@dataclass
class BackPart:
    """A back made by one back source, in frame C, generated Gaussians only.

    ``kind`` is ``"mould"`` (a closed height-field back from this module) or ``"cloud"``
    (a free Gaussian cloud such as TRELLIS's, merged with a closure under-layer).
    ``ok=False`` with a plain-language ``reason`` makes the orchestrator fall back to the
    next source.  ``back_z`` is the back depth per mould cell (NaN where unknown), used by
    the under-layer and the floater filter.  ``views`` holds generated / user views
    (``views_lift.GenView``) for previews and the Brush polish (later milestones).
    """

    sc: SplatCloud
    kind: str = "mould"
    source: str = "shell"
    ok: bool = True
    reason: str = ""
    normals: np.ndarray | None = None  # (N, 3) outward normals of sc's Gaussians (C)
    back_z: FloatMap | None = None
    hull: HullIntervals | None = None
    views: list[Any] | None = None
    diag: dict[str, Any] | None = None


def inflation_prior(front: FrontModel, canon: Canon, *, thickness: float) -> tuple[FloatMap, FloatMap]:
    """``(Z_prior, T)`` on the mould grid: ``T = kappa * prof * mpp`` where ``prof`` comes
    from ``billboard.inflation_profile(matte)`` and ``mpp = D_f / f_m`` (metres per cell,
    ``f_m = cam.f / grid.cell_px``); ``kappa`` (<= 4) makes ``median(T) = thickness *
    canon.height_s``.  ``Z_prior = D_smooth + T`` with ``D_smooth`` the masked blur of
    ``D_f`` (sigma = 4 % of the silhouette height).  NaN outside the matte."""
    raise NotImplementedError


def poisson_heightfield(
    target: FloatMap,
    mask: np.ndarray,
    z_rim: FloatMap,
    *,
    weights: FloatMap | None = None,
    lam_rim: float = 10.0,
    lam_target: float = 0.01,
    iters: int = 150,
    levels: int = 3,
    cancel: CancelToken | None = None,
) -> FloatMap:
    """Minimise ``sum_(p,q) w_pq (z_q - z_p - (T_q - T_p))^2 + lam_rim sum_boundary (z_p -
    z_rim_p)^2 + lam_target sum (z_p - T_p)^2`` over ``mask`` cells (4-neighbour edges
    inside the mask, ``w_pq = min(w_p, w_q)`` from ``weights``, default 1).  The boundary is
    the set of mask cells with a 4-neighbour outside the mask.  Matrix-free Jacobi
    preconditioned conjugate gradient, coarse to fine (``levels`` halvings, ``iters`` per
    level, warm start by upsampling).  Returns ``z`` (NaN outside ``mask``)."""
    raise NotImplementedError


def gaussians_from_heightfield(
    z: FloatMap,
    mask: np.ndarray,
    front: FrontModel,
    colors: np.ndarray,
    *,
    stride: int = 1,
    opacity: float = BACK_OPACITY,
) -> tuple[SplatCloud, np.ndarray]:
    """One flat Gaussian per ``stride``-th mask cell at ``ray(p) * z(p)`` (frame C).

    Tangents by central differences, Gram-Schmidt: ``t1 = t_u^``, ``t2 = (t_v - (t_v.t1)
    t1)^``, ``n = t1 x t2``; if ``n . ray < 0`` negate ``t2`` and ``n`` so the normal points
    away from the camera (det stays +1).  Scales ``0.6*stride*|t_u|``, ``0.6*stride*|t_v_perp|``
    clamped to [1 mm, 5 cm], ``s3 = max(0.5 mm, 0.1*min)``.  ``colors`` is ``Hm x Wm x 3``
    uint8 (or float [0, 1]).  Returns ``(cloud, normals)``."""
    raise NotImplementedError


def rim_wall(
    front: FrontModel,
    z_rim: FloatMap,
    *,
    colors_front: np.ndarray,
    colors_back: np.ndarray | None = None,
    spacing_cells: float = 1.0,
) -> SplatCloud:
    """Gaussians along the matte contour (``cv2.findContours`` on the mould matte) from
    ``D_ext`` to ``z_rim`` along each boundary ray, every ~``spacing_cells`` cells of metric
    size; axes along the ray and the contour (0.6 x spacing), normal = outward contour
    normal; colour lerps from the front edge colour to the back edge colour.  Empty cloud
    where ``z_rim - D_ext <= 1 cm``."""
    raise NotImplementedError


def mould_back(
    front: FrontModel,
    canon: Canon,
    target: FloatMap,
    *,
    colors: np.ndarray,
    z_rim: FloatMap | None = None,
    clip: FloatMap | None = None,
    stride: int = 1,
    weights: FloatMap | None = None,
    source: str = "mould",
    cancel: CancelToken | None = None,
) -> BackPart:
    """Shared back builder: Poisson-solve ``target`` with rim values ``z_rim`` (default
    ``D_ext``), clamp to ``[D_f + MIN_GAP_M, clip - MIN_GAP_M]``, re-solve once with the
    clamped cells as soft constraints, sanity-check ``median(z_b - D_f)`` (falls back to the
    target with ``diag["fallback"]``), then back Gaussians + rim wall.  ``kind="mould"``,
    ``back_z`` = the solved depth."""
    raise NotImplementedError


def shell_back(
    front: FrontModel,
    canon: Canon,
    *,
    thickness: float = 0.11,
    back: str = "diffuse",
    stride: int = 1,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> BackPart:
    """The built-in back (no AI, design 3.5.1): ``inflation_prior`` used directly as the back
    depth (no solve), clamped to ``>= D_f + 5 mm``; colours ``"diffuse"`` = push-pull fill of
    an 8-cell band just inside the silhouette, x 0.9 (no face on the back of a head), or
    ``"mirror"`` = the photo colours.  Plus the rim wall.  ``source="shell"``."""
    raise NotImplementedError


__all__ = [
    "BACK_OPACITY",
    "DETAIL_M",
    "MIN_GAP_M",
    "BackPart",
    "HullIntervals",
    "gaussians_from_heightfield",
    "inflation_prior",
    "mould_back",
    "poisson_heightfield",
    "rim_wall",
    "shell_back",
]
