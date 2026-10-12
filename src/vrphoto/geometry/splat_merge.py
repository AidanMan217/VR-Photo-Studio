"""SHARP 360: stitch a back part onto SHARP's front, clean up, and make it life-size.

Design sections 3.7-3.10 and 3.12 (docs/design/SHARP360.md).

**Front guarantee.** A Gaussian of ``FrontModel.sc`` whose view angle ``theta`` is below
``seam_deg - 10`` or that lies in the protected outline band (``front.rim``) is kept
bit-identical by :func:`merge_parts` and is never recoloured or removed by the later steps
(only the opacity < 0.04 prune of the final decimation applies to it).

Everything works in SHARP's camera frame C except :func:`to_life_size`, which works on the
exported glTF-frame cloud after ``normalize_orientation``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.types import CancelToken, SplatCloud
from .back_mould import BackPart, HullIntervals
from .splat_fusion import Canon, FrontModel

#: Default seam angle (deg): the photo is trusted where the surface faces the camera within it.
SEAM_DEG = 60.0
#: Half width of the dithered seam band (deg).
SEAM_BAND_DEG = 10.0
#: Most Gaussians in the final object (``run.MAX_OBJECT_SPLATS``).
MAX_SPLATS = 750_000


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep from 0 at ``e0`` to 1 at ``e1`` (works for ``e0 > e1`` too)."""
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def seam_weights(cos_theta: np.ndarray, seam_deg: float = SEAM_DEG) -> np.ndarray:
    """SHARP weight ``w_s = smoothstep(cos(seam + 10), cos(seam - 10), cos_theta)``: 1 where
    the surface faces the photo camera, 0 beyond the seam band.  NaN -> 0."""
    raise NotImplementedError


def merge_parts(
    front: FrontModel,
    part: BackPart | None,
    *,
    seam_deg: float = SEAM_DEG,
    seed: int = 0,
    budget: int = MAX_SPLATS,
    cancel: CancelToken | None = None,
) -> tuple[SplatCloud, np.ndarray, dict[str, Any]]:
    """Visibility partition + dithered seam (design 3.7).  Returns ``(cloud, is_front,
    diag)``; ``is_front[i]`` marks rows that are unmodified rows of ``front.sc``.

    * Front Gaussians: protected (theta < seam - 10 or rim) are kept as-is; the others are
      kept with probability ``w_s`` but only dropped where *covered* (a generated Gaussian
      within 2 cm, 2 cm voxel hash, 27-neighbourhood).
    * Generated Gaussians: the centre and both ends ``mu +- 2 s1 a1`` are tested; dropped if
      any test point projects outside the 1 %-dilated matte or the image, or lies in front
      of the photo (``z < D_f - tau``); on the photographed surface (``|z - D_f| <= tau``)
      kept with probability ``1 - w_s``; hidden ones kept.
    * Budget: the generated part is decimated first to ``budget - |F| - 5000``; the front is
      decimated only if ``|F| > 0.8 budget`` (logged).
    ``part=None`` returns the front alone.  Deterministic for a given ``seed``.
    """
    raise NotImplementedError


def add_underlayer(
    sc: SplatCloud,
    is_front: np.ndarray,
    front: FrontModel,
    canon: Canon,
    part: BackPart,
    *,
    thickness: float = 0.11,
    cancel: CancelToken | None = None,
) -> tuple[SplatCloud, np.ndarray]:
    """For ``cloud`` parts (TRELLIS): add a shell inset 5 mm and clipped per ray to
    ``part.back_z - 5 mm``, keeping only shell Gaussians farther than 1.5 cm from every
    other Gaussian, so holes in the cloud show matching soft colour instead of see-through."""
    raise NotImplementedError


def feather_colors(sc: SplatCloud, is_front: np.ndarray, *, radius_m: float = 0.03) -> SplatCloud:
    """Blend generated Gaussians within ``radius_m`` of the front toward the front's local
    mean colour: ``c += w (c_F - c)``, ``w = 1 - smoothstep(0, radius, d)``, ``c_F`` = mean
    front colour over the 27 neighbouring 1.5 cm voxels, ``d`` = distance to the nearest
    front voxel centroid.  Front rows are never changed."""
    raise NotImplementedError


def voxel_components(P: np.ndarray, voxel: float) -> np.ndarray:
    """Connected-component labels (26-connectivity) of the occupied ``voxel``-sized cells of
    points ``P`` (Nx3), vectorised: edges via ``searchsorted`` over sorted int64 voxel keys
    for 13 offsets, labels by ``np.minimum.at`` propagation + pointer jumping.  Returns an
    int64 label per point (0..K-1, ordered by first occurrence)."""
    raise NotImplementedError


def remove_floaters(
    sc: SplatCloud,
    is_front: np.ndarray,
    front: FrontModel,
    canon: Canon,
    *,
    hull: HullIntervals | None = None,
    cancel: CancelToken | None = None,
) -> tuple[SplatCloud, np.ndarray, dict[str, Any]]:
    """Design 3.10.  Generated Gaussians are dropped if opacity < 0.05, max scale > 0.035
    height, anisotropy > 25 with max scale > 0.012 height, outside the 1 %-dilated matte
    (projected), more than 2 cm outside the hull, or radius outliers (< 3 neighbours in the
    27 cells of 1.5 cm).  Then, for everything, 1.5 cm voxel components: keep components
    with >= 1 % of the opacity mass or within 5 cm of the largest.  Protected front rows
    (theta < seam - 10 or rim) are never removed."""
    raise NotImplementedError


def to_life_size(
    sc: SplatCloud,
    *,
    mode: str,
    height_m: float,
    cut: bool = False,
) -> tuple[SplatCloud, float, np.ndarray, str]:
    """Scale an upright glTF-frame cloud (feet at y = 0, after ``normalize_orientation``)
    to life size.  Measured height ``h = P99.8(y | opacity > 0.5)``.  ``mode``:
    ``"height"`` -> ``k = height_m / h``; ``"photo"`` -> ``k = 1`` (SHARP's metric size);
    ``"auto"`` -> height unless ``cut`` (subject cut off by the photo border), then photo.
    Returns ``(scaled cloud, final height, box, used_mode)``, ``box`` =
    ``[[xmin, -0.02, zmin], [xmax, h' + 0.03, zmax]]`` with x/z from the P0.2/P99.8
    percentiles +- 3 cm, for ``crop_splats(aabb=box)``."""
    raise NotImplementedError


__all__ = [
    "MAX_SPLATS",
    "SEAM_BAND_DEG",
    "SEAM_DEG",
    "add_underlayer",
    "feather_colors",
    "merge_parts",
    "remove_floaters",
    "seam_weights",
    "smoothstep",
    "to_life_size",
    "voxel_components",
]
