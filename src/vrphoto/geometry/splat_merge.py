"""SHARP 360: stitch a back part onto SHARP's front, clean up, and make it life-size.

Design sections 3.7-3.10 and 3.12 (docs/design/SHARP360.md).

**Front guarantee.** A Gaussian of ``FrontModel.sc`` whose view angle ``theta`` is below
``seam_deg - 10`` or that lies in the protected outline band (``front.rim``) is kept
bit-identical by :func:`merge_parts` and is never recoloured or removed by the later steps
(only the opacity < 0.04 prune of the final decimation applies to it, and a front that
alone exceeds 80 % of the Gaussian budget is decimated, logged).

Everything works in SHARP's camera frame C except :func:`to_life_size`, which works on the
exported glTF-frame cloud after ``normalize_orientation``.

* :func:`merge_parts` - visibility partition with a dithered seam.  The SHARP weight
  ``w_s(theta)`` (:func:`seam_weights`) is 1 where the surface faces the photo camera and
  falls to 0 across the seam band ``seam +- 10 deg``.  Generated Gaussians are tested at
  their centre and both ends of their long axis (``mu +- 2 s1 a1``) against the photo: one
  that would show around the outline from the photo's camera (outside the 1 %-dilated
  matte) or in front of the photographed surface is dropped; one on the photographed
  surface survives with probability ``1 - w_s``, and SHARP's own unprotected Gaussians
  with probability ``w_s`` (only where a generated Gaussian covers them).  Dithering keeps
  about one opaque layer everywhere; cross-fading two 0.95 layers to 50 % each would leave
  holes (``1 - 0.525**2 ~ 0.72``).  One uniform number per Gaussian is drawn from
  ``np.random.default_rng(seed)`` up front, so a seed changes nothing but the seam.
* :func:`add_underlayer` - TRELLIS-like clouds have holes; a built-in shell just inside the
  cloud's back fills them (only where no other Gaussian is near).
* :func:`feather_colors` - generated colours near the photo are pulled toward the
  photo's local mean colour, so the seam does not show as a colour step.
* :func:`remove_floaters` - per-Gaussian rules for generated Gaussians, then 26-connected
  components of 1.5 cm voxels (:func:`voxel_components`) for everything; the voxels grow
  with a back that is sampled more coarsely (``BackPart.spacing_m``).
* :func:`to_life_size` - one explicit scale so the subject has the requested height.

Neighbourhood queries use a voxel hash: integer voxel coordinates packed into exact int64
keys (each axis rank-compressed when far outliers would overflow 62 bits), ``np.unique``
for the occupied voxels and one ``searchsorted`` per neighbour offset.  Queries run once
per occupied voxel, not per Gaussian.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..core.errors import InputError
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter, SplatCloud
from .back_mould import BackPart, HullIntervals, shell_back
from .mesh import _as_reporter, _check
from .splat import _validate, concat_splats, decimate_splats, quat_to_rotmat, scale_splats, subset_splats
from .splat_fusion import Canon, FrontModel, angle_deg, depth_tol

log = get_logger(__name__)

#: Default seam angle (deg): the photo is trusted where the surface faces the camera within it.
SEAM_DEG = 60.0
#: Half width of the dithered seam band (deg).
SEAM_BAND_DEG = 10.0
#: Most Gaussians in the final object (``run.MAX_OBJECT_SPLATS``).
MAX_SPLATS = 750_000
#: ``to_life_size`` modes (``run.SIZE_FROM``).
SIZE_MODES = ("auto", "height", "photo")

_COVER_VOXEL_M = 0.02  # a front Gaussian is "covered" by a generated one within this voxel hash
_FRONT_CAP_FRAC = 0.8  # the front is decimated only above this share of the budget
_BUDGET_MARGIN = 5000  # rows left free for later additions (at most 1 % of the budget)
_RELIABLE_OPACITY = 0.5  # front Gaussians this opaque give the reference colour
_UNDER_GAP_M = 0.015  # an under-layer Gaussian is added only this far from every other one
_FLOATER_VOXEL_M = 0.015
_SPACING_VOXELS = 1.5  # ... at least 1.5 x the back's own spacing: neighbours share or touch voxels
_MIN_OPACITY = 0.05
_MAX_SCALE_FRAC = 0.035  # of the subject height
_NEEDLE_RATIO = 25.0  # largest / middle scale ...
_NEEDLE_FRAC = 0.012  # ... with the largest above this share of the height
_HULL_SLACK_M = 0.02
_MIN_NEIGHBOURS = 3
_COMP_MASS_FRAC = 0.01
_COMP_NEAR_M = 0.05
_HEIGHT_PCT = 99.8
_BOX_PCT = (0.2, 99.8)
_BOX_PAD_M = 0.03
_FLOOR_PAD_M = 0.02
_SIZE_OPACITY = 0.5
_RIM_CELLS, _RIM_FRAC = 2.0, 0.004  # splat_fusion's outline band: dt <= max(2 cells, 0.4 % long)
_CHUNK = 1 << 18
_KEY_LIMIT = float(2**62)
_COORD_MAX = float(2**50)  # voxel coordinates are clipped here (far garbage shares edge voxels)

_OFFS27 = np.array(list(itertools.product((-1, 0, 1), repeat=3)), np.int64)
_OFFS13 = _OFFS27[14:]  # the half after (0, 0, 0): each 26-neighbour pair once


def _sphere_offsets(radius: float) -> np.ndarray:
    """Integer offsets with ``|o| <= radius`` (voxel units)."""
    r = int(math.ceil(radius))
    o = np.array(list(itertools.product(range(-r, r + 1), repeat=3)), np.int64)
    return o[(o * o).sum(1) <= radius * radius + 1e-9]


# --------------------------------------------------------------------------- arguments
def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep from 0 at ``e0`` to 1 at ``e1`` (works for ``e0 > e1`` too)."""
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _seam_arg(seam_deg: Any) -> float:
    try:
        s = float(seam_deg)
    except (TypeError, ValueError):
        s = math.nan
    if not (math.isfinite(s) and SEAM_BAND_DEG <= s <= 90.0):
        raise InputError(
            f"The seam angle must be between {SEAM_BAND_DEG:g} and 90 degrees (got {seam_deg!r})."
        )
    return s


def _positive_arg(x: Any, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = math.nan
    if not (math.isfinite(v) and v > 0):
        raise InputError(f"{what} must be a positive number (got {x!r}).")
    return v


def _mask_arg(is_front: Any, n: int) -> np.ndarray:
    m = np.asarray(is_front)
    if m.shape != (n,) or (m.dtype != bool and m.dtype.kind not in "iu"):
        raise InputError(f"is_front must be a boolean mask with {n} entries (got shape {m.shape}).")
    return m.astype(bool)


def _front_rows(front: FrontModel) -> tuple[int, np.ndarray, np.ndarray]:
    """``(len(front.sc), theta_deg, rim)`` with the per-Gaussian arrays checked."""
    n = _validate(front.sc)
    theta = np.asarray(front.theta_deg, dtype=np.float64)
    rim = np.asarray(front.rim)
    if theta.shape != (n,) or rim.shape != (n,):
        raise InputError(
            f"The cleaned front is inconsistent: {n} Gaussians but {theta.shape} angles / {rim.shape} "
            "outline flags."
        )
    return n, np.nan_to_num(theta, nan=90.0), rim.astype(bool)


@dataclass
class _Maps:
    """The front maps the merge needs, flat over the mould grid."""

    band: np.ndarray  # bool: 1 %-dilated matte (matte | ring)
    D_f: np.ndarray  # float64 front depth (NaN outside the band)
    cos_t: np.ndarray  # float64 cos(theta) (NaN outside the matte)
    dt: np.ndarray  # float64 distance to the outline (cells)
    rim_cells: float


def _maps(front: FrontModel) -> _Maps:
    shape = (int(front.grid.Hm), int(front.grid.Wm))

    def get(x: Any, what: str, dtype: Any) -> np.ndarray:
        a = np.asarray(x, dtype=dtype)
        if a.shape != shape:
            raise InputError(f"The front {what} map ({a.shape}) does not match the mould grid {shape}.")
        return a.ravel()

    matte = get(front.matte, "outline", bool)
    ring = get(front.ring, "outline ring", bool) if front.ring is not None else np.zeros_like(matte)
    return _Maps(
        band=matte | ring,
        D_f=get(front.D_f, "depth", np.float64),
        cos_t=get(front.cos_theta, "view angle", np.float64),
        dt=np.nan_to_num(get(front.dt, "outline distance", np.float64), nan=0.0),
        rim_cells=max(_RIM_CELLS, _RIM_FRAC * max(shape)),
    )


def _cells(front: FrontModel, P: np.ndarray) -> np.ndarray:
    """Flat mould cell of the projection of ``P`` (N x 3, frame C), -1 outside the photo or
    behind the camera.  Points inside the photo always get a cell (the grid can end a
    fraction of a cell short of the photo edge), as in ``sharp_front``."""
    cam, grid = front.cam, front.grid
    u, v, _ = cam.project(P)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    col = np.minimum(np.floor(np.where(ok, u, 0.0) / grid.cell_px), grid.Wm - 1).astype(np.int64)
    row = np.minimum(np.floor(np.where(ok, v, 0.0) / grid.cell_px), grid.Hm - 1).astype(np.int64)
    return np.where(ok, row * int(grid.Wm) + col, -1)


def _finite_rows(sc: SplatCloud) -> np.ndarray:
    return (
        np.isfinite(np.asarray(sc.means)).all(axis=1)
        & np.isfinite(np.asarray(sc.scales)).all(axis=1)
        & np.isfinite(np.asarray(sc.opacity))
    )


def _concat(first: SplatCloud, second: SplatCloud) -> SplatCloud:
    """``first`` then ``second``; when only ``first`` has higher SH bands, ``second`` gets zero
    bands (view-independent colour) instead of ``first`` losing its own."""
    if (
        first.sh_rest is not None
        and len(second)
        and (second.sh_rest is None or second.sh_rest.shape[1] != first.sh_rest.shape[1])
    ):
        second = replace(second, sh_rest=np.zeros((len(second), first.sh_rest.shape[1], 3), np.float32))
    return concat_splats([first, second])


# --------------------------------------------------------------------------- voxel hash
class _Lattice:
    """Integer voxel coordinates on one lattice (edge ``voxel``, origin shared by all the
    point sets given) with exact int64 keys for them and for every neighbour up to
    ``reach`` voxels away along each axis.  Only points of those sets have valid keys
    (the rank-compressed axes know no other values)."""

    def __init__(self, voxel: float, sets: list[np.ndarray], reach: int = 1) -> None:
        self.voxel = float(voxel)
        self.reach = int(reach)
        pts = [np.asarray(p, dtype=np.float64) for p in sets if len(p)]
        self.lo = np.min([p.min(axis=0) for p in pts], axis=0) if pts else np.zeros(3)
        qs = [self.coords(p) for p in pts]
        hi = np.max([q.max(axis=0) for q in qs], axis=0) if qs else np.zeros(3, np.int64)
        dims = hi + self.reach + 1
        self.ranks: list[np.ndarray] | None = None
        if float(np.prod(dims.astype(np.float64))) >= _KEY_LIMIT:
            # a few far outliers blow up the box: number each axis by the values that occur
            # (and their neighbours), which keeps the keys exact
            offs = np.arange(-self.reach, self.reach + 1)
            self.ranks = [
                np.unique(np.concatenate([(q[:, a, None] + offs).ravel() for q in qs])) for a in range(3)
            ]
            dims = np.array([len(r) for r in self.ranks], np.int64)
        self.dims = dims
        self.hashed = float(np.prod(dims.astype(np.float64))) >= _KEY_LIMIT

    def coords(self, P: np.ndarray) -> np.ndarray:
        """``N x 3`` int64 voxel coordinates (>= ``reach``) of finite points ``P``."""
        g = np.asarray(P, dtype=np.float64) - self.lo
        g /= self.voxel
        np.clip(g, 0.0, _COORD_MAX, out=g)
        q = np.floor(g, out=g).astype(np.int64)
        q += self.reach
        return q

    def keys(self, q: np.ndarray) -> np.ndarray:
        """int64 key per row of voxel coordinates (a voxel or a neighbour within ``reach``)."""
        if self.ranks is not None:
            q = np.stack([np.searchsorted(self.ranks[a], q[:, a]) for a in range(3)], axis=1)
        if self.hashed:  # pragma: no cover - needs ~1e6 distinct occupied values on every axis
            u = q.astype(np.uint64)
            mix = (
                u[:, 0] * np.uint64(0x9E3779B97F4A7C15)
                ^ u[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)
                ^ u[:, 2] * np.uint64(0x165667B19E3779F9)
            )
            return mix.view(np.int64)
        return (q[:, 0] * int(self.dims[1]) + q[:, 1]) * int(self.dims[2]) + q[:, 2]


def _lookup(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of each ``query`` key in ``sorted_keys`` (unique, ascending), -1 if absent."""
    if not len(sorted_keys):
        return np.full(len(query), -1, np.int64)
    pos = np.minimum(np.searchsorted(sorted_keys, query), len(sorted_keys) - 1)
    return np.where(sorted_keys[pos] == query, pos, -1)


def _occupied(lat: _Lattice, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(keys, coords, inverse)`` of the occupied voxels of coordinates ``q``: sorted unique
    keys, one coordinate row per voxel and each point's voxel index."""
    keys = lat.keys(q)
    uniq, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    return uniq, q[first], inv.reshape(-1)


def _near(query: np.ndarray, ref: np.ndarray, voxel: float) -> np.ndarray:
    """Per ``query`` point: is a ``ref`` point in one of the 27 voxels around it (so it may
    be within ``voxel``; one outside them is at least ``voxel`` away).  Non-finite query
    points are never near."""
    Q = np.asarray(query, dtype=np.float64)
    R = np.asarray(ref, dtype=np.float64)
    out = np.zeros(len(Q), bool)
    okq = np.flatnonzero(np.isfinite(Q).all(axis=1))
    R = R[np.isfinite(R).all(axis=1)]
    if not len(okq) or not len(R):
        return out
    lat = _Lattice(voxel, [Q[okq], R])
    rk = np.unique(lat.keys(lat.coords(R)))
    _, qv, inv = _occupied(lat, lat.coords(Q[okq]))
    hit = np.zeros(len(qv), bool)
    for o in _OFFS27:
        hit |= _lookup(rk, lat.keys(qv + o)) >= 0
    out[okq] = hit[inv]
    return out


def _voxel_roots(lat: _Lattice, keys: np.ndarray, qv: np.ndarray, cancel: CancelToken | None) -> np.ndarray:
    """Component root (the smallest voxel index) of each occupied voxel, 26-connectivity.

    Edges come from 13 neighbour offsets.  Each round hooks the larger root of every
    still-split edge onto the smaller one (``np.minimum.at``) and then jumps pointers
    (``lab = lab[lab]``) until every voxel points at a root; labels only decrease and
    ``lab[i] <= i`` always, so this ends, usually after a few rounds."""
    n = len(keys)
    # int32 and reused buffers: the first rounds touch ~13 edges per voxel, and fresh
    # allocations of that size cost more than the arithmetic
    idx = np.int32 if n < 2**31 - 1 else np.int64
    a_parts, b_parts = [], []
    for o in _OFFS13:
        j = _lookup(keys, lat.keys(qv + o))
        hit = np.flatnonzero(j >= 0)
        a_parts.append(hit.astype(idx))
        b_parts.append(j[hit].astype(idx))
    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    del a_parts, b_parts
    lab = np.arange(n, dtype=idx)
    nxt = np.empty_like(lab)
    la, lb, m = np.empty_like(a), np.empty_like(a), np.empty_like(a)
    while len(a):
        _check(cancel)
        e = len(a)
        np.take(lab, a, out=la[:e])
        np.take(lab, b, out=lb[:e])
        split = la[:e] != lb[:e]
        n_split = int(np.count_nonzero(split))
        if not n_split:
            break
        # hooking a joined edge is a no-op, so all edges can be used; once joined an edge
        # stays joined (both ends follow the same root), so the list shrinks when it pays
        np.minimum(la[:e], lb[:e], out=m[:e])
        np.minimum.at(lab, la[:e], m[:e])
        np.minimum.at(lab, lb[:e], m[:e])
        if n_split < e // 2:
            a, b = a[split], b[split]
        while True:
            np.take(lab, lab, out=nxt)
            if np.array_equal(nxt, lab):
                break
            lab, nxt = nxt, lab
    return lab.astype(np.int64)


def _first_occurrence_labels(roots: np.ndarray) -> np.ndarray:
    """Renumber per-point roots as ``0..K-1`` in order of first occurrence."""
    _, first, inv = np.unique(roots, return_index=True, return_inverse=True)
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[inv.reshape(-1)]


def voxel_components(P: np.ndarray, voxel: float, *, cancel: CancelToken | None = None) -> np.ndarray:
    """Connected-component labels (26-connectivity) of the occupied ``voxel``-sized cells of
    points ``P`` (Nx3), vectorised: edges via ``searchsorted`` over sorted int64 voxel keys
    for 13 offsets, labels by ``np.minimum.at`` propagation + pointer jumping.  Returns an
    int64 label per point (0..K-1, ordered by first occurrence).

    Rows with a non-finite coordinate get ``-1`` and ``cancel`` (optional) is checked every
    round (additions to the skeleton).  Raises :class:`InputError` for a bad shape or
    voxel size."""
    pts = np.asarray(P, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise InputError(f"Points must be an N x 3 array (got shape {pts.shape}).")
    v = _positive_arg(voxel, "The voxel size")
    labels = np.full(len(pts), -1, np.int64)
    ok = np.flatnonzero(np.isfinite(pts).all(axis=1))
    if not len(ok):
        return labels
    lat = _Lattice(v, [pts[ok]])
    keys, qv, inv = _occupied(lat, lat.coords(pts[ok]))
    labels[ok] = _first_occurrence_labels(_voxel_roots(lat, keys, qv, cancel)[inv])
    return labels


# --------------------------------------------------------------------------- merge
def seam_weights(cos_theta: np.ndarray, seam_deg: float = SEAM_DEG) -> np.ndarray:
    """SHARP weight ``w_s = smoothstep(cos(seam + 10), cos(seam - 10), cos_theta)``: 1 where
    the surface faces the photo camera, 0 beyond the seam band.  NaN -> 0."""
    seam = _seam_arg(seam_deg)
    c = np.asarray(cos_theta, dtype=np.float64)
    lo = math.cos(math.radians(seam + SEAM_BAND_DEG))
    hi = math.cos(math.radians(seam - SEAM_BAND_DEG))
    return np.where(np.isnan(c), 0.0, smoothstep(lo, hi, c))


def _generated_keep(
    front: FrontModel,
    maps: _Maps,
    G: SplatCloud,
    seam: float,
    rand: np.ndarray,
    cancel: CancelToken | None,
    keep_rim: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    """Visibility partition of the generated Gaussians (design 3.7), chunked.  With
    ``keep_rim`` the surface dither spares Gaussians whose centre cell is in the protected
    outline band."""
    n = len(G)
    keep = np.zeros(n, bool)
    cnt = {"outside": 0, "in_front": 0, "surface": 0, "surface_rim": 0, "surface_dropped": 0, "hidden": 0}
    for a in range(0, n, _CHUNK):
        _check(cancel)
        b = min(n, a + _CHUNK)
        m = b - a
        P = np.asarray(G.means[a:b], dtype=np.float64)
        s = np.abs(np.asarray(G.scales[a:b], dtype=np.float64))
        ok = np.isfinite(P).all(axis=1) & np.isfinite(s).all(axis=1)
        P = np.where(ok[:, None], P, 0.0)  # z = 0 projects nowhere: dropped below
        s = np.where(ok[:, None], s, 0.0)
        rows = np.arange(m)
        i1 = np.argmax(s, axis=1)
        a1 = quat_to_rotmat(G.quats[a:b])[rows, :, i1]
        reach = (2.0 * s[rows, i1])[:, None] * a1
        in_front = np.zeros(m, bool)
        c0 = d0 = None
        for k, Q in enumerate((P, P + reach, P - reach)):
            cell = _cells(front, Q)
            inside = cell >= 0
            inside[inside] = maps.band[cell[inside]]
            ok &= inside
            d = np.full(m, np.nan)
            d[inside] = maps.D_f[cell[inside]]
            with np.errstate(invalid="ignore"):
                in_front |= Q[:, 2] < d - depth_tol(d)
            if k == 0:
                c0, d0 = cell, d
        assert c0 is not None and d0 is not None
        in_front &= ok
        vis = ok & ~in_front
        with np.errstate(invalid="ignore"):
            surface = vis & (np.abs(P[:, 2] - d0) <= depth_tol(d0))
        # A mould back's rim wall and outermost cells close the side: the outline band keeps
        # its photo rows anyway, and the depth-map normals there look camera-facing although
        # the true surface is grazing, so dithering would remove most of that closure while
        # the front's rim discs are edge-on from the side and cannot fill it.
        rim = np.zeros(m, bool)
        if keep_rim:
            rim[surface] = maps.dt[c0[surface]] <= maps.rim_cells
        dither = surface & ~rim
        cos_t = np.full(m, np.nan)
        cos_t[dither] = maps.cos_t[c0[dither]]
        w = seam_weights(cos_t, seam)
        dropped = dither & (rand[a:b] < w)  # survives with probability 1 - w_s
        keep[a:b] = vis & ~dropped
        cnt["outside"] += int((~ok).sum())
        cnt["in_front"] += int(in_front.sum())
        cnt["surface"] += int(surface.sum())
        cnt["surface_rim"] += int((surface & rim).sum())
        cnt["surface_dropped"] += int(dropped.sum())
        cnt["hidden"] += int((vis & ~surface).sum())
    return keep, cnt


def merge_parts(
    front: FrontModel,
    part: BackPart | None,
    *,
    seam_deg: float = SEAM_DEG,
    seed: int = 0,
    budget: int = MAX_SPLATS,
    cancel: CancelToken | None = None,
    progress: ProgressReporter | None = None,
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

    Details: the output is the kept front rows in ``front.sc`` order, then the kept
    generated rows (``decimate_splats`` reorders those when it runs).  The generated
    in-front test applies to all three test points and uses ``D_f`` of the cell each one
    projects to; the surface test and ``w_s`` use the centre's cell.  For a ``"mould"``
    part, Gaussians on the photographed surface whose centre cell lies in the protected
    outline band (``dt <= max(2 cells, 0.4 % long)``) are not dithered: they are the rim
    wall and the outermost back cells that close the side, the photo rows there are kept
    anyway, and the depth-map normals make that band look camera-facing (``w_s ~ 1``)
    although the true surface is grazing (``diag["gen_surface_rim"]``; ``"cloud"`` parts
    are dithered everywhere as designed).  ``seed`` is drawn as
    one uniform number per generated and per front Gaussian before anything else, so
    another seed changes only the dithered seam rows.  ``|F|`` in the generated budget is
    capped at ``0.8 budget`` and the margin is ``min(5000, budget // 100)``.  The front
    is decimated only when what is left of it after the seam still exceeds ``0.8
    budget``; its merged rows then still count as front (``diag["front_decimated"]``).  A
    part-less front is only decimated above ``budget``, and a part with ``ok=False`` or no
    Gaussians counts as no part.  If the front has higher SH bands and the part does not,
    the generated rows get zero bands so the front keeps its own.  ``diag`` holds
    JSON-safe counts.  ``progress`` is an optional addition.  Raises :class:`InputError`
    for a bad seam angle, budget or seed, or an inconsistent front."""
    rep = _as_reporter(progress)
    seam = _seam_arg(seam_deg)
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or int(budget) < 1:
        raise InputError(f"The Gaussian budget must be a positive whole number (got {budget!r}).")
    n_budget = int(budget)
    try:
        rng = np.random.default_rng(int(seed) & 0xFFFF_FFFF_FFFF_FFFF)
    except (TypeError, ValueError):
        raise InputError(f"The variation seed must be a whole number (got {seed!r}).") from None
    F = front.sc
    nF, theta, rim = _front_rows(front)
    maps = _maps(front)
    protected = (theta < seam - SEAM_BAND_DEG) | rim
    diag: dict[str, Any] = {
        "seam_deg": seam,
        "seed": int(seed),
        "budget": n_budget,
        "front_in": nF,
        "front_protected": int(protected.sum()),
        "front_decimated": False,
    }
    _check(cancel)
    if part is None or not part.ok or len(part.sc) == 0:
        diag.update(part=None, front_kept=nF, gen_in=0, gen_kept=0)
        out = F
        if nF > n_budget:
            log.warning("SHARP front has %d Gaussians (> %d); decimating it", nF, n_budget)
            out = decimate_splats(F, n_budget, cancel=cancel)
            diag["front_decimated"] = True
        diag["out"] = len(out)
        return out, np.ones(len(out), bool), diag

    G = part.sc
    nG = _validate(G)
    rand_gen = rng.random(nG)
    rand_front = rng.random(nF)
    rep(0.0, "Joining the photo front and the back")

    # ---- generated part: drop what would show from the photo's camera
    keep_g, gcnt = _generated_keep(front, maps, G, seam, rand_gen, cancel, keep_rim=part.kind == "mould")
    G1 = subset_splats(G, keep_g)
    diag.update(part=part.source, gen_in=nG, **{f"gen_{k}": v for k, v in gcnt.items()})
    _check(cancel)
    rep(0.4, "Joining the photo front and the back")

    # ---- budget: the generated part gives way first
    margin = min(_BUDGET_MARGIN, n_budget // 100)
    f_cap = max(1, int(_FRONT_CAP_FRAC * n_budget))
    g_budget = n_budget - min(nF, f_cap) - margin
    if len(G1) > g_budget:
        log.info("Back part: decimating %d -> %d Gaussians (budget %d)", len(G1), max(g_budget, 0), n_budget)
        G1 = decimate_splats(G1, g_budget, cancel=cancel) if g_budget >= 1 else subset_splats(G1, [])
        diag["gen_decimated_to"] = len(G1)
    _check(cancel)
    rep(0.6, "Dithering the seam")

    # ---- front: unprotected Gaussians give way where the back covers them
    cand = np.flatnonzero(~protected)
    w_f = seam_weights(np.cos(np.radians(theta[cand])), seam)
    covered = _near(np.asarray(F.means)[cand], G1.means, _COVER_VOXEL_M)
    keep_f = protected.copy()
    keep_f[cand] = ~covered | (rand_front[cand] < w_f)
    Fk = subset_splats(F, keep_f)
    diag.update(front_seam=len(cand), front_covered=int(covered.sum()), front_kept=len(Fk))
    if len(Fk) > f_cap:
        log.warning("SHARP front has %d Gaussians (> %d); decimating it too", len(Fk), f_cap)
        Fk = decimate_splats(Fk, f_cap, cancel=cancel)
        diag["front_decimated"] = True
    _check(cancel)

    out = _concat(Fk, G1)
    is_front = np.zeros(len(out), bool)
    is_front[: len(Fk)] = True
    diag.update(gen_kept=len(G1), out=len(out))
    log.info(
        "Merge: front %d of %d kept (%d protected), back %d of %d kept (%d outside the outline, %d in "
        "front, %d of %d on the photo dithered away)",
        len(Fk),
        nF,
        diag["front_protected"],
        len(G1),
        nG,
        gcnt["outside"],
        gcnt["in_front"],
        gcnt["surface_dropped"],
        gcnt["surface"],
    )
    rep(1.0, "Front and back joined")
    return out, is_front, diag


# --------------------------------------------------------------------------- under-layer
def _cloud_back_depth(front: FrontModel, part: BackPart) -> np.ndarray:
    """``part.back_z`` when it is a usable mould-grid map, else the farthest part Gaussian
    per mould cell of the matte (NaN where there is none)."""
    shape = (int(front.grid.Hm), int(front.grid.Wm))
    if part.back_z is not None:
        bz = np.asarray(part.back_z, dtype=np.float64)
        if bz.shape == shape and np.isfinite(bz).any():
            return bz
    matte = np.asarray(front.matte, dtype=bool).ravel()
    P = np.asarray(part.sc.means, dtype=np.float64)
    P = P[np.isfinite(P).all(axis=1)]
    cell = _cells(front, P)
    ok = cell >= 0
    ok[ok] = matte[cell[ok]]
    z = np.full(shape[0] * shape[1], -np.inf)
    np.maximum.at(z, cell[ok], P[ok, 2])
    return np.where(np.isfinite(z), z, np.nan).reshape(shape)


def add_underlayer(
    sc: SplatCloud,
    is_front: np.ndarray,
    front: FrontModel,
    canon: Canon,
    part: BackPart,
    *,
    thickness: float | None = None,
    cancel: CancelToken | None = None,
) -> tuple[SplatCloud, np.ndarray]:
    """For ``cloud`` parts (TRELLIS): add a shell inset 5 mm and clipped per ray to
    ``part.back_z - 5 mm``, keeping only shell Gaussians farther than 1.5 cm from every
    other Gaussian, so holes in the cloud show matching soft colour instead of see-through.

    The shell is :func:`back_mould.shell_back` (diffuse colours, rim wall included) with
    ``clip=back_z``.  Without a usable ``part.back_z`` the farthest part Gaussian per mould
    cell is used.  "Farther than 1.5 cm" is tested on a 1.5 cm voxel hash: the 27 voxels
    around the shell Gaussian hold no other Gaussian.  ``thickness`` defaults to the
    user's thickness, ``canon.T_prior / canon.height_s`` (the skeleton's fixed 0.11 default
    ignored that option).  Mould parts (closed by construction) and ``part=None`` return
    the input unchanged; so does a shell that cannot be built (logged), since the cloud
    alone is still a usable back.  Added rows go last with ``is_front = False``."""
    n = _validate(sc)
    isf = _mask_arg(is_front, n)
    if part is None or part.kind != "cloud":
        return sc, isf
    if thickness is None:
        h = float(canon.height_s)
        thickness = float(canon.T_prior) / h if math.isfinite(h) and h > 0 else 0.11
    _check(cancel)
    back_z = _cloud_back_depth(front, part)
    try:
        shell = shell_back(front, canon, thickness=thickness, back="diffuse", clip=back_z, cancel=cancel)
    except InputError as e:
        log.warning("Closure under-layer skipped: %s", e)
        return sc, isf
    _check(cancel)
    lonely = ~_near(shell.sc.means, sc.means, _UNDER_GAP_M)
    add = subset_splats(shell.sc, lonely)
    log.info("Closure under-layer: %d of %d shell Gaussians fill holes in the back", len(add), len(shell.sc))
    if not len(add):
        return sc, isf
    return _concat(sc, add), np.concatenate([isf, np.zeros(len(add), bool)])


# --------------------------------------------------------------------------- colours
def feather_colors(
    sc: SplatCloud,
    is_front: np.ndarray,
    *,
    radius_m: float = 0.03,
    cancel: CancelToken | None = None,
) -> SplatCloud:
    """Blend generated Gaussians within ``radius_m`` of the front toward the front's local
    mean colour: ``c += w (c_F - c)``, ``w = 1 - smoothstep(0, radius, d)``, ``c_F`` = mean
    front colour over the 27 neighbouring 1.5 cm voxels, ``d`` = distance to the nearest
    front voxel centroid.  Front rows are never changed.

    "Front" here is the reliable front: ``is_front`` rows with opacity >= 0.5 (faint
    wisps carry background colour).  The voxel edge is ``radius_m / 2`` (1.5 cm at the
    default), so the 27 voxels reach at least ``radius_m / 2`` in every direction; a
    Gaussian with no front voxel among them is unchanged.  Only the DC colour changes;
    the other arrays are shared with ``sc``.  ``cancel`` is an optional addition."""
    n = _validate(sc)
    isf = _mask_arg(is_front, n)
    r = _positive_arg(radius_m, "The feathering radius")
    P = np.asarray(sc.means, dtype=np.float64)
    finite = _finite_rows(sc)
    ref = np.flatnonzero(isf & finite & (np.asarray(sc.opacity) >= _RELIABLE_OPACITY))
    gen = np.flatnonzero(~isf & finite)
    if not len(ref) or not len(gen):
        return sc
    lat = _Lattice(0.5 * r, [P[ref], P[gen]])
    fkeys, _, finv = _occupied(lat, lat.coords(P[ref]))
    col = np.asarray(sc.colors, dtype=np.float64)
    nf = len(fkeys)
    cnt = np.bincount(finv, minlength=nf).astype(np.float64)
    csum = np.stack([np.bincount(finv, weights=col[ref, i], minlength=nf) for i in range(3)], axis=1)
    cen = (
        np.stack([np.bincount(finv, weights=P[ref, i], minlength=nf) for i in range(3)], axis=1)
        / cnt[:, None]
    )
    _, gv, ginv = _occupied(lat, lat.coords(P[gen]))
    vcnt = np.zeros(len(gv))
    vsum = np.zeros((len(gv), 3))
    for o in _OFFS27:
        j = _lookup(fkeys, lat.keys(gv + o))
        ok = j >= 0
        vcnt[ok] += cnt[j[ok]]
        vsum[ok] += csum[j[ok]]
    hv = np.flatnonzero(vcnt > 0)  # generated voxels next to the front
    if not len(hv):
        return sc
    _check(cancel)
    slot = np.full(len(gv), -1, np.int64)
    slot[hv] = np.arange(len(hv))
    rows = np.flatnonzero(slot[ginv] >= 0)
    g = gen[rows]
    vs = slot[ginv[rows]]
    d = np.full(len(rows), np.inf)
    for o in _OFFS27:
        j = _lookup(fkeys, lat.keys(gv[hv] + o))[vs]
        ok = np.flatnonzero(j >= 0)
        d[ok] = np.minimum(d[ok], np.linalg.norm(cen[j[ok]] - P[g[ok]], axis=1))
    w = 1.0 - smoothstep(0.0, r, d)
    upd = w > 0  # the others keep their colour exactly (even outside [0, 1])
    g, vs, w = g[upd], vs[upd], w[upd]
    c = col[g]
    cbar = vsum[hv] / vcnt[hv, None]
    out = np.array(sc.colors, dtype=np.float32, copy=True)
    out[g] = np.clip(c + w[:, None] * (cbar[vs] - c), 0.0, 1.0).astype(np.float32)
    return replace(sc, colors=out)


# --------------------------------------------------------------------------- floaters
def _row_hash(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(bits, hash)``: the float32 bit patterns of each row and a 64-bit mix of them."""
    bits = np.ascontiguousarray(means, dtype=np.float32).view(np.uint32).reshape(-1, 3)
    u = bits.astype(np.uint64)
    h = u[:, 0] * np.uint64(0x9E3779B97F4A7C15) ^ u[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F) ^ u[:, 2]
    return bits, h


def _match_rows(ref: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of a row of ``ref`` with bit-identical position for each ``query`` row, -1 if
    none (a hash collision only costs a miss: the caller has a fallback)."""
    out = np.full(len(query), -1, np.int64)
    if not len(ref) or not len(query):
        return out
    rb, rh = _row_hash(ref)
    qb, qh = _row_hash(query)
    order = np.argsort(rh, kind="stable")
    hs = rh[order]
    pos = np.minimum(np.searchsorted(hs, qh), len(hs) - 1)
    cand = order[pos]
    ok = (hs[pos] == qh) & (rb[cand] == qb).all(axis=1)
    out[ok] = cand[ok]
    return out


def _protected(
    sc: SplatCloud, is_front: np.ndarray, front: FrontModel, maps: _Maps, seam: float
) -> np.ndarray:
    """Front rows that may never be removed: theta < seam - 10 or in the outline band.

    Rows bit-identical to a ``front.sc`` row take its ``theta_deg`` / ``rim``; others (a
    decimated front) are looked up by their projection the way ``sharp_front`` does."""
    out = np.zeros(len(sc), bool)
    rows = np.flatnonzero(is_front)
    if not len(rows):
        return out
    _, theta_f, rim_f = _front_rows(front)
    P = np.asarray(sc.means)[rows]
    j = _match_rows(np.asarray(front.sc.means), P)
    hit = j >= 0
    theta = np.full(len(rows), 90.0)
    rim = np.zeros(len(rows), bool)
    theta[hit] = theta_f[j[hit]]
    rim[hit] = rim_f[j[hit]]
    miss = np.flatnonzero(~hit)
    if len(miss):
        cell = _cells(front, np.asarray(P[miss], dtype=np.float64))
        inside = cell >= 0
        cos_t = np.full(len(miss), np.nan)
        dt = np.zeros(len(miss))
        cos_t[inside] = maps.cos_t[cell[inside]]
        dt[inside] = maps.dt[cell[inside]]
        theta[miss] = angle_deg(cos_t)
        rim[miss] = dt <= maps.rim_cells
    out[rows] = (theta < seam - SEAM_BAND_DEG) | rim
    return out


def remove_floaters(
    sc: SplatCloud,
    is_front: np.ndarray,
    front: FrontModel,
    canon: Canon,
    *,
    hull: HullIntervals | None = None,
    seam_deg: float = SEAM_DEG,
    spacing_m: float | None = None,
    cancel: CancelToken | None = None,
    progress: ProgressReporter | None = None,
) -> tuple[SplatCloud, np.ndarray, dict[str, Any]]:
    """Design 3.10.  Generated Gaussians are dropped if opacity < 0.05, max scale > 0.035
    height, anisotropy > 25 with max scale > 0.012 height, outside the 1 %-dilated matte
    (projected), more than 2 cm outside the hull, or radius outliers (< 3 neighbours in the
    27 cells of 1.5 cm).  Then, for everything, 1.5 cm voxel components: keep components
    with >= 1 % of the opacity mass or within 5 cm of the largest.  Protected front rows
    (theta < seam - 10 or rim) are never removed.

    Details: anisotropy is largest / middle scale (a needle; flat discs, whose smallest
    axis is tiny by design, are fine).  "Height" is ``canon.height_s`` (the size rules are
    skipped when it is unknown).  The hull test uses the cell the centre projects to,
    where ``hull.valid``.  Neighbours are counted among the Gaussians that survive the
    other rules, front included.  "Largest" is by opacity mass and "within 5 cm" means a
    voxel centre of the component within ``max(5 cm, 1 voxel)`` of one of the largest
    component's.  ``spacing_m`` (an optional addition, ``BackPart.spacing_m``) is the
    typical distance between neighbouring generated Gaussians: the voxel edge becomes
    ``max(1.5 cm, 1.5 * spacing_m)`` (``diag["voxel_m"]``), so a back sampled more
    coarsely than 1.5 cm (a strided mould back of a subject several metres away) is not
    taken apart into "isolated" Gaussians and small components.  Rows with a non-finite
    position, scale or opacity are always dropped.  Protection (``seam_deg``, an optional
    addition) is looked up as in :func:`_protected`.  The order of the kept rows is
    unchanged.  ``diag`` holds JSON-safe counts per rule."""
    rep = _as_reporter(progress)
    n = _validate(sc)
    isf = _mask_arg(is_front, n)
    seam = _seam_arg(seam_deg)
    voxel = _FLOATER_VOXEL_M
    if spacing_m is not None:
        voxel = max(voxel, _SPACING_VOXELS * _positive_arg(spacing_m, "The back's Gaussian spacing"))
    near_offs = _sphere_offsets(max(_COMP_NEAR_M / voxel, 1.0))
    maps = _maps(front)
    rep(0.0, "Removing stray blobs")
    prot = _protected(sc, isf, front, maps, seam)
    finite = _finite_rows(sc)
    prot &= finite
    gen = ~isf & finite
    P = np.asarray(sc.means, dtype=np.float64)
    op = np.asarray(sc.opacity, dtype=np.float64)
    diag: dict[str, Any] = {
        "in": n,
        "protected": int(prot.sum()),
        "non_finite": int((~finite).sum()),
        "voxel_m": voxel,
    }

    # ---- per-Gaussian rules for generated Gaussians
    drop = ~finite
    g = np.flatnonzero(gen)
    rules: dict[str, np.ndarray] = {"low_opacity": op[g] < _MIN_OPACITY}
    height = float(canon.height_s)
    s = np.sort(np.abs(np.asarray(sc.scales, dtype=np.float64)[g]), axis=1)
    if math.isfinite(height) and height > 0:
        rules["too_big"] = s[:, 2] > _MAX_SCALE_FRAC * height
        with np.errstate(divide="ignore", invalid="ignore"):
            rules["needle"] = (s[:, 2] > _NEEDLE_RATIO * s[:, 1]) & (s[:, 2] > _NEEDLE_FRAC * height)
    else:
        log.warning("Floater filter: subject height unknown; size rules skipped")
    cell = _cells(front, P[g])
    inside = cell >= 0
    inside[inside] = maps.band[cell[inside]]
    rules["outside_matte"] = ~inside
    if hull is not None:
        shape = (int(front.grid.Hm), int(front.grid.Wm))
        z_in = np.asarray(hull.z_in, dtype=np.float64)
        z_out = np.asarray(hull.z_out, dtype=np.float64)
        hv = np.asarray(hull.valid, dtype=bool)
        if z_in.shape != shape or z_out.shape != shape or hv.shape != shape:
            raise InputError(f"The hull maps do not match the mould grid {shape}.")
        z_in, z_out, hv = z_in.ravel(), z_out.ravel(), hv.ravel()
        out_h = np.zeros(len(g), bool)
        k = np.flatnonzero(inside)
        ck = cell[k]
        z = P[g[k], 2]
        with np.errstate(invalid="ignore"):
            out_h[k] = hv[ck] & ((z < z_in[ck] - _HULL_SLACK_M) | (z > z_out[ck] + _HULL_SLACK_M))
        rules["outside_hull"] = out_h
    for name, bad in rules.items():
        diag[name] = int(bad.sum())
        drop[g[bad]] = True
    drop &= ~prot
    _check(cancel)
    rep(0.3, "Removing stray blobs")

    alive = np.flatnonzero(~drop)
    lat = _Lattice(voxel, [P[alive]], reach=int(np.abs(near_offs).max()))
    q = lat.coords(P[alive])
    keys, qv, inv = _occupied(lat, q)

    # ---- radius outliers among the generated survivors
    counts = np.bincount(inv, minlength=len(keys)).astype(np.int64)
    gv = np.unique(inv[gen[alive]])
    nb = np.zeros(len(keys), np.int64)
    for o in _OFFS27:
        j = _lookup(keys, lat.keys(qv[gv] + o))
        nb[gv] += np.where(j >= 0, counts[np.maximum(j, 0)], 0)
    lonely = gen[alive] & (nb[inv] - 1 < _MIN_NEIGHBOURS)
    diag["isolated"] = int(lonely.sum())
    if lonely.any():
        keep_a = ~lonely
        alive, q = alive[keep_a], q[keep_a]
        keys, qv, inv = _occupied(lat, q)
    _check(cancel)
    rep(0.55, "Removing stray blobs")

    # ---- connected components of everything that is left
    comp_drop = 0
    n_comp = 0
    if len(alive):
        roots = _voxel_roots(lat, keys, qv, cancel)
        vlab = np.unique(roots, return_inverse=True)[1].reshape(-1)  # per voxel, 0..K-1
        n_comp = int(vlab.max()) + 1
        plab = vlab[inv]
        mass = np.bincount(plab, weights=np.clip(op[alive], 0.0, 1.0), minlength=n_comp)
        big = int(np.argmax(mass))
        keep_c = mass >= _COMP_MASS_FRAC * mass.sum()
        keep_c[big] = True
        cand_v = np.flatnonzero(~keep_c[vlab])
        if len(cand_v):
            _check(cancel)
            big_keys = keys[vlab == big]  # sorted: a subsequence of the sorted keys
            near = np.zeros(len(cand_v), bool)
            for o in near_offs:
                near |= _lookup(big_keys, lat.keys(qv[cand_v] + o)) >= 0
            keep_c[vlab[cand_v[near]]] = True
        gone = ~keep_c[plab] & ~prot[alive]
        comp_drop = int(gone.sum())
        alive = alive[~gone]
    diag.update(components=n_comp, component_rows_dropped=comp_drop)
    keep = np.zeros(n, bool)
    keep[alive] = True
    keep |= prot
    diag.update(
        out=int(keep.sum()), front_dropped=int((isf & ~keep).sum()), gen_dropped=int((~isf & ~keep).sum())
    )
    log.info(
        "Floaters: %d of %d Gaussians removed (back %d, front %d; %d components)",
        n - diag["out"],
        n,
        diag["gen_dropped"],
        diag["front_dropped"],
        n_comp,
    )
    rep(1.0, "Stray blobs removed")
    return subset_splats(sc, keep), isf[keep], diag


# --------------------------------------------------------------------------- life size
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
    percentiles +- 3 cm, for ``crop_splats(aabb=box)``.

    The percentiles use the Gaussians with opacity > 0.5 (all finite ones when there are
    none).  ``height_m`` is only read when the height is used.  Raises
    :class:`InputError` for an unknown mode, a missing or non-positive height where it is
    needed, an empty cloud or a cloud with no height above the floor."""
    n = _validate(sc)
    if mode not in SIZE_MODES:
        raise InputError(f"Unknown size mode {mode!r}; use one of {', '.join(SIZE_MODES)}.")
    P = np.asarray(sc.means, dtype=np.float64)
    finite = np.isfinite(P).all(axis=1)
    rel = finite & (np.asarray(sc.opacity, dtype=np.float64) > _SIZE_OPACITY)
    if not rel.any():
        rel = finite
    if n == 0 or not rel.any():
        raise InputError("The 3D result is empty, so it cannot be sized.")
    h = float(np.percentile(P[rel, 1], _HEIGHT_PCT))
    if not (math.isfinite(h) and h > 1e-6):
        raise InputError(
            "The subject's height could not be measured (nothing stands above the floor).",
            hint="Try 'Size from: photo', or check the subject cut-out.",
        )
    used = mode
    if mode == "auto":
        used = "photo" if cut else "height"
        if cut:
            log.warning("Subject cut off by the photo border: keeping SHARP's metric size (%.2f m)", h)
    if used == "height":
        k = _positive_arg(height_m, "The subject's height (m)") / h
    else:
        k = 1.0
    out = scale_splats(sc, k)
    Ps = P[rel] * k
    h2 = h * k
    x_lo, x_hi = (float(t) for t in np.percentile(Ps[:, 0], _BOX_PCT))
    z_lo, z_hi = (float(t) for t in np.percentile(Ps[:, 2], _BOX_PCT))
    box = np.array(
        [
            [x_lo - _BOX_PAD_M, -_FLOOR_PAD_M, z_lo - _BOX_PAD_M],
            [x_hi + _BOX_PAD_M, h2 + _BOX_PAD_M, z_hi + _BOX_PAD_M],
        ]
    )
    log.info("Life size: measured %.3f m, scale %.4f (%s) -> %.3f m", h, k, used, h2)
    return out, h2, box, used


__all__ = [
    "MAX_SPLATS",
    "SEAM_BAND_DEG",
    "SEAM_DEG",
    "SIZE_MODES",
    "add_underlayer",
    "feather_colors",
    "merge_parts",
    "remove_floaters",
    "seam_weights",
    "smoothstep",
    "to_life_size",
    "voxel_components",
]
