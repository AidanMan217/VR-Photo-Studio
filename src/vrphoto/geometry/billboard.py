"""Walk-around "billboard" people/objects from one photo + matte (algorithm I).

The subject silhouette is inflated into a closed, rounded relief (like a pillow) so it
can be placed in passthrough AR and looked at from the side:

1. The matte (``alpha >= 0.5``, specks removed) is cropped to its bounding box and
   processed on a working grid (long side <= 320 px); every working pixel inside the
   silhouette is a vertex, triangles have all three vertices inside.
2. ``dt`` is the Euclidean distance to the silhouette boundary.  ``Rloc`` is the local
   thickness radius: the radius of the largest inscribed disc that contains the pixel,
   i.e. a grey dilation of ``dt`` with a *variable* (disc-sized) structuring element,
   computed by morphological openings at geometrically spaced radii.  (A fixed-size
   grey dilation would let the torso's radius leak into thin limbs.)
3. Inflation ``infl = kappa * mpp * sqrt(max(0, Rloc^2 - (Rloc - dt)^2))`` (a circular
   cross-section of radius ``Rloc`` scaled by ``kappa``), with ``kappa`` chosen so the
   median thickness over the vertices equals ``thickness * height_m`` (clamped to
   ``[0, 4]`` for degenerate silhouettes).
4. Detail: the high-passed disparity (Gaussian, sigma = 4 % of the silhouette height,
   normalised by its 99th percentile but never amplifying variations below 0.01) adds
   at most +-3 cm, tapered to zero at the boundary.
5. Front surface ``z = infl + detail`` (toward the viewer), back ``z = -0.9 * infl``.
   Boundary vertices have zero offset and identical positions in both meshes, so front
   and back close up without a gap.
6. Geometry is a near-orthographic reconstruction: the silhouette plane is scaled so the
   mesh is exactly ``height_m`` tall (``K`` provides the pixel aspect ``fy/fx``); feet
   (lowest vertex) at ``y = 0``, bounding box centred at ``x = z = 0``, front facing +z.
7. Textures: front = the photo crop as RGBA (alpha MASK; colours outside the matte are
   push-pull filled to avoid halos).  Back ``"diffuse"`` = push-pull fill of an 8 px band
   just inside the silhouette (at the <= 1024 px back-texture resolution), 10 % darker,
   so no face appears on the back of the head; ``"mirror"`` = the front texture seen
   from behind.  Both meshes use the same UVs and alpha MASK.

Returns ``[front, back]`` meshes in the glTF frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..core.errors import InputError
from ..core.fill import push_pull_fill
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, DepthResult, FloatMap, Intrinsics, Mesh, ProgressReporter
from .mesh import MeshError, _as_reporter, _check, _compact, _filter_components, _to_rgb_u8

log = get_logger(__name__)

BACK_MODES = ("diffuse", "mirror")
_WORK_LONG = 320  # working grid long side (pixels)
_BACK_TEX_LONG = 1024
_BAND_PX = 8
_BACK_DARKEN = 0.9
_BACK_RATIO = 0.9  # back inflation relative to the front
_DETAIL_M = 0.03
_KAPPA_MAX = 4.0
_MIN_SPECK_FRAC = 0.005


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Drop connected specks smaller than 0.5 % of the matte area (and < 16 px)."""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= max(16.0, _MIN_SPECK_FRAC * float(areas.sum()))
    if not keep.any():
        keep[1 + int(np.argmax(areas))] = True
    return keep[lab]


def _subject_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box ``(x0, y0, x1, y1)`` of the matte (``>= 0.5``) ignoring specks, plus a
    2 % margin.  Specks are found on a strided proxy (long side <= 1024 px) so 24 MP
    mattes need no full-resolution label image."""
    h, w = alpha.shape
    k = max(1, math.ceil(max(h, w) / 1024))
    with np.errstate(invalid="ignore"):
        small = np.asarray(alpha[::k, ::k]) >= 0.5
    if not small.any() and k > 1:
        with np.errstate(invalid="ignore"):
            small, k = np.asarray(alpha) >= 0.5, 1  # thin subject missed by the proxy
    if not small.any():
        raise InputError("The subject mask is empty; select or segment the subject first.")
    ys, xs = np.nonzero(_clean_mask(small))
    bx0, bx1 = int(xs.min()) * k, (int(xs.max()) + 1) * k
    by0, by1 = int(ys.min()) * k, (int(ys.max()) + 1) * k
    margin = int(round(0.02 * max(bx1 - bx0, by1 - by0))) + 2 + k
    return max(0, bx0 - margin), max(0, by0 - margin), min(w, bx1 + margin), min(h, by1 + margin)


def _bleed(rgb: np.ndarray, known: np.ndarray, max_px: int = 1_500_000) -> np.ndarray:
    """Replace the colours outside the matte by a push-pull fill of the inside colours (no
    background halo when the texture is filtered).  Large crops are filled on a reduced
    grid from fully-inside pixels only and upsampled, which keeps 24 MP inputs cheap."""
    if known.all():
        return rgb.copy()
    h, w = known.shape
    s = min(1.0, math.sqrt(max_px / float(h * w)))
    if s >= 1.0:
        return push_pull_fill(rgb, known)
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    inside = cv2.resize(known.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) >= 255
    fill = push_pull_fill(small, inside) if inside.any() else small
    up = cv2.resize(fill, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.where(known[..., None], rgb, up)


def _dist(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) of inside pixels to the nearest outside pixel; image
    borders count as outside."""
    pad = np.pad(mask.astype(np.uint8), 1)
    return cv2.distanceTransform(pad, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]


def local_thickness(dt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``Rloc``: radius of the largest disc inside the shape that covers each pixel.

    ``dt`` is the distance to the boundary.  For each radius ``r`` the opening of the
    shape by a disc of radius ``r`` is ``{p : dist(p, {dt >= r}) <= r}``; ``Rloc`` is the
    largest ``r`` whose opening contains the pixel (>= ``dt`` everywhere).
    """
    rloc = np.where(mask, dt, 0.0).astype(np.float32)
    dmax = float(dt[mask].max()) if mask.any() else 0.0
    r = 1.0
    while r <= dmax:
        core = dt >= r
        if not core.any():
            break
        d = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        rloc[(d <= r) & mask] = r
        r = max(r + 1.0, r * 1.12)
    return np.maximum(rloc, np.where(mask, dt, 0.0)).astype(np.float32)


def inflation_profile(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inflation profile of a silhouette (step 3 above, without ``kappa`` / ``mpp``).

    ``dt_eff = max(dt - 1, 0)`` is the distance (pixels) to the boundary, with image borders
    counting as outside, minus the boundary pixel itself; ``prof = sqrt(max(dt_eff * (2 *
    Rloc - dt_eff), 0))`` is the height of a circular cross-section of radius ``Rloc``
    (:func:`local_thickness` of ``dt_eff``) over the pixel, in pixels.  Both are full-grid
    float32 maps, 0 outside ``mask``.  Also used by the SHARP 360 shell back on its mould
    grid, so a change here changes both."""
    m = np.asarray(mask, dtype=bool)
    dt = _dist(m)
    dt_eff = np.maximum(dt - 1.0, 0.0)  # boundary pixels (dt ~ 1) are the silhouette itself
    rloc = local_thickness(dt_eff, m)
    prof = np.sqrt(np.maximum(dt_eff * (2.0 * rloc - dt_eff), 0.0))
    prof[~m] = 0.0
    return prof.astype(np.float32, copy=False), dt_eff.astype(np.float32, copy=False)


def _edge_counts(faces: np.ndarray, n: int) -> np.ndarray:
    """Per face and edge (M x 3): how many faces use that edge."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    key = e[:, 0] * n + e[:, 1]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inv].reshape(3, -1).T


def _prune_ears(faces: np.ndarray, n: int, passes: int = 2) -> np.ndarray:
    """Remove triangles with two or more boundary edges (staircase corners / slivers).

    Such an "ear" has all three vertices on the silhouette, so its front and back copies
    would coincide and pinch the closed surface into a non-manifold edge."""
    for _ in range(passes):
        if len(faces) == 0:
            break
        ears = (_edge_counts(faces, n) == 1).sum(axis=1) >= 2
        if not ears.any():
            break
        faces = faces[~ears]
    return faces


def _boundary_vertices(faces: np.ndarray, n: int) -> np.ndarray:
    """Vertices on edges used by exactly one face."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    key = e[:, 0] * n + e[:, 1]
    uniq, counts = np.unique(key, return_counts=True)
    b = uniq[counts == 1]
    out = np.zeros(n, bool)
    out[b // n] = True
    out[b % n] = True
    return out


def _masked_blur(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    mf = m.astype(np.float32)
    num = cv2.GaussianBlur(x.astype(np.float32) * mf, (0, 0), sigma)
    den = cv2.GaussianBlur(mf, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def billboard(
    img: RGB,
    alpha: FloatMap,
    depth: DepthResult,
    K: Intrinsics,
    *,
    height_m: float = 1.7,
    thickness: float = 0.17,
    back: str = "diffuse",
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> list[Mesh]:
    """Inflate the matted subject into a closed walk-around mesh (algorithm I).

    ``alpha`` is the subject matte (float [0, 1]), ``depth`` the depth estimate of the
    whole photo (its high-passed disparity adds surface detail) and ``K`` the camera.
    Returns ``[front, back]`` meshes in the glTF frame: feet at ``y = 0``, bounding box
    centred on the y axis, height ``height_m``, front texture facing +z, median thickness
    ``thickness * height_m``.  ``back`` is ``"diffuse"`` (no face on the back) or
    ``"mirror"``.  Raises :class:`InputError` for an empty matte or bad parameters.
    """
    rep = _as_reporter(progress)
    if back not in BACK_MODES:
        raise InputError(f"Unknown billboard back texture {back!r}; use 'diffuse' or 'mirror'.")
    if not (math.isfinite(height_m) and height_m > 0):
        raise InputError(f"height_m must be a positive height in metres (got {height_m!r}).")
    if not (math.isfinite(thickness) and thickness >= 0):
        raise InputError(f"thickness must be a non-negative ratio (got {thickness!r}).")
    rgb = _to_rgb_u8(img)[..., :3]
    h, w = rgb.shape[:2]
    a = np.asarray(alpha)
    if a.shape != (h, w):
        size = f"{a.shape[1]}x{a.shape[0]}" if a.ndim == 2 else f"shape {a.shape}"
        raise InputError(f"The subject mask ({size}) must match the image ({w}x{h}).")
    if depth is None or not hasattr(depth, "disparity"):
        raise InputError("A depth estimate (DepthResult) is required to build the 3D cut-out.")
    disp = np.asarray(depth.disparity)
    if disp.ndim != 2 or disp.size == 0:
        raise InputError("The depth map must be a 2-D array.")
    fx = float(K.fx) if K.fx > 0 else 1.0
    fy = float(K.fy) if K.fy > 0 else fx
    _check(cancel)

    rep(0.05, "Preparing silhouette")
    x0, y0, x1, y1 = _subject_bbox(a)
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_a = np.clip(np.nan_to_num(np.asarray(a[y0:y1, x0:x1], dtype=np.float32)), 0.0, 1.0)
    if disp.shape != (h, w):  # depth at another resolution: resample to the image grid
        disp = cv2.resize(np.asarray(disp, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    crop_d = np.nan_to_num(np.asarray(disp[y0:y1, x0:x1], dtype=np.float32))
    ch, cw = crop_a.shape
    s = min(1.0, _WORK_LONG / max(ch, cw))
    ww, wh = max(2, int(round(cw * s))), max(2, int(round(ch * s)))
    sx, sy = cw / ww, ch / wh  # full-res pixels per working pixel
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    a_w = cv2.resize(crop_a, (ww, wh), interpolation=interp) if (ww, wh) != (cw, ch) else crop_a
    d_w = cv2.resize(crop_d, (ww, wh), interpolation=interp) if (ww, wh) != (cw, ch) else crop_d
    m_w = _clean_mask(a_w >= 0.5)
    _check(cancel)

    # ---- triangulate the silhouette on the working grid
    rep(0.2, "Triangulating")
    k1 = m_w[:-1, :-1] & m_w[:-1, 1:] & m_w[1:, :-1]
    k2 = m_w[:-1, 1:] & m_w[1:, :-1] & m_w[1:, 1:]
    k1, k2 = _filter_components(k1, k2, 8)
    if not (k1.any() or k2.any()):
        raise InputError("The subject is too small or too thin to build a 3D model from.")
    idx = np.arange(wh * ww, dtype=np.int64).reshape(wh, ww)
    ia, ib, ic, idd = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tri = np.stack([np.stack([ia, ic, ib], -1), np.stack([ib, ic, idd], -1)], 2)
    faces_full = _prune_ears(tri[np.stack([k1, k2], 2)], wh * ww)
    if len(faces_full) == 0:
        raise InputError("The subject is too small or too thin to build a 3D model from.")
    used, faces = _compact(wh * ww, faces_full)
    rows = (np.arange(wh * ww) // ww)[used]
    cols = (np.arange(wh * ww) % ww)[used]
    nv = int(used.sum())
    bnd = _boundary_vertices(faces, nv)
    used_grid = np.zeros((wh, ww), bool)
    used_grid.ravel()[used] = True
    _check(cancel)

    # ---- inflation (algorithm I)
    rep(0.35, "Inflating")
    prof_map, dt_eff = inflation_profile(used_grid)
    prof = prof_map[rows, cols]
    prof[bnd] = 0.0
    r_min, r_max = int(rows.min()), int(rows.max())
    if r_max <= r_min:
        raise InputError("The subject is too small to build a 3D model from.")
    mpp_y = float(height_m) / ((r_max - r_min) * sy)  # metres per full-res pixel (vertical)
    mpp_x = mpp_y * fy / fx
    med = float(np.median(prof))
    target_t = float(thickness) * float(height_m)
    kappa = 0.0
    if med > 1e-9 and target_t > 0:
        kappa = target_t / ((1.0 + _BACK_RATIO) * med * sy * mpp_y)
        if kappa > _KAPPA_MAX:
            log.info("Billboard inflation clamped (kappa %.2f -> %.1f)", kappa, _KAPPA_MAX)
            kappa = _KAPPA_MAX
    infl = kappa * prof * sy * mpp_y
    _check(cancel)

    # ---- detail from the high-passed disparity
    sil_h = max(2.0, float(r_max - r_min))
    lp = _masked_blur(d_w, used_grid, max(1.0, 0.04 * sil_h))
    hp = (d_w - lp)[rows, cols].astype(np.float64)
    inside = ~bnd
    detail = np.zeros(nv)
    if inside.any() and target_t > 0:
        scale = max(float(np.percentile(np.abs(hp[inside]), 99)), 0.01)
        taper = np.clip(dt_eff[rows, cols] / 3.0, 0.0, 1.0)
        detail = _DETAIL_M * np.clip(hp / scale, -1.0, 1.0) * taper
        detail[bnd] = 0.0
    off_front = np.maximum(infl + detail, 0.2 * infl)
    off_back = _BACK_RATIO * infl

    # ---- assemble the glTF-frame geometry
    rep(0.55, "Building meshes")
    x = cols * sx * mpp_x
    y = (r_max - rows) * sy * mpp_y
    front_v = np.column_stack([x, y, off_front])
    back_v = np.column_stack([x, y, -off_back])
    cx = 0.5 * (x.min() + x.max())
    cz = 0.5 * (min(front_v[:, 2].min(), back_v[:, 2].min()) + max(front_v[:, 2].max(), back_v[:, 2].max()))
    for v in (front_v, back_v):
        v[:, 0] -= cx
        v[:, 2] -= cz
    uv = np.column_stack([(cols + 0.5) / ww, (rows + 0.5) / wh]).astype(np.float32)
    _check(cancel)

    # ---- textures
    rep(0.7, "Texturing")
    front_rgb = _bleed(crop_rgb, crop_a >= 0.5)
    a8 = (crop_a * 255.0 + 0.5).astype(np.uint8)
    front_tex = np.dstack([front_rgb, a8])
    if back == "mirror":
        back_tex = front_tex
    else:
        bs = min(1.0, _BACK_TEX_LONG / max(ch, cw))
        bw_, bh_ = max(1, int(round(cw * bs))), max(1, int(round(ch * bs)))
        if (bw_, bh_) != (cw, ch):
            small = cv2.resize(crop_rgb, (bw_, bh_), interpolation=cv2.INTER_AREA)
            small_a = cv2.resize(crop_a, (bw_, bh_), interpolation=cv2.INTER_AREA)
        else:
            small, small_a = crop_rgb, crop_a
        inside_b = small_a >= 0.5
        band = inside_b & (_dist(inside_b) <= _BAND_PX)
        if not band.any():
            band = inside_b
        fill = push_pull_fill(small, band) if band.any() else small
        fill = np.clip(fill.astype(np.float32) * _BACK_DARKEN + 0.5, 0, 255).astype(np.uint8)
        back_tex = np.dstack([fill, (np.clip(small_a, 0, 1) * 255.0 + 0.5).astype(np.uint8)])
    _check(cancel)

    f32 = np.float32
    front = Mesh(
        vertices=front_v.astype(f32),
        faces=faces.astype(np.uint32),
        uvs=uv,
        texture=front_tex,
        name="front",
        alpha_mode="MASK",
        alpha_cutoff=0.5,
    )
    back_mesh = Mesh(
        vertices=back_v.astype(f32),
        faces=faces[:, [0, 2, 1]].astype(np.uint32),
        uvs=uv.copy(),
        texture=back_tex,
        name="back",
        alpha_mode="MASK",
        alpha_cutoff=0.5,
    )
    if len(front.faces) == 0:  # pragma: no cover - guarded above
        raise MeshError("The billboard has no faces.")
    rep(1.0, "Billboard ready")
    log.info(
        "Billboard: %d vertices, %d faces per side, kappa %.2f, %.3f m x %.3f m",
        nv,
        len(faces),
        kappa,
        float(x.max() - x.min()),
        float(height_m),
    )
    return [front, back_mesh]


__all__ = ["billboard", "inflation_profile", "local_thickness", "BACK_MODES"]
