"""Textured triangle meshes from depth maps and layered depth images (algorithm H).

Pipeline (docs/CONTRACTS.md, algorithm H):

1. **Grid.**  Vertices sit on a regular grid of pixel centres with spacing derived from
   the contract stride ``s = sqrt(W*H / (target/2))``.  Instead of rounding ``s`` up to
   an integer (which can undershoot the face budget by 4x), the grid uses
   ``round((W-1)/s) + 1`` columns at (nearly) uniform integer pixel positions, so a full
   frame gives ``2*(nx-1)*(ny-1) ~= target_faces`` triangles.  The outermost vertex
   rows/columns are moved half a pixel outwards to the image border so the mesh covers
   the whole photo (UVs span exactly [0, 1]).  Depth and alpha are sampled at the
   vertex pixel (nearest, never interpolated across depth edges).
2. **Cuts.**  A triangle is dropped when ``fx * ref_baseline_m * (max inv_z - min inv_z)
   > cut_px`` over its three vertices (it would tear by more than ``cut_px`` pixels for a
   ``ref_baseline_m`` head movement), or when any vertex has alpha < 0.5 or an invalid
   depth.
3. **Debris.**  Vertex-connected components with fewer than 50 faces are removed.  The
   50-face threshold is defined for the 300k reference budget and scales with
   ``target_faces`` (constant image-area fraction), clamped to [8, 50] (and below 8 for
   grids of < 400 faces).  If *every* component is below the threshold (a tiny matte)
   they are all kept instead of failing.
4. **Simplification** (optional, when ``fast_simplification`` is importable): the front
   surface is first built at 2x density per axis (4x faces) and then reduced to
   ``target_faces`` with quadric edge collapse *in screen-parallax space*
   ``(u_px, v_px, fx*ref_baseline_m*inv_z)``.  In that space planar surfaces stay planar
   (inverse depth is affine in screen space for a plane) and the error metric is in
   pixels of parallax, consistent with the cut rule.  Cut borders are preserved, so the
   silhouettes keep the dense-grid resolution.  This is why decimation happens here,
   *before* the glTF conversion: the simplified vertices are unprojected with the
   source intrinsics and their UVs are their exact projection into the source camera.
   The rare slivers that a collapse folds over their neighbours (back-facing from the
   camera, so invisible) are dropped, keeping every face front-facing.
   Without ``fast_simplification`` the regular grid at the target density is used.
5. **UVs** are the projection of each vertex into the source camera,
   ``uv = ((u + 0.5) / W, (v + 0.5) / H)`` (glTF: (0, 0) = top-left).
6. **Frame.**  Vertices are unprojected with ``K`` and converted to the glTF frame with
   :func:`geometry.camera.cv_to_gltf` (the source camera sits at the origin looking
   down -z).

:func:`layered_meshes` additionally builds, for the background layer, a detail mesh
restricted to the hallucinated region (where the background layer differs from the
front layer, plus where the front layer is transparent) and a coarse backdrop at 4x the
stride, pushed back 1 %, whose vertex depths are the farthest depth of the adjacent
cells so it lies behind every real surface (see :func:`layered_meshes`).  Its
``target_faces`` is the budget of the whole scene (all layers together), split by the
area each layer covers.

:func:`decimate` reduces an arbitrary glTF-frame :class:`Mesh`; UVs of the new vertices
come from a projective camera fitted to the input (exact for depth meshes and
billboards), falling back to the UV of the nearest collapsed source vertex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from ..core.errors import InputError, VRPhotoError
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, FloatMap, Intrinsics, Layer, LayeredImage, Mesh, ProgressReporter
from .camera import cv_to_gltf

log = get_logger(__name__)

MIN_COMPONENT_FACES = 50  # contract value at the reference budget
_REF_TARGET_FACES = 300_000
_MIN_COMPONENT_FLOOR = 8
BACKDROP_PUSH = 1.01  # backdrop pushed back 1 % along the viewing rays
DETAIL_PUSH = 1.002  # background detail pushed back 0.2 % (no z-fighting with the front)
_OVERSAMPLE = 2  # per-axis grid oversampling before quadric simplification
_ANCHOR_WINDOW = 0.05  # half-size of the central window used for the anchor depth (fraction)
#: Face density of background detail meshes relative to the front mesh (they are only
#: seen in the slivers revealed by head movement, and show inpainted, smooth content).
BACKGROUND_DENSITY = 0.5
_BACKDROP_SHARE = 16  # backdrop budget: target_faces / 16 (4x the stride of a full-frame grid)
_MIN_INV_Z = 1e-6
_RGB_DIFF = 12  # colour difference (0-255) that marks an inpainted background pixel


class MeshError(VRPhotoError):
    """Mesh generation failed (for example nothing is left to mesh after the cuts)."""


# --------------------------------------------------------------------------- shared helpers
def _as_reporter(progress: Any) -> ProgressReporter:
    """Normalise ``progress`` (None, ProgressReporter or callable(frac, msg))."""
    if progress is None:
        return ProgressReporter.null()
    if isinstance(progress, ProgressReporter):
        return progress
    if callable(progress):
        return ProgressReporter(progress)
    raise InputError("progress must be a ProgressReporter or a callable(frac, msg).")


def _check(cancel: CancelToken | None) -> None:
    if cancel is not None:
        cancel.check()


def _fast_simplification() -> Any | None:
    """Return the ``fast_simplification`` module or None when it is not installed."""
    try:
        import fast_simplification  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return fast_simplification


def _to_rgb_u8(img: Any, what: str = "image") -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise InputError(f"The {what} must be an HxWx3 RGB array (got shape {a.shape}).")
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = (np.clip(np.nan_to_num(a), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        elif a.dtype == np.uint16:  # 16-bit photo: rescale, do not saturate
            a = ((a.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def _qem(fs: Any, points: np.ndarray, faces: np.ndarray, target: int, *, collapses: bool = False) -> tuple:
    """Quadric edge-collapse simplification with border preservation when supported."""
    p = np.ascontiguousarray(points, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    kw: dict[str, Any] = {"target_count": int(target)}
    if collapses:
        kw["return_collapses"] = True
    try:
        return fs.simplify(p, f, preserve_border=True, **kw)
    except TypeError:  # fast-simplification < 0.2 has no preserve_border
        return fs.simplify(p, f, **kw)


def _clean_faces(faces: np.ndarray) -> np.ndarray:
    """Drop faces with repeated vertex indices."""
    f = np.asarray(faces)
    if f.size == 0:
        return f.reshape(0, 3)
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return f[ok]


def _compact(n_vertices: int, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (used-vertex boolean mask, faces re-indexed onto the used vertices)."""
    used = np.zeros(n_vertices, dtype=bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used, dtype=np.int64) - 1
    return used, remap[faces]


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals (``(0, 0, 1)`` for isolated vertices)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        for c in range(3):
            n[:, c] += np.bincount(f[:, k], weights=fn[:, c], minlength=len(v))
    ln = np.linalg.norm(n, axis=1)
    bad = ln < 1e-20
    n[bad] = (0.0, 0.0, 1.0)
    ln[bad] = 1.0
    return (n / ln[:, None]).astype(np.float32)


# --------------------------------------------------------------------------- grid meshing
@dataclass
class _Grid:
    """Vertex grid: sample pixel indices and (border-extended) pixel positions."""

    xi: np.ndarray  # int64 (nx,) sampled pixel column per vertex column
    yi: np.ndarray  # int64 (ny,)
    xp: np.ndarray  # float64 (nx,) vertex u coordinate (pixel-centre convention)
    yp: np.ndarray  # float64 (ny,)

    @property
    def nx(self) -> int:
        return int(self.xi.size)

    @property
    def ny(self) -> int:
        return int(self.yi.size)


@dataclass
class _ScreenMesh:
    """Mesh in screen space: columns (u_px, v_px, inv_z)."""

    pts: np.ndarray  # float64 (N, 3)
    faces: np.ndarray  # int64 (M, 3)


def _axis(n_px: int, n_vert: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, n_px - 1.0, n_vert)
    xi = np.floor(xs + 0.5).astype(np.int64)
    xp = xi.astype(np.float64)
    xp[0] = -0.5
    xp[-1] = n_px - 0.5
    return xi, xp


def _make_grid(width: int, height: int, target_faces: float, oversample: float = 1.0) -> _Grid:
    """Grid whose full-frame triangulation has ~``target_faces * oversample**2`` faces."""
    if width < 2 or height < 2:
        raise InputError(f"The image is too small to mesh ({width}x{height} pixels).")
    target = max(2.0, float(target_faces))
    s = math.sqrt((width - 1) * (height - 1) * 2.0 / target) / max(oversample, 1e-6)
    s = max(s, 1.0)
    nx = int(min(width, max(2, round((width - 1) / s) + 1)))
    ny = int(min(height, max(2, round((height - 1) / s) + 1)))
    xi, xp = _axis(width, nx)
    yi, yp = _axis(height, ny)
    return _Grid(xi, yi, xp, yp)


def _min_component_faces(target_faces: int, oversample: float = 1.0, grid_faces: int | None = None) -> int:
    """Debris threshold (faces): 50 at the 300k reference budget, scaled with the budget
    and clamped to [8, 50]; for tiny grids (< 400 faces) it shrinks further so a small
    preview is never discarded as "debris"."""
    base = int(round(MIN_COMPONENT_FACES * target_faces / _REF_TARGET_FACES))
    base = min(MIN_COMPONENT_FACES, max(_MIN_COMPONENT_FLOOR, base))
    out = int(round(base * oversample * oversample))
    if grid_faces is not None:
        out = min(out, max(1, int(grid_faces) // 50))
    return out


def _filter_components(k1: np.ndarray, k2: np.ndarray, min_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove vertex-connected components with fewer than ``min_faces`` triangles.

    ``k1``/``k2`` flag the kept upper-left ``(a, c, b)`` and lower-right ``(b, c, d)``
    triangles of each grid cell.  Components are labelled on a 2x-upsampled lattice
    (vertices at even, edge midpoints at odd coordinates) with 4-connectivity, which
    connects exactly the triangles that share a vertex.
    """
    if min_faces <= 1 or not (k1.any() or k2.any()):
        return k1, k2
    gh, gw = k1.shape
    img = np.zeros((2 * gh + 1, 2 * gw + 1), np.uint8)
    u1 = k1.astype(np.uint8)
    u2 = k2.astype(np.uint8)
    u12 = u1 | u2
    img[0:-1:2, 0:-1:2] |= u1  # a
    img[0:-1:2, 2::2] |= u12  # b
    img[2::2, 0:-1:2] |= u12  # c
    img[2::2, 2::2] |= u2  # d
    img[0:-1:2, 1::2] |= u1  # ab
    img[1::2, 0:-1:2] |= u1  # ac
    img[1::2, 1::2] |= u12  # bc (diagonal, shared by both triangles of the cell)
    img[1::2, 2::2] |= u2  # bd
    img[2::2, 1::2] |= u2  # cd
    n, labels = cv2.connectedComponents(img, connectivity=4, ltype=cv2.CV_32S)
    lab = labels[1::2, 1::2]
    counts = np.bincount(lab[k1], minlength=n) + np.bincount(lab[k2], minlength=n)
    small = counts < min_faces
    small[0] = True
    drop = small[lab]
    return k1 & ~drop, k2 & ~drop


def _grid_mesh(
    w_v: np.ndarray,
    valid_v: np.ndarray,
    grid: _Grid,
    thr: float | None,
    *,
    region: np.ndarray | None = None,
    min_faces: int = 0,
) -> _ScreenMesh | None:
    """Triangulate a sampled vertex grid with the cut / validity / region rules."""
    ny, nx = w_v.shape
    va, vb, vc, vd = valid_v[:-1, :-1], valid_v[:-1, 1:], valid_v[1:, :-1], valid_v[1:, 1:]
    k1 = va & vb & vc
    k2 = vb & vc & vd
    if thr is not None:
        wa, wb, wc, wd = w_v[:-1, :-1], w_v[:-1, 1:], w_v[1:, :-1], w_v[1:, 1:]
        bc_max = np.maximum(wb, wc)
        bc_min = np.minimum(wb, wc)
        k1 &= (np.maximum(bc_max, wa) - np.minimum(bc_min, wa)) <= thr
        k2 &= (np.maximum(bc_max, wd) - np.minimum(bc_min, wd)) <= thr
    if region is not None:
        k1 &= region
        k2 &= region
    if not (k1.any() or k2.any()):
        return None
    f1, f2 = _filter_components(k1, k2, min_faces)
    if f1.any() or f2.any():
        k1, k2 = f1, f2
    else:  # every piece is "debris" (e.g. a tiny matte): keep them rather than fail
        log.info("All mesh components are below %d faces; keeping them", min_faces)
    idx = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tri = np.stack([np.stack([a, c, b], -1), np.stack([b, c, d], -1)], 2)
    faces = tri[np.stack([k1, k2], 2)]
    used, faces = _compact(ny * nx, faces)
    uu = np.broadcast_to(grid.xp[None, :], (ny, nx)).ravel()[used]
    vv = np.broadcast_to(grid.yp[:, None], (ny, nx)).ravel()[used]
    pts = np.stack([uu, vv, w_v.ravel()[used]], axis=1)
    return _ScreenMesh(pts, faces)


def _sample(arr: np.ndarray, grid: _Grid) -> np.ndarray:
    return np.asarray(arr)[np.ix_(grid.yi, grid.xi)]


def _vertex_samples(
    inv_z: np.ndarray, alpha: np.ndarray | None, grid: _Grid
) -> tuple[np.ndarray, np.ndarray]:
    w_v = _sample(inv_z, grid).astype(np.float64)
    valid = np.isfinite(w_v) & (w_v > _MIN_INV_Z)
    if alpha is not None:
        valid &= _sample(alpha, grid) >= 0.5
    w_v[~valid] = 0.0
    return w_v, valid


def _simplify_screen(sm: _ScreenMesh, target: int, dscale: float, width: int, height: int) -> _ScreenMesh:
    """Quadric simplification in (u, v, dscale*inv_z) space; no-op when not needed."""
    if len(sm.faces) <= target:
        return sm
    fs = _fast_simplification()
    if fs is None:
        return sm
    p = sm.pts.copy()
    p[:, 2] *= dscale
    out_p, out_f = _qem(fs, p, sm.faces, target)[:2]
    out_p = np.asarray(out_p, dtype=np.float64)
    out_f = _clean_faces(np.asarray(out_f, dtype=np.int64))
    if len(out_f) == 0:
        log.warning("Mesh simplification produced no faces; keeping the dense mesh")
        return sm
    out_p[:, 2] /= dscale
    w_lo, w_hi = float(sm.pts[:, 2].min()), float(sm.pts[:, 2].max())
    out_p[:, 2] = np.clip(out_p[:, 2], w_lo, w_hi)
    out_p[:, 0] = np.clip(out_p[:, 0], -0.5, width - 0.5)
    out_p[:, 1] = np.clip(out_p[:, 1], -0.5, height - 0.5)
    # Quadric collapses can (rarely, on noisy depth) fold a sliver over its neighbours.
    # Seen from the source camera such a triangle is back-facing - culled, and the
    # neighbours already cover its pixels - so dropping it changes nothing visible but
    # restores the invariant that every face is front-facing toward the camera.
    pa, pb, pc = out_p[out_f[:, 0]], out_p[out_f[:, 1]], out_p[out_f[:, 2]]
    signed = (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
    folded = signed >= 0.0  # grid triangles (a, c, b) have negative signed area (v down)
    if folded.any():
        log.debug("Dropping %d folded triangles after simplification", int(folded.sum()))
        out_f = out_f[~folded]
        if len(out_f) == 0:
            return sm
    used, out_f = _compact(len(out_p), out_f)
    return _ScreenMesh(out_p[used], out_f)


def _finalize(sm: _ScreenMesh, K: Intrinsics, push: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a screen mesh to source-camera points (float64) and UVs (float32)."""
    u, v, w = sm.pts[:, 0], sm.pts[:, 1], sm.pts[:, 2]
    z = push / w
    pts = np.empty((len(u), 3), np.float64)
    pts[:, 0] = (u - K.cx) / K.fx * z
    pts[:, 1] = (v - K.cy) / K.fy * z
    pts[:, 2] = z
    uv = np.empty((len(u), 2), np.float32)
    uv[:, 0] = (u + 0.5) / K.width
    uv[:, 1] = (v + 0.5) / K.height
    return pts, uv


def _front_screen_mesh(
    inv_z: np.ndarray,
    alpha: np.ndarray | None,
    K: Intrinsics,
    target_faces: int,
    cut_px: float,
    ref_baseline_m: float,
    rep: ProgressReporter,
    cancel: CancelToken | None,
    density: int | None = None,
) -> _ScreenMesh | None:
    """Cut grid mesh of a full layer, adaptively simplified when possible.

    ``density`` is the full-frame face count of the grid (default ``target_faces``); the
    result has at most ~``target_faces`` faces (a matte covering part of the frame at
    ``density`` gives ``density * coverage`` faces before simplification).
    """
    width, height = K.width, K.height
    dens = int(target_faces if density is None else max(2, int(density)))
    over = float(_OVERSAMPLE) if _fast_simplification() is not None else 1.0
    grid = _make_grid(width, height, dens, over)
    if over > 1.0 and 2 * (grid.nx - 1) * (grid.ny - 1) <= dens * 1.05:
        over = 1.0  # image too small to oversample meaningfully
    rep(0.05, "Sampling depth")
    w_v, valid = _vertex_samples(inv_z, alpha, grid)
    _check(cancel)
    dscale = K.fx * ref_baseline_m
    thr = cut_px / dscale
    rep(0.2, "Building mesh")
    grid_faces = 2 * (grid.nx - 1) * (grid.ny - 1)
    sm = _grid_mesh(w_v, valid, grid, thr, min_faces=_min_component_faces(dens, over, grid_faces))
    _check(cancel)
    if sm is None:
        return None
    if over > 1.0 and len(sm.faces) > target_faces:
        rep(0.4, "Simplifying mesh")
        sm = _simplify_screen(sm, target_faces, dscale, width, height)
        _check(cancel)
    rep(1.0, "Mesh ready")
    return sm


# --------------------------------------------------------------------------- validation
def _face_budget(target_faces: Any, minimum: int = 2) -> int:
    """Validate a face budget: an integer (or integral float such as ``3e5``) >= ``minimum``."""
    ok = isinstance(target_faces, (int, np.integer)) and not isinstance(target_faces, bool)
    if not ok and isinstance(target_faces, (float, np.floating)):
        ok = math.isfinite(target_faces) and float(target_faces).is_integer()
    if not ok or int(target_faces) < minimum:
        raise InputError(f"target_faces must be an integer >= {minimum} (got {target_faces!r}).")
    return int(target_faces)


def _validate_params(target_faces: Any, cut_px: float, ref_baseline_m: float) -> None:
    _face_budget(target_faces)
    if not (isinstance(cut_px, (int, float, np.number)) and cut_px > 0):
        raise InputError(f"cut_px must be a positive number of pixels (got {cut_px!r}).")
    if not (
        isinstance(ref_baseline_m, (int, float, np.number))
        and math.isfinite(ref_baseline_m)
        and ref_baseline_m > 0
    ):
        raise InputError(f"ref_baseline_m must be a positive distance in metres (got {ref_baseline_m!r}).")


def _validate_layer(
    img: Any, inv_z: Any, K: Intrinsics, alpha: Any | None, what: str = "image"
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rgb = _to_rgb_u8(img, what)
    h, w = rgb.shape[:2]
    if (K.width, K.height) != (w, h):
        raise InputError(f"Camera intrinsics are for {K.width}x{K.height} but the {what} is {w}x{h}.")
    z = np.asarray(inv_z)
    if z.shape != (h, w):
        raise InputError(f"Depth map shape {z.shape} does not match the {what} size {w}x{h}.")
    a = None
    if alpha is not None:
        a = np.asarray(alpha)
        if a.shape != (h, w):
            raise InputError(f"Alpha mask shape {a.shape} does not match the {what} size {w}x{h}.")
    if rgb.shape[2] == 4:
        if a is None:
            a = rgb[..., 3].astype(np.float32) / 255.0
        rgb = rgb[..., :3]
    return rgb, z, a


def _layer_texture(rgb: np.ndarray, alpha: np.ndarray | None) -> tuple[np.ndarray, str]:
    """RGB texture (OPAQUE), or RGBA + MASK when the alpha has transparent pixels."""
    if alpha is None:
        return rgb, "OPAQUE"
    a = np.asarray(alpha, dtype=np.float32)
    if float(np.nanmin(a)) >= 0.5:
        return rgb, "OPAQUE"
    a8 = (np.clip(np.nan_to_num(a), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.dstack([rgb, a8]), "MASK"


# --------------------------------------------------------------------------- public API
def depth_mesh(
    img: RGB,
    inv_z: FloatMap,
    K: Intrinsics,
    *,
    alpha: FloatMap | None = None,
    target_faces: int = 300_000,
    cut_px: float = 2.0,
    ref_baseline_m: float = 0.1,
    name: str = "layer",
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> Mesh:
    """Build a textured mesh from an image and its inverse depth (algorithm H).

    The result is in the glTF frame with the source camera at the origin looking down
    -z (``P_gltf = cv_to_gltf(unproject(K, 1/inv_z))``).  Triangles that would tear by
    more than ``cut_px`` pixels for a ``ref_baseline_m`` head movement are cut, as are
    triangles touching pixels with ``alpha < 0.5`` or invalid depth; tiny components are
    removed.  The texture is ``img`` (RGBA with ``alpha_mode="MASK"`` when ``alpha`` has
    transparent pixels).  Raises :class:`MeshError` when nothing is left to mesh.
    """
    rep = _as_reporter(progress)
    _validate_params(target_faces, cut_px, ref_baseline_m)
    rgb, z, a = _validate_layer(img, inv_z, K, alpha)
    _check(cancel)
    sm = _front_screen_mesh(
        z, a, K, int(target_faces), float(cut_px), float(ref_baseline_m), rep.sub(0, 0.95), cancel
    )
    if sm is None:
        raise MeshError(
            "Nothing is left to mesh: the subject mask is empty or the depth map is invalid.",
            hint="Check the depth map and the subject mask.",
        )
    pts, uv = _finalize(sm, K)
    tex, mode = _layer_texture(rgb, a)
    rep(1.0, "Mesh ready")
    return Mesh(
        vertices=cv_to_gltf(pts).astype(np.float32),
        faces=sm.faces.astype(np.uint32),
        uvs=uv,
        texture=tex,
        name=str(name),
        alpha_mode=mode,  # type: ignore[arg-type]
    )


def _pool_any(mask: np.ndarray, grid: _Grid) -> np.ndarray:
    """Per grid cell: does any pixel of the cell have ``mask`` set?  (ny-1, nx-1) bool."""
    m = mask.view(np.uint8) if mask.dtype == bool else (mask != 0).view(np.uint8)
    rows = np.maximum.reduceat(m, grid.yi[:-1], axis=0)
    return np.maximum.reduceat(rows, grid.xi[:-1], axis=1) > 0


def _differs(layer: Layer, front: Layer, strip: int = 1024) -> np.ndarray:
    """Pixels where ``layer`` differs from ``front`` (colour or depth) or the front is transparent."""
    h, w = front.inv_z.shape
    out = np.zeros((h, w), bool)
    for y0 in range(0, h, strip):
        y1 = min(h, y0 + strip)
        diff = cv2.absdiff(
            np.ascontiguousarray(layer.rgb[y0:y1, :, :3]), np.ascontiguousarray(front.rgb[y0:y1, :, :3])
        )
        m = diff.max(axis=2) > _RGB_DIFF
        zb = np.asarray(layer.inv_z[y0:y1], dtype=np.float32)
        zf = np.asarray(front.inv_z[y0:y1], dtype=np.float32)
        with np.errstate(invalid="ignore"):
            m |= np.abs(zb - zf) > 1e-4 * np.maximum(np.abs(zf), 1e-6)
            m |= ~(np.asarray(front.alpha[y0:y1]) >= 0.5)
        out[y0:y1] = m
    return out


def _backdrop_screen_mesh(
    inv_z: np.ndarray, alpha: np.ndarray | None, K: Intrinsics, target: int
) -> _ScreenMesh | None:
    """Coarse, uncut backdrop whose vertices take the farthest depth of their adjacent cells.

    Because inverse depth is affine in screen space over a planar triangle and every
    vertex value is <= the minimum inverse depth of each cell it touches, the backdrop
    lies on or behind the layer surface everywhere (before the 1 % push).
    """
    grid = _make_grid(K.width, K.height, max(2, target), 1.0)
    z = np.asarray(inv_z)
    starts = grid.yi[:-1]
    bounds = np.append(starts, K.height)
    cells = np.empty((grid.ny - 1, grid.nx - 1), np.float32)
    k0 = 0
    while k0 < len(starts):  # chunks of cell rows keep the temporaries small on 24 MP inputs
        k1 = int(np.searchsorted(bounds, bounds[k0] + 512, side="right"))
        k1 = min(len(starts), max(k0 + 1, k1 - 1))
        sub = np.asarray(z[bounds[k0] : bounds[k1]], dtype=np.float32)
        with np.errstate(invalid="ignore"):
            sub = np.where(np.isfinite(sub) & (sub > _MIN_INV_Z), sub, np.float32(np.inf))
        rows = np.minimum.reduceat(sub, starts[k0:k1] - bounds[k0], axis=0)
        cells[k0:k1] = np.minimum.reduceat(rows, grid.xi[:-1], axis=1)
        k0 = k1
    p = np.pad(cells, 1, mode="constant", constant_values=np.inf)
    vm = np.minimum(np.minimum(p[:-1, :-1], p[:-1, 1:]), np.minimum(p[1:, :-1], p[1:, 1:])).astype(np.float64)
    valid = np.isfinite(vm)
    if alpha is not None:
        valid &= _sample(alpha, grid) >= 0.5
    vm[~valid] = 0.0
    return _grid_mesh(vm, valid, grid, None)


def _anchor_depth(front: Layer, K: Intrinsics) -> float:
    """Median depth (z) of the central window of the front layer (opaque pixels)."""
    return _window_depth(front.inv_z, front.alpha)


def _window_depth(inv_z: np.ndarray, alpha: np.ndarray | None) -> float:
    h, w = inv_z.shape
    r = max(1, int(round(_ANCHOR_WINDOW * min(w, h))))
    cy, cx = (h - 1) // 2, (w - 1) // 2
    sl = (slice(max(0, cy - r), min(h, cy + r + 1)), slice(max(0, cx - r), min(w, cx + r + 1)))
    for region in (sl, (slice(None), slice(None))):
        z = np.asarray(inv_z[region], dtype=np.float64)
        ok = np.isfinite(z) & (z > _MIN_INV_Z)
        if alpha is not None:
            ok &= np.asarray(alpha[region]) >= 0.5
        if ok.any():
            return float(np.median(1.0 / z[ok]))
    raise MeshError("The front layer has no valid depth to anchor the scene.")


def anchor_scale(
    inv_z: FloatMap, K: Intrinsics, anchor_distance_m: float, *, alpha: FloatMap | None = None
) -> float:
    """The scale ``s`` that :func:`layered_meshes` applies for ``anchor_distance_m``.

    ``inv_z`` / ``alpha`` are those of the FRONT layer (for :func:`geometry.layers.build_layers`
    output: the photo's inverse depth, and the subject matte in clean-plate mode).  The
    scene is scaled by ``s`` about the eye, so a real head movement ``d`` produces the
    parallax of ``d / s`` in the depth units of ``inv_z``: look-behind content prepared
    for a movement range ``m`` must be built for ``build_layers(max_baseline_m=m / s)``.
    """
    ok = isinstance(anchor_distance_m, (int, float, np.number)) and not isinstance(anchor_distance_m, bool)
    if not (ok and math.isfinite(anchor_distance_m) and anchor_distance_m > 0):
        raise InputError(f"anchor_distance_m must be a positive distance (got {anchor_distance_m!r}).")
    z = np.asarray(inv_z)
    if z.ndim != 2 or z.shape != (K.height, K.width):
        raise InputError(f"inv_z must be {K.height}x{K.width} (got shape {z.shape}).")
    a = None if alpha is None else np.asarray(alpha)
    if a is not None and a.shape != z.shape:
        raise InputError(f"alpha must have the shape of inv_z {z.shape} (got {a.shape}).")
    return float(anchor_distance_m) / _window_depth(z, a)


def _cell_region(diff: np.ndarray, grid: _Grid) -> np.ndarray:
    """Grid cells holding any ``diff`` pixel, dilated by one cell."""
    region = _pool_any(diff, grid)
    if region.any():
        region = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return region


def _opaque_fraction(layer: Layer) -> float:
    """Share of pixels a layer can mesh (opaque, valid depth); subsampled for speed."""
    h, w = layer.inv_z.shape
    st = max(1, int(math.sqrt(h * w / 1_000_000)))
    z = np.asarray(layer.inv_z[::st, ::st], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(z) & (z > _MIN_INV_Z) & (np.asarray(layer.alpha[::st, ::st]) >= 0.5)
    return float(ok.mean()) if ok.size else 0.0


def _split_budget(target: int, front_cover: float, back_cover: list[float]) -> tuple[int, int, int, int]:
    """Faces for ``(front, front grid density, background grid density, backdrop)``.

    The backdrop takes ``target / 16``; the rest is shared so the front and the
    background detail meshes (at :data:`BACKGROUND_DENSITY` of the front's density)
    together use about the remaining budget for the areas they cover.  Densities are
    full-frame face counts and never exceed ``target`` (the old per-layer budget).
    """
    backdrop = max(2, target // _BACKDROP_SHARE)
    rest = max(2, target - backdrop)
    weight = max(front_cover, 0.0) + BACKGROUND_DENSITY * sum(max(c, 0.0) for c in back_cover)
    d_front = rest / max(weight, 1e-6)
    front_faces = int(max(2, min(rest, round(d_front * max(front_cover, 0.0)))))
    front_density = int(max(2, min(target, round(d_front))))
    back_density = int(max(2, min(target, round(BACKGROUND_DENSITY * d_front))))
    return front_faces, front_density, back_density, backdrop


def _merge_parts(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts, uvs, faces = [], [], []
    off = 0
    for v, uv, f in parts:
        verts.append(v)
        uvs.append(uv)
        faces.append(f + off)
        off += len(v)
    return np.concatenate(verts), np.concatenate(uvs), np.concatenate(faces)


def layered_meshes(
    li: LayeredImage,
    *,
    target_faces: int = 300_000,
    cut_px: float = 2.0,
    ref_baseline_m: float = 0.1,
    anchor_distance_m: float | None = None,
    eye_height_m: float = 1.5,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> list[Mesh]:
    """Meshes for a layered depth image, ordered back -> front, in the glTF frame.

    ``target_faces`` is the budget of the whole result (all meshes together, about
    ``target_faces`` triangles), shared as follows:

    * **Backdrop**: ``target_faces/16`` (4x the stride of a full-frame grid).
    * **Front layer**: :func:`depth_mesh`-style cut mesh (RGBA + MASK texture when the
      layer has a matte).
    * **Background layer** (``"background"``): a detail mesh restricted to the cells
      where the background differs from the front (inpainted colour or pushed-back
      depth, i.e. the hallucinated region) or the front is transparent, dilated by one
      cell and pushed back 0.2 % (avoids z-fighting where it coincides with the front);
      plus the uncut backdrop whose vertex depths are the farthest depth of the adjacent
      cells, pushed back 1 %, so revealed areas show stretched background instead of
      holes.  Further middle layers get detail meshes only.
    * The front and the detail meshes share the rest by the area they cover, the detail
      meshes at :data:`BACKGROUND_DENSITY` times the front's face density.
    * A single-layer image yields ``[backdrop, foreground]`` (the backdrop built from
      the only layer).

    **Placement.**  All meshes share one similarity transform that keeps them an exact
    perspective reconstruction seen from the eye: ``P = s * cv_to_gltf(P_cam) +
    (0, eye_height_m, 0)``.  The source camera (eye) is therefore at
    ``(0, eye_height_m, 0)`` looking down -z, and scaling about the eye moves every point
    only along its own viewing ray (the photo looks identical from the eye).  With
    ``anchor_distance_m`` the scale is ``s = anchor_distance_m / z_c`` where ``z_c`` is
    the median depth of the opaque front pixels in a central window (10 % of the short
    side), so the image centre lands at ``(0, eye_height_m, -anchor_distance_m)`` (exact
    when the principal point is the image centre; the optical axis maps to -z, the
    horizon is not re-levelled).  Without an anchor ``s = 1`` (depth units as given).
    :func:`anchor_scale` computes ``s`` up front (to size the look-behind layers for a
    real head-movement range before they are built).
    """
    rep = _as_reporter(progress)
    _validate_params(target_faces, cut_px, ref_baseline_m)
    if not li.layers:
        raise InputError("The layered image has no layers.")
    if not math.isfinite(eye_height_m):
        raise InputError(f"eye_height_m must be a finite height in metres (got {eye_height_m!r}).")
    K = li.K
    layers: list[Layer] = []
    for i, layer in enumerate(li.layers):
        rgb, z, a = _validate_layer(layer.rgb, layer.inv_z, K, layer.alpha, what=f"layer {i}")
        layers.append(Layer(rgb, z, a if a is not None else np.ones(z.shape, np.float32)))
    front = layers[-1]
    target = int(target_faces)
    dscale = K.fx * float(ref_baseline_m)
    thr = float(cut_px) / dscale

    scale = 1.0
    if anchor_distance_m is not None:
        scale = anchor_scale(front.inv_z, K, anchor_distance_m, alpha=front.alpha)
    offset = np.array([0.0, float(eye_height_m), 0.0])

    # Face budget: total ~ target over all meshes (E2E-6: was ~2x with a full look-behind band).
    diffs = [_differs(layer, front) for layer in layers[:-1]]
    _check(cancel)
    probe = _make_grid(K.width, K.height, max(2, int(target * BACKGROUND_DENSITY)), 1.0)
    back_cover = [float(_cell_region(d, probe).mean()) for d in diffs]
    front_faces, front_density, back_density, backdrop_faces = _split_budget(
        target, _opaque_fraction(front), back_cover
    )
    log.debug(
        "Mesh budget %d: front %d faces (grid %d), background grid %d (covers %s), backdrop %d",
        target, front_faces, front_density, back_density, [round(c, 3) for c in back_cover], backdrop_faces,
    )  # fmt: skip

    def to_scene(p: np.ndarray) -> np.ndarray:
        return (cv_to_gltf(p) * scale + offset).astype(np.float32)

    meshes: list[Mesh] = []
    n_back = max(1, len(layers) - 1)
    for i, layer in enumerate(layers[:-1]):
        _check(cancel)
        rep(0.5 * i / n_back, "Meshing background")
        parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        grid = _make_grid(K.width, K.height, back_density, 1.0)
        region = _cell_region(diffs[i], grid)
        if region.any():
            w_v, valid = _vertex_samples(layer.inv_z, layer.alpha, grid)
            det = _grid_mesh(w_v, valid, grid, thr, region=region, min_faces=_MIN_COMPONENT_FLOOR)
            if det is not None:
                p, uv = _finalize(det, K, DETAIL_PUSH)
                parts.append((p, uv, det.faces))
        _check(cancel)
        if i == 0:
            bd = _backdrop_screen_mesh(layer.inv_z, layer.alpha, K, backdrop_faces)
            if bd is not None:
                p, uv = _finalize(bd, K, BACKDROP_PUSH)
                parts.append((p, uv, bd.faces))
        if not parts:
            log.warning("Background layer %d produced no geometry", i)
            continue
        v, uv, f = _merge_parts(parts)
        tex, mode = _layer_texture(layer.rgb, layer.alpha)
        meshes.append(
            Mesh(to_scene(v), f.astype(np.uint32), uv, tex, "background" if i == 0 else f"layer{i}", mode)  # type: ignore[arg-type]
        )

    if len(layers) == 1:
        bd = _backdrop_screen_mesh(front.inv_z, front.alpha, K, backdrop_faces)
        if bd is not None:
            p, uv = _finalize(bd, K, BACKDROP_PUSH)
            tex, mode = _layer_texture(front.rgb, front.alpha)
            meshes.append(Mesh(to_scene(p), bd.faces.astype(np.uint32), uv, tex, "backdrop", mode))  # type: ignore[arg-type]

    _check(cancel)
    rep(0.5, "Meshing foreground")
    sm = _front_screen_mesh(
        front.inv_z,
        front.alpha,
        K,
        front_faces,
        float(cut_px),
        float(ref_baseline_m),
        rep.sub(0.5, 0.98),
        cancel,
        density=front_density,
    )
    if sm is None:
        raise MeshError("Nothing is left to mesh in the front layer (empty matte or invalid depth).")
    p, uv = _finalize(sm, K)
    tex, mode = _layer_texture(front.rgb, front.alpha)
    meshes.append(Mesh(to_scene(p), sm.faces.astype(np.uint32), uv, tex, "foreground", mode))  # type: ignore[arg-type]
    rep(1.0, "Meshes ready")
    return meshes


# --------------------------------------------------------------------------- decimation
def _fit_projection(vertices: np.ndarray, uvs: np.ndarray, tex_shape: tuple[int, ...]) -> np.ndarray | None:
    """Fit a 3x4 camera ``uv ~ P @ [X, 1]`` (DLT).  Returns P when every vertex reprojects
    within half a texel (depth meshes, billboards), else None."""
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(uvs, dtype=np.float64)
    n = len(v)
    if n < 8 or not (np.isfinite(v).all() and np.isfinite(t).all()):
        return None
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, size=min(n, 4096), replace=False))
    X, x = v[idx], t[idx]
    mX, mx = X.mean(0), x.mean(0)
    sX = math.sqrt(3.0) / max(float(np.sqrt(((X - mX) ** 2).sum(1).mean())), 1e-12)
    sx = math.sqrt(2.0) / max(float(np.sqrt(((x - mx) ** 2).sum(1).mean())), 1e-12)
    Xh = np.hstack([(X - mX) * sX, np.ones((len(X), 1))])
    xn = (x - mx) * sx
    A = np.zeros((2 * len(X), 12))
    A[0::2, 0:4] = Xh
    A[0::2, 8:12] = -xn[:, :1] * Xh
    A[1::2, 4:8] = Xh
    A[1::2, 8:12] = -xn[:, 1:2] * Xh
    try:
        pn = np.linalg.svd(A, full_matrices=False)[2][-1].reshape(3, 4)
    except np.linalg.LinAlgError:
        return None
    TX = np.diag([sX, sX, sX, 1.0])
    TX[:3, 3] = -sX * mX
    Tx_inv = np.array([[1 / sx, 0, mx[0]], [0, 1 / sx, mx[1]], [0, 0, 1.0]])
    P = Tx_inv @ pn @ TX
    h = v @ P[:, :3].T + P[:, 3]
    den = h[:, 2]
    if not (np.all(den > 1e-12) or np.all(den < -1e-12)):
        return None
    err = np.abs(h[:, :2] / den[:, None] - t) * np.array([tex_shape[1], tex_shape[0]], dtype=np.float64)
    if float(err.max()) > 0.5:
        return None
    return P


def _project_uv(P: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    h = np.asarray(vertices, dtype=np.float64) @ P[:, :3].T + P[:, 3]
    return (h[:, :2] / h[:, 2:3]).astype(np.float32)


def decimate(
    mesh: Mesh,
    target_faces: int,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> Mesh:
    """Reduce ``mesh`` to about ``target_faces`` triangles (quadric edge collapse).

    Uses ``fast_simplification`` with open borders preserved; when it is not installed
    the mesh is returned unchanged with a warning (also when it already has few enough
    faces).  UVs of the simplified vertices are recomputed by projecting them with a
    projective camera fitted to the input vertices/UVs (exact for depth meshes, whose UVs
    are a perspective projection, and billboards, an orthographic one); meshes with any
    other parametrisation take the UV of the nearest source vertex collapsed into each
    output vertex.  Normals are recomputed when the input had normals.
    """
    rep = _as_reporter(progress)
    target_faces = _face_budget(target_faces, minimum=1)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) <= target_faces:
        return mesh
    fs = _fast_simplification()
    if fs is None:
        log.warning("fast-simplification is not installed; mesh '%s' was not decimated", mesh.name)
        return mesh
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    uvs = np.asarray(mesh.uvs, dtype=np.float64)
    if faces.min() < 0 or faces.max() >= len(verts):
        raise InputError(f"Mesh '{mesh.name}' has face indices outside its vertex array.")
    _check(cancel)
    rep(0.05, "Simplifying mesh")
    P = _fit_projection(verts, uvs, np.asarray(mesh.texture).shape)
    if P is not None:
        out_v, out_f = _qem(fs, verts, faces, int(target_faces))[:2]
        out_v = np.asarray(out_v, dtype=np.float64)
        out_f = _clean_faces(np.asarray(out_f, dtype=np.int64))
        used, out_f = _compact(len(out_v), out_f)
        out_v = out_v[used]
        out_uv = _project_uv(P, out_v)
    else:
        _, _, collapses = _qem(fs, verts, faces, int(target_faces), collapses=True)
        _check(cancel)
        out_v, out_f, mapping = fs.replay_simplification(
            np.ascontiguousarray(verts), np.ascontiguousarray(faces, dtype=np.int32), collapses
        )
        out_v = np.asarray(out_v, dtype=np.float64)
        mapping = np.asarray(mapping, dtype=np.int64)
        out_uv = np.zeros((len(out_v), 2), np.float32)
        ok = (mapping >= 0) & (mapping < len(out_v))
        src = np.nonzero(ok)[0]
        dst = mapping[src]
        dist = np.linalg.norm(verts[src] - out_v[dst], axis=1)
        order = np.lexsort((dist, dst))
        first = np.unique(dst[order], return_index=True)[1]
        out_uv[dst[order][first]] = uvs[src[order][first]]
        out_f = _clean_faces(np.asarray(out_f, dtype=np.int64))
        used, out_f = _compact(len(out_v), out_f)
        out_v, out_uv = out_v[used], out_uv[used]
    _check(cancel)
    if len(out_f) == 0:
        log.warning("Simplification of '%s' produced no faces; returning the input", mesh.name)
        return mesh
    normals = vertex_normals(out_v, out_f) if mesh.normals is not None else None
    rep(1.0, "Mesh simplified")
    return replace(
        mesh,
        vertices=out_v.astype(np.float32),
        faces=out_f.astype(np.uint32),
        uvs=out_uv.astype(np.float32),
        normals=normals,
    )


__all__ = [
    "depth_mesh",
    "layered_meshes",
    "anchor_scale",
    "BACKGROUND_DENSITY",
    "decimate",
    "vertex_normals",
    "MeshError",
    "MIN_COMPONENT_FACES",
    "BACKDROP_PUSH",
    "DETAIL_PUSH",
]
