"""SHARP 360 back mould (geometry/back_mould.py): shell prior, Poisson height field, back
Gaussians and the rim wall, on the analytic synthetic person (tests/geometry/synth_person.py)."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest
import synth_person as sp

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, SplatCloud
from vrphoto.geometry import back_mould as bm
from vrphoto.geometry.splat import concat_splats, quat_to_rotmat, subset_splats

MAGENTA = np.array([255, 0, 255], np.uint8)


@pytest.fixture(scope="module")
def body() -> sp.Body:
    return sp.make_body()


@pytest.fixture(scope="module")
def cam():
    return sp.make_cam()


@pytest.fixture(scope="module")
def front(body, cam):
    return sp.synth_front_model(body, cam)


@pytest.fixture(scope="module")
def canon(body, cam):
    return sp.synth_canon(body, cam, thickness=0.11)


@pytest.fixture(scope="module")
def shell(front, canon) -> bm.BackPart:
    return bm.shell_back(front, canon)


# --------------------------------------------------------------------------- helpers
def _cell_rays(front) -> np.ndarray:
    """``(Hm, Wm, 3)`` z = 1 rays through the mould-cell centres."""
    u, v = sp.grid_centers(front.grid)
    return sp.pixel_rays(front.cam, u, v)


def _cells_of(front, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, v, _ = sp.project_points(front.cam, P)
    return np.floor(v / front.grid.cell_px).astype(int), np.floor(u / front.grid.cell_px).astype(int)


def _back_rows(part: bm.BackPart) -> SplatCloud:
    return subset_splats(part.sc, np.arange(part.diag["n_back"]))


def _wall_rows(part: bm.BackPart) -> SplatCloud:
    return subset_splats(part.sc, np.arange(part.diag["n_back"], len(part.sc)))


def ortho_opacity(sc: SplatCloud, canon, phi: float, m: float, N: int, *, k: float = 3.0) -> np.ndarray:
    """Accumulated opacity ``1 - prod(1 - a_i g_i)`` of the Gaussians in :func:`sp.ortho_view`'s
    pixel grid (order-independent, so no sorting): each Gaussian's 3D covariance projected
    orthographically, plus the usual 0.3 px^2 anti-aliasing blur, evaluated within ``k``
    sigma."""
    right, down, _ = sp.ortho_axes(canon, phi)
    u, v, _ = sp.ortho_project(canon, phi, m, N, sc.means)
    J = np.stack([right, down]) / m
    M = np.einsum("ij,njk->nik", J, quat_to_rotmat(sc.quats)) * sc.scales[:, None, :].astype(np.float64)
    cov = M @ np.swapaxes(M, 1, 2) + 0.3 * np.eye(2)
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
    ia, ib, ic = cov[:, 1, 1] / det, -cov[:, 0, 1] / det, cov[:, 0, 0] / det
    rad = np.minimum(np.ceil(k * np.sqrt(np.maximum(cov[:, 0, 0], cov[:, 1, 1]))), 16).astype(int)
    log_t = np.zeros(N * N)
    op = sc.opacity.astype(np.float64)
    for r in np.unique(rad):  # one vectorised pass per footprint size
        sel = np.flatnonzero(rad == r)
        off = np.arange(-r, r + 1)
        dy, dx = (g.ravel()[None] for g in np.meshgrid(off, off, indexing="ij"))
        px = np.floor(u[sel])[:, None] + dx
        py = np.floor(v[sel])[:, None] + dy
        ex, ey = px + 0.5 - u[sel][:, None], py + 0.5 - v[sel][:, None]
        q = ia[sel, None] * ex**2 + 2 * ib[sel, None] * ex * ey + ic[sel, None] * ey**2
        a = np.minimum(op[sel, None] * np.exp(-0.5 * q), 0.99)
        ok = (px >= 0) & (px < N) & (py >= 0) & (py < N) & (q <= k * k)
        np.add.at(log_t, (py[ok] * N + px[ok]).astype(np.int64), np.log1p(-a[ok]))
    return 1.0 - np.exp(log_t.reshape(N, N))


def _coverage(sc: SplatCloud, body, canon, phi: float, m: float = 0.0125, N: int = 160) -> float:
    _, alpha, _ = sp.ortho_view(body, canon, phi, m, N)
    sil = alpha >= 0.5
    return float((ortho_opacity(sc, canon, phi, m, N)[sil] >= 0.5).mean())


def _ellipsoid_back(Hm: int = 96, Wm: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """A tilted ellipsoid seen by a pinhole camera on an ``Hm x Wm`` grid: ``(z where each
    cell-centre ray leaves it, mask of the rays that hit it)``."""
    cam = sp.make_cam(Wm, Hm, 50.0)
    grid = sp.mould_grid(Wm, Hm, max(Wm, Hm))
    c, s = np.cos(0.4), np.sin(0.4)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    ball = sp.Body(
        names=("ball",),
        centers=np.array([[0.1, -0.05, 3.0]]),
        rot=np.array([rot]),
        radii=np.array([[0.6, 0.45, 0.2]]),
        R_cw=np.eye(3),
        R_cb=np.eye(3),
        origin=np.zeros(3),
        distance=3.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        cam_height=1.0,
        wall_z=9.0,
    )
    u, v = sp.grid_centers(grid)
    _, t_out = ball.intervals(np.zeros(3), sp.pixel_rays(cam, u, v).reshape(-1, 3))
    back = t_out[:, 0].reshape(grid.Hm, grid.Wm)
    mask = np.isfinite(back)
    return back, mask


# --------------------------------------------------------------------------- shell prior
def test_shell_median_thickness_and_shape(body, cam, front, canon, shell):
    m = front.matte
    target = 0.11 * canon.height_s
    th = (shell.back_z - front.D_f)[m]
    assert shell.kind == "mould" and shell.source == "shell" and shell.ok
    assert abs(np.median(th) - target) <= 0.1 * target
    assert th.min() >= bm.MIN_GAP_M - 1e-6
    assert np.isnan(shell.back_z[~m]).all() and np.isfinite(shell.back_z[m]).all()
    assert 0.0 < shell.diag["kappa"] <= 4.0
    # wide parts get an elliptical cross-section: the thickness follows the body's
    true = (sp.true_back_depth(body, cam) - front.D_f)[m]
    assert np.corrcoef(th, true)[0, 1] > 0.8
    assert th.max() < 3.0 * target  # no balloon torso

    Z, T = bm.inflation_prior(front, canon, thickness=0.11)
    assert Z.shape == T.shape == m.shape and Z.dtype == np.float32
    assert np.isnan(Z[~m]).all() and np.isnan(T[~m]).all()
    assert (T[m] >= 0).all()
    np.testing.assert_allclose(np.maximum(Z, front.D_f + bm.MIN_GAP_M)[m], shell.back_z[m], atol=1e-5)

    thin = bm.shell_back(front, canon, thickness=0.05)
    assert np.median((thin.back_z - front.D_f)[m]) == pytest.approx(0.05 * canon.height_s, rel=0.1)
    flat = bm.shell_back(front, canon, thickness=0.0)  # only the smoothed front remains
    assert flat.diag["kappa"] == 0.0
    assert np.median((flat.back_z - front.D_f)[m]) < 0.01


def test_shell_is_closed_from_eight_azimuths(body, front, canon, shell):
    both = concat_splats([front.sc, shell.sc])
    cover = {phi: _coverage(both, body, canon, phi) for phi in range(0, 360, 45)}
    assert min(cover.values()) >= 0.97, cover
    # the check can fail: the photographed front alone leaves the sides open
    assert _coverage(front.sc, body, canon, 90) < 0.9


@pytest.mark.parametrize("kw", [{"pitch_deg": 15.0}, {"yaw_deg": 30.0}])
def test_shell_is_closed_for_a_tilted_camera_and_a_turned_body(cam, kw):
    body = sp.make_body(**kw)
    front = sp.synth_front_model(body, cam)
    canon = sp.synth_canon(body, cam)
    part = bm.shell_back(front, canon)
    both = concat_splats([front.sc, part.sc])
    for phi in (90, 180, 270):
        assert _coverage(both, body, canon, phi) >= 0.97


def test_back_frames_are_rotations_with_normals_away_from_the_camera(front, shell):
    R = quat_to_rotmat(shell.sc.quats)
    np.testing.assert_allclose(R.transpose(0, 2, 1) @ R, np.broadcast_to(np.eye(3), R.shape), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)
    assert shell.normals.shape == (len(shell.sc), 3)
    np.testing.assert_allclose(shell.normals, R[:, :, 2], atol=1e-5)

    nb = shell.diag["n_back"]
    ray = shell.sc.means[:nb] / np.linalg.norm(shell.sc.means[:nb], axis=1, keepdims=True)
    assert ((shell.normals[:nb] * ray).sum(1) > 0).all()
    s = shell.sc.scales
    assert (s[:, :2] >= 0.001 - 1e-7).all() and (s[:, :2] <= 0.05 + 1e-7).all()
    assert (s[:, 2] >= 0.0005 - 1e-7).all() and (s[:, 2] <= s[:, :2].min(1) + 1e-7).all()
    # every back Gaussian sits on the ray through its own mould cell, at the back depth
    rows, cols = _cells_of(front, shell.sc.means[:nb])
    assert front.matte[rows, cols].all()
    np.testing.assert_allclose(shell.sc.means[:nb, 2], shell.back_z[rows, cols], rtol=1e-5)
    assert np.allclose(shell.sc.opacity, bm.BACK_OPACITY)


def test_gaussians_from_heightfield_on_planes(front):
    m = front.matte
    rays = _cell_rays(front)
    flat, n_flat = bm.gaussians_from_heightfield(np.full(m.shape, 3.0), m, front, front.rgb)
    assert len(flat) == m.sum()
    np.testing.assert_allclose(n_flat, np.broadcast_to([0.0, 0.0, 1.0], n_flat.shape), atol=1e-6)
    cell_m = 3.0 * front.grid.cell_px / front.cam.f  # metric cell size at 3 m
    np.testing.assert_allclose(flat.scales[:, :2], 0.6 * cell_m, rtol=1e-4)
    np.testing.assert_allclose(flat.colors, front.rgb[m].astype(np.float32) / 255.0, atol=1e-6)

    # a tilted plane n . P = d: its normal comes back (oriented away from the camera)
    n = np.array([0.3, -0.2, 1.0])
    n /= np.linalg.norm(n)
    z = 2.5 / (rays @ n)
    sc, nrm = bm.gaussians_from_heightfield(z, m, front, np.zeros(m.shape + (3,)), stride=2)
    np.testing.assert_allclose(sc.means @ n, 2.5, atol=1e-5)
    rr, cc = np.nonzero(m)
    assert len(sc) == int(((rr % 2 == 0) & (cc % 2 == 0)).sum())
    # exact wherever the cell has a neighbour along both grid axes; one-cell-thin spots fall
    # back to a step at constant depth along the missing axis, still facing away
    rows, cols = _cells_of(front, sc.means)
    pad = np.pad(m, 1)
    both = (pad[rows + 1, cols] | pad[rows + 1, cols + 2]) & (pad[rows, cols + 1] | pad[rows + 2, cols + 1])
    assert both.mean() > 0.95
    np.testing.assert_allclose(nrm[both], np.broadcast_to(n, nrm[both].shape), atol=1e-4)
    assert ((nrm * sc.means).sum(1) > 0).all()


def test_big_grids_are_processed_in_chunks_with_the_same_result(monkeypatch, front, canon, shell):
    one, n_one = bm.gaussians_from_heightfield(shell.back_z, front.matte, front, front.rgb)
    monkeypatch.setattr(bm, "_CHUNK", 97)  # many chunks
    many, n_many = bm.gaussians_from_heightfield(shell.back_z, front.matte, front, front.rgb)
    for f in ("means", "scales", "quats", "colors", "opacity"):
        np.testing.assert_array_equal(getattr(one, f), getattr(many, f))
    np.testing.assert_array_equal(n_one, n_many)
    monkeypatch.setattr(bm, "_KAPPA_SAMPLES", 400)  # kappa from a regular subsample
    part = bm.shell_back(front, canon)
    assert part.diag["kappa"] == pytest.approx(shell.diag["kappa"], rel=0.1)
    assert part.diag["median_thickness_m"] == pytest.approx(0.11 * canon.height_s, rel=0.1)


# --------------------------------------------------------------------------- colours
def test_diffuse_back_has_no_front_face_colours_and_mirror_does(front, canon):
    # paint everything deeper than the outline band magenta (a "face" in the middle of the
    # front); the outline band keeps the photo's own colours
    band = max(2.0, 8 * max(front.matte.shape) / 1024)
    rgb = front.rgb.copy()
    face = front.matte & (front.dt > band + 1.0)
    rgb[face] = MAGENTA
    painted = dataclasses.replace(front, rgb=rgb)
    names = (*sp.SIDES, "magenta")
    sp.PALETTE.setdefault("magenta", MAGENTA)
    try:
        diffuse = bm.shell_back(painted, canon, back="diffuse")
        mirror = bm.shell_back(painted, canon, back="mirror")
        cls_d = sp.classify_rgb(diffuse.sc.colors, names=names)
        cls_m = sp.classify_rgb(_back_rows(mirror).colors, names=names)
    finally:
        sp.PALETTE.pop("magenta", None)
    assert not (cls_d == "magenta").any()
    assert (cls_m == "magenta").sum() >= 0.9 * face.sum()
    # mirror = the photo colours; diffuse = the outline band, 10 % darker
    rows, cols = _cells_of(front, mirror.sc.means[: mirror.diag["n_back"]])
    inner = front.dt[rows, cols] > 1.5
    np.testing.assert_allclose(
        _back_rows(mirror).colors[inner], rgb[rows, cols][inner].astype(np.float32) / 255.0, atol=1e-6
    )
    nb = diffuse.diag["n_back"]
    rows, cols = _cells_of(front, diffuse.sc.means[:nb])
    ring = (front.dt[rows, cols] > 1.5) & (front.dt[rows, cols] <= band + 1.0)
    expect = np.clip(rgb[rows, cols][ring].astype(np.float32) * 0.9 + 0.5, 0, 255).astype(np.uint8)
    np.testing.assert_allclose(diffuse.sc.colors[:nb][ring], expect.astype(np.float32) / 255.0, atol=1e-6)


# --------------------------------------------------------------------------- rim wall
def test_rim_wall_spans_dext_to_zrim(front):
    m = front.matte
    z_rim = np.where(m, front.D_ext + 0.10, np.nan)
    red = np.zeros(m.shape + (3,), np.uint8)
    red[...] = (255, 0, 0)
    blue = np.zeros_like(red)
    blue[...] = (0, 0, 255)
    wall = bm.rim_wall(front, z_rim, colors_front=red, colors_back=blue)
    assert len(wall) > 0
    rows, cols = _cells_of(front, wall.means)
    assert m[rows, cols].all() and (front.dt[rows, cols] < 1.5).all()  # boundary cells only
    # per boundary ray: from D_ext to z_rim, both ends included
    key = rows * m.shape[1] + cols
    order = np.lexsort((wall.means[:, 2], key))
    k, z = key[order], wall.means[order, 2]
    first = np.r_[True, k[1:] != k[:-1]]
    last = np.r_[k[1:] != k[:-1], True]
    d_ext = front.D_ext.ravel()[k]
    np.testing.assert_allclose(z[first], d_ext[first], atol=1e-5)
    np.testing.assert_allclose(z[last], d_ext[last] + 0.10, atol=1e-5)
    assert (z >= d_ext - 1e-5).all() and (z <= d_ext + 0.10 + 1e-5).all()
    # spacing ~ one cell along the ray, so the wall is watertight from the side
    ray_len = np.linalg.norm(_cell_rays(front)[rows, cols], axis=1)
    n_per = np.bincount(np.unique(k, return_inverse=True)[1])
    cell_m = front.grid.cell_px * (front.D_ext.ravel()[np.unique(k)] + 0.05) / front.cam.f
    assert (n_per >= np.ceil(0.10 * ray_len[0] / cell_m)).all()
    # colour lerp red -> blue; axes along the ray / contour; normal outward
    np.testing.assert_allclose(wall.colors[order][first], [[1.0, 0.0, 0.0]] * int(first.sum()), atol=1e-6)
    np.testing.assert_allclose(wall.colors[order][last], [[0.0, 0.0, 1.0]] * int(last.sum()), atol=1e-6)
    R = quat_to_rotmat(wall.quats)
    ray = wall.means / np.linalg.norm(wall.means, axis=1, keepdims=True)
    np.testing.assert_allclose(np.abs((R[:, :, 0] * ray).sum(1)), 1.0, atol=1e-5)
    np.testing.assert_allclose((R[:, :, 2] * ray).sum(1), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)
    step = np.rint(R[:, :2, 2] / np.linalg.norm(R[:, :2, 2], axis=1, keepdims=True) * 2).astype(int)
    rr, cc = np.clip(rows + step[:, 1], 0, m.shape[0] - 1), np.clip(cols + step[:, 0], 0, m.shape[1] - 1)
    assert (~m[rr, cc]).mean() > 0.9  # two cells along the normal is outside the subject

    # no wall where the back edge is within 1 cm of the front edge, or unknown
    assert len(bm.rim_wall(front, np.where(m, front.D_ext + 0.009, np.nan), colors_front=red)) == 0
    assert len(bm.rim_wall(front, np.full(m.shape, np.nan), colors_front=red)) == 0
    sparse = bm.rim_wall(front, z_rim, colors_front=red, spacing_cells=2.0)
    assert 0.15 * len(wall) < len(sparse) < 0.4 * len(wall)


def test_shell_rim_wall_closes_the_outline(front, shell):
    wall = _wall_rows(shell)
    assert shell.diag["n_wall"] == len(wall) > 0
    rows, cols = _cells_of(front, wall.means)
    assert (front.dt[rows, cols] < 1.5).all()
    z = wall.means[:, 2]
    assert (z >= front.D_ext[rows, cols] - 1e-5).all()
    assert (z <= shell.back_z[rows, cols] + 1e-5).all()


# --------------------------------------------------------------------------- Poisson
def test_poisson_recovers_an_ellipsoid_back():
    back, mask = _ellipsoid_back()
    rr, cc = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    target = back + 0.05 + 0.001 * cc - 0.0007 * rr  # right shape, wrong offset and tilt
    z_rim = np.where(mask, back, np.nan)
    z = bm.poisson_heightfield(target, mask, z_rim, lam_rim=1e4, lam_target=0.0)
    assert z.dtype == np.float32 and np.isnan(z[~mask]).all()
    assert np.abs(z - back)[mask].max() < 0.01
    bnd = bm._boundary(mask)
    assert np.abs(z - back)[bnd].max() < 1e-3  # equals z_rim on the boundary
    # with the design's defaults the exact target and rim give the exact back
    z = bm.poisson_heightfield(back, mask, z_rim)
    assert np.abs(z - back)[mask].max() < 1e-4
    # the screening term pulls the interior toward the target, the rim term the boundary
    z = bm.poisson_heightfield(back + 0.05, mask, z_rim, lam_rim=1e4, lam_target=1.0)
    centre = mask & (sp._dist_cells(mask) > 10)
    assert np.abs(z - back - 0.05)[centre].max() < 0.005
    assert np.abs(z - back)[bnd].max() < 0.002


def test_poisson_components_tiny_masks_and_nan():
    m = np.zeros((40, 50), bool)
    m[2:10, 2:12] = True
    m[20, 20] = True  # single cell
    m[30:32, 5:45] = True  # thin strip
    m[15:18, 40] = True  # thin column
    m[0:3, 47:50] = True  # touches the grid border
    T = np.full(m.shape, 2.0)
    T[5, 5] = np.nan  # a hole in the target is filled from its neighbours
    z = bm.poisson_heightfield(T, m, np.full(m.shape, 2.1), lam_target=0.0)
    assert np.isfinite(z[m]).all() and np.isnan(z[~m]).all()
    np.testing.assert_allclose(z[m], 2.1, atol=1e-4)
    z = bm.poisson_heightfield(np.full(m.shape, 2.0), m, np.full(m.shape, 2.1))
    assert (z[m] > 2.0).all() and (z[m] <= 2.1 + 1e-6).all()
    one = np.zeros((5, 5), bool)
    one[2, 2] = True
    z = bm.poisson_heightfield(np.ones((5, 5)), one, np.full((5, 5), 1.5))
    assert z[2, 2] == pytest.approx(1.0 + 0.5 * 10.0 / 10.01, abs=1e-5)
    assert np.isnan(bm.poisson_heightfield(np.ones((4, 4)), np.zeros((4, 4), bool), np.ones((4, 4)))).all()
    # boundary cells without a rim value get no rim term
    z = bm.poisson_heightfield(np.full(m.shape, 2.0), m, np.full(m.shape, np.nan))
    np.testing.assert_allclose(z[m], 2.0, atol=1e-6)


def test_poisson_edge_weights_decouple_regions():
    m = np.zeros((30, 41), bool)
    m[5:25, 5:36] = True
    g = np.where(np.arange(41) < 20, 1.0, 2.0)[None, :].repeat(30, 0)
    T = np.zeros(m.shape)
    z_full = bm.poisson_heightfield(T, m, g, lam_target=0.0, iters=400)
    w = np.ones(m.shape)
    w[:, 20] = 0.0  # no edges into column 20: the halves are independent
    z_cut = bm.poisson_heightfield(T, m, g, weights=w, lam_target=0.0, iters=400)
    np.testing.assert_allclose(z_cut[5:25, 5:20], 1.0, atol=1e-3)
    np.testing.assert_allclose(z_cut[5:25, 21:36], 2.0, atol=1e-3)
    assert np.abs(z_full[15, 18] - 1.0) > 0.1  # without the cut the halves blend


def test_poisson_anchor_and_warm_start():
    back, mask = _ellipsoid_back()
    z_rim = np.where(mask, back, np.nan)
    anchor = np.full(mask.shape, np.nan)
    anchor[40:50, 50:60] = back[40:50, 50:60] + 0.03
    z = bm.poisson_heightfield(back, mask, z_rim, anchor=anchor, lam_anchor=1e4, lam_target=0.0)
    np.testing.assert_allclose(z[40:50, 50:60], anchor[40:50, 50:60], atol=2e-3)
    cold = bm.poisson_heightfield(back + 0.02, mask, z_rim)
    warm = bm.poisson_heightfield(back + 0.02, mask, z_rim, init=cold)
    np.testing.assert_allclose(warm[mask], cold[mask], atol=1e-4)


def test_poisson_rejects_bad_input_and_cancels():
    m = np.ones((8, 8), bool)
    with pytest.raises(InputError):
        bm.poisson_heightfield(np.ones((8, 7)), m, np.ones((8, 8)))
    with pytest.raises(InputError):
        bm.poisson_heightfield(np.ones((8, 8)), m, np.ones((8, 8)), lam_rim=-1.0)
    with pytest.raises(InputError):
        bm.poisson_heightfield(np.full((8, 8), np.nan), m, np.ones((8, 8)))
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        bm.poisson_heightfield(np.ones((8, 8)), m, np.ones((8, 8)), cancel=tok)


# --------------------------------------------------------------------------- mould_back
def test_mould_back_solves_clamps_and_closes(front, canon):
    Z, _ = bm.inflation_prior(front, canon, thickness=0.11)
    m = front.matte
    part = bm.mould_back(front, canon, Z, colors=front.rgb)
    assert part.ok and part.kind == "mould" and part.source == "mould"
    assert not part.diag["fallback"]
    assert np.isnan(part.back_z[~m]).all()
    assert ((part.back_z - front.D_f)[m] >= bm.MOULD_GAP_M - 1e-5).all()
    assert 0.5 * canon.T_prior <= np.median((part.back_z - front.D_f)[m]) <= 2.0 * canon.T_prior
    assert part.diag["n_wall"] == 0  # default rim = D_ext: the back meets the front edge
    assert part.normals.shape == (len(part.sc), 3)

    # a target partly in front of the photo is clamped behind it, then re-solved
    bad = Z.copy()
    bad[m & (front.dt > 3)] = front.D_f[m & (front.dt > 3)] - 0.05
    part = bm.mould_back(front, canon, bad, colors=front.rgb)
    assert part.diag["clamped_cells"] > 0 and part.diag["resolved"]
    assert ((part.back_z - front.D_f)[m] >= bm.MOULD_GAP_M - 1e-5).all()

    # a deeper rim (e.g. from a visual hull) adds a wall from D_ext to it
    z_rim = np.where(m, front.D_ext + 0.08, np.nan)
    part = bm.mould_back(front, canon, Z, colors=front.rgb, z_rim=z_rim, stride=2)
    assert part.diag["n_wall"] > 0 and part.diag["stride"] == 2
    z_med = np.median(part.back_z[m])
    assert part.spacing_m == pytest.approx(2 * front.grid.cell_px * z_med / front.cam.f, rel=1e-6)


def test_mould_back_clip_hull_and_sanity_fallback(front, canon):
    Z, _ = bm.inflation_prior(front, canon, thickness=0.11)
    m = front.matte
    clip = np.where(m, front.D_f + 0.08, np.nan)
    part = bm.mould_back(front, canon, Z, colors=front.rgb, clip=clip)
    assert ((part.back_z - front.D_f)[m] <= 0.08 - bm.MIN_GAP_M + 1e-5).all()
    hull = bm.HullIntervals(
        z_in=front.D_f.copy(), z_out=np.where(m, front.D_f + 0.06, np.nan), valid=m.copy()
    )
    part = bm.mould_back(front, canon, Z, colors=front.rgb, hull=hull)
    assert ((part.back_z - front.D_f)[m] <= 0.06 + 0.02 + 1e-5).all() and part.hull is hull
    assert not part.diag["fallback"]  # no sanity check against the prior with a hull

    far = np.where(m, front.D_f + 1.5, np.nan)  # an absurd 1.5 m thick back
    part = bm.mould_back(front, canon, far, colors=front.rgb, source="test")
    assert part.diag["fallback"] and "built-in" in part.diag["warning"]
    assert np.median((part.back_z - front.D_f)[m]) == pytest.approx(canon.T_prior, rel=0.15)
    assert part.ok and part.source == "test"


# --------------------------------------------------------------------------- options
def test_stride_reduces_the_count(body, front, canon, shell):
    s2 = bm.shell_back(front, canon, stride=2)
    ratio = s2.diag["n_back"] / shell.diag["n_back"]
    assert 0.2 < ratio < 0.32
    # the floater filter sizes its voxels from this: twice the cell size at the back's depth
    assert (
        s2.spacing_m == pytest.approx(2.0 * shell.spacing_m, rel=1e-6)
        and s2.diag["spacing_m"] == s2.spacing_m
    )
    assert len(s2.sc) / len(shell.sc) < 0.4
    assert np.allclose(s2.diag["median_thickness_m"], shell.diag["median_thickness_m"])
    # the bigger Gaussians still close the shell
    both = concat_splats([front.sc, s2.sc])
    assert min(_coverage(both, body, canon, phi) for phi in (90, 180, 270)) >= 0.97


def test_shell_clip_keeps_the_back_in_front_of_another_back(front, canon):
    m = front.matte
    clip = np.where(m, front.D_f + 0.06, np.nan)
    part = bm.shell_back(front, canon, clip=clip)
    th = (part.back_z - front.D_f)[m]
    assert (th <= 0.06 - bm.MIN_GAP_M + 1e-5).all() and (th >= bm.MIN_GAP_M - 1e-6).all()


def test_shell_reports_progress_and_cancels(front, canon):
    seen: list[float] = []
    bm.shell_back(front, canon, progress=lambda f, msg: seen.append(f))
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        bm.shell_back(front, canon, cancel=tok)
    with pytest.raises(Cancelled):
        bm.mould_back(front, canon, front.D_f + 0.1, colors=front.rgb, cancel=tok)

    calls = {"n": 0}

    class Later(CancelToken):  # cancels after the first check: mid-way through the build
        def check(self) -> None:
            calls["n"] += 1
            if calls["n"] > 1:
                raise Cancelled("Operation cancelled")

    with pytest.raises(Cancelled):
        bm.shell_back(front, canon, cancel=Later())


def test_bad_options_are_input_errors(front, canon):
    with pytest.raises(InputError, match="back colour"):
        bm.shell_back(front, canon, back="shiny")
    for stride in (0, 1.5, "2"):
        with pytest.raises(InputError, match="stride"):
            bm.shell_back(front, canon, stride=stride)
    with pytest.raises(InputError, match="thickness"):
        bm.shell_back(front, canon, thickness=-0.1)
    with pytest.raises(InputError, match="empty"):
        bm.shell_back(dataclasses.replace(front, matte=np.zeros_like(front.matte)), canon)
    with pytest.raises(InputError, match="mould grid"):
        bm.shell_back(dataclasses.replace(front, matte=front.matte[:-1]), canon)
    with pytest.raises(InputError, match="height"):
        bm.shell_back(front, dataclasses.replace(canon, height_s=0.0))
    with pytest.raises(InputError):
        bm.gaussians_from_heightfield(front.D_f, front.matte, front, front.rgb[:-1])
    with pytest.raises(InputError):
        bm.rim_wall(front, front.D_ext, colors_front=front.rgb, spacing_cells=0)


@pytest.mark.slow
def test_shell_on_a_768_grid_is_fast():
    body = sp.make_body()
    cam = sp.make_cam(576, 768)
    front = sp.synth_front_model(body, cam, 768, stride=8, ss=1)
    canon = sp.synth_canon(body, cam)
    assert max(front.grid.Hm, front.grid.Wm) == 768
    t0 = time.perf_counter()
    part = bm.shell_back(front, canon)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, elapsed
    assert part.diag["n_back"] == int(front.matte.sum())
    assert abs(part.diag["median_thickness_m"] - 0.11 * canon.height_s) < 0.01
