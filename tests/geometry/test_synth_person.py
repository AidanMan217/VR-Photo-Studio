"""Sanity tests of the analytic SHARP 360 test subject (tests/geometry/synth_person.py)."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest
import synth_person as sp

from vrphoto.geometry.splat import fit_plane_ransac, quat_to_rotmat


@pytest.fixture(scope="module")
def body() -> sp.Body:
    return sp.make_body()


@pytest.fixture(scope="module")
def cam():
    return sp.make_cam()


@pytest.fixture(scope="module")
def scene(body, cam) -> sp.SharpScene:
    return sp.sharp_like_scene(body, cam)


@pytest.fixture(scope="module")
def canon(body, cam):
    return sp.synth_canon(body, cam)


def _frac(names: np.ndarray, *which: str) -> float:
    return float(np.isin(names, which).mean()) if names.size else 0.0


# --------------------------------------------------------------------------- ray casting
def _ball_body(center=(0.0, 0.0, 3.0), radii=(0.5, 0.5, 0.5), rot=None) -> sp.Body:
    eye = np.eye(3)
    return sp.Body(
        names=("ball",),
        centers=np.array([center], np.float64),
        rot=np.array([eye if rot is None else rot]),
        radii=np.array([radii], np.float64),
        R_cw=eye,
        R_cb=eye,
        origin=np.zeros(3),
        distance=3.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        cam_height=1.0,
        wall_z=5.0,
    )


def test_raycast_is_the_closed_form_ellipsoid_hit():
    b = _ball_body()
    dirs = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.1, 0.0, 1.0], [1.0, 0.0, 1.0]])
    t, n, rgb, hit = b.raycast(np.zeros(3), dirs)
    assert hit.tolist() == [True, True, True, False]
    assert t[0] == pytest.approx(2.5) and t[1] == pytest.approx(1.25)  # t is in units of |d|
    np.testing.assert_allclose(n[0], [0.0, 0.0, -1.0], atol=1e-12)
    # sphere: |o + t d - c| = r and the normal is radial
    P = t[2] * dirs[2]
    assert np.linalg.norm(P - [0, 0, 3]) == pytest.approx(0.5)
    np.testing.assert_allclose(n[2], (P - [0, 0, 3]) / 0.5, atol=1e-12)
    assert np.isinf(t[3]) and np.isnan(n[3]).all() and (rgb[3] == 0).all()
    assert (rgb[0] == sp.FRONT_RGB).all()  # faces the camera
    # rotated ellipsoid: hits satisfy its equation, normals are the normalised gradient
    a = math.radians(30)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    e = _ball_body(radii=(0.6, 0.2, 0.3), rot=R)
    rng = np.random.default_rng(0)
    d = np.column_stack([rng.uniform(-0.15, 0.15, 500), rng.uniform(-0.15, 0.15, 500), np.ones(500)])
    o = np.array([0.05, -0.02, 0.1])
    t, n, _, hit = e.raycast(o, d)
    assert hit.sum() > 100
    loc = ((o + t[hit, None] * d[hit]) - [0, 0, 3]) @ R / [0.6, 0.2, 0.3]
    np.testing.assert_allclose((loc**2).sum(1), 1.0, atol=1e-9)
    g = (loc / [0.6, 0.2, 0.3]) @ R.T
    np.testing.assert_allclose(n[hit], g / np.linalg.norm(g, axis=1, keepdims=True), atol=1e-9)
    assert ((n[hit] * d[hit]).sum(1) < 0).all()


def test_first_two_finds_the_torso_behind_the_forearm(body):
    fore = body.part("arm_right_fore")
    P = body.centers[fore]
    ft = body.first_two(np.zeros(3), P[None] / P[2])
    assert ft["part0"][0] == fore
    assert body.names[ft["part1"][0]] in ("torso", "pelvis")
    assert ft["s0"][0] < ft["e0"][0] < ft["s1"][0] < np.inf
    # a torso ray: nothing else of the body behind it
    tor = body.centers[body.part("torso")] + body.R_cb @ [0.05, -0.1, 0.0]
    ft = body.first_two(np.zeros(3), tor[None] / tor[2])
    assert body.names[ft["part0"][0]] == "torso" and ft["part1"][0] == -1 and np.isinf(ft["s1"][0])


def test_body_size_and_asymmetry(body, canon):
    assert canon.height_s == pytest.approx(sp.BODY_HEIGHT, abs=1e-9)
    assert 1.0 < canon.width_s < 1.2
    assert canon.z_axis == pytest.approx(2.5)
    xb = body.R_cb[:, 0]
    tip = body.centers[body.part("arm_left")] @ xb - body.origin @ xb
    bag = body.centers[body.part("bag")] @ xb - body.origin @ xb
    assert tip > 0.4 and bag < -0.2  # raised arm on the body's left, bag on its right
    up = body.up_C
    lo, hi = body.extent(up)
    assert hi - body.origin @ up == pytest.approx(1.70) and lo - body.origin @ up == pytest.approx(0.01)
    # the B frame round-trips and puts the body's own left on +x_B
    P = body.from_B([[0.3, -1.0, 0.1]])
    np.testing.assert_allclose(body.to_B(P), [[0.3, -1.0, 0.1]], atol=1e-12)


# --------------------------------------------------------------------------- photo
def test_photo_matte_and_colours(scene, cam):
    a, rgb = scene.alpha, scene.rgb
    assert a.shape == (cam.height, cam.width) and a.dtype == np.float32
    assert rgb.shape == (cam.height, cam.width, 3) and rgb.dtype == np.uint8
    assert 0.0 <= a.min() and a.max() == 1.0 and ((a > 0) & (a < 1)).any()  # soft edges
    rows, cols = np.nonzero(a >= 0.5)
    assert rows.min() > 5 and rows.max() < cam.height - 5 and cols.min() > 5 and cols.max() < cam.width - 5
    assert 1800 < (a >= 0.5).sum() < 3200
    inside = sp.classify_rgb(rgb[a == 1.0], sp.SIDES)  # blends at red / green edges exist
    assert _frac(inside, "front") > 0.75 and _frac(inside, "front", "left", "right") == 1.0
    for bg in (sp.WALL_RGB, sp.FLOOR_RGB):
        assert not (rgb[a == 1.0] == bg).all(axis=1).any()
    outside = sp.classify_rgb(rgb[a == 0.0])
    assert _frac(outside, "wall", "floor") == 1.0
    assert (sp.classify_rgb(rgb[-3:, :]) == "floor").all() and (sp.classify_rgb(rgb[:3, :]) == "wall").all()
    # the hard pixel-centre matte agrees with the coverage matte
    hard = sp.render_photo(sp.make_body(), cam, ss=1)[1]
    assert set(np.unique(hard)) <= {0.0, 1.0}
    inter = ((hard >= 0.5) & (a >= 0.5)).sum()
    assert inter / ((hard >= 0.5) | (a >= 0.5)).sum() > 0.95


@pytest.mark.parametrize(("yaw", "side"), [(90, "right"), (-90, "left"), (180, "back")])
def test_photo_colour_follows_the_body_side_facing_the_camera(cam, yaw, side):
    _, a, rgb = sp.sharp_like_cloud(sp.make_body(yaw_deg=yaw), cam)
    names = sp.classify_rgb(rgb[a == 1.0], sp.SIDES)
    u, c = np.unique(names, return_counts=True)
    assert u[np.argmax(c)] == side and _frac(names, side) > 0.5  # a thin body shows its slopes too


def test_turning_round_puts_the_raised_arm_on_the_other_image_side(cam):
    def arm_u(yaw):
        b = sp.make_body(yaw_deg=yaw)
        tip = b.centers[b.part("arm_left")]
        return sp.project_points(cam, tip)[0]

    assert arm_u(0) > cam.width / 2 + 15 and arm_u(180) < cam.width / 2 - 15


# --------------------------------------------------------------------------- SHARP-like cloud
def test_layers_kinds_and_counts(scene):
    G = scene.cam.grid
    n = len(scene.sc)
    assert n == 2 * G * G == scene.cam.n_raw and scene.cam.layers == 2
    assert round(math.sqrt(n / 2)) == G  # SharpCam's grid detection rule recovers G
    for arr in (scene.layer, scene.kind, scene.part, scene.cell, scene.surface_z):
        assert arr.shape == (n,)
    for layer, kinds in ((0, {sp.KIND_BODY, sp.KIND_FLOOR, sp.KIND_WALL, sp.KIND_CARD}),
                         (1, {sp.KIND_HIDDEN_BODY, sp.KIND_HIDDEN_BG, sp.KIND_BG_LAYER1})):  # fmt: skip
        m = scene.layer == layer
        assert set(np.unique(scene.kind[m])) == kinds
        np.testing.assert_array_equal(scene.cell[m], np.arange(G * G))  # one per cell, row-major
    assert (scene.layer[: G * G] == 0).all()
    counts = np.bincount(scene.kind, minlength=len(sp.KIND_NAMES))
    assert counts[sp.KIND_BODY] > 800 and counts[sp.KIND_CARD] > 50 and counts[sp.KIND_HIDDEN_BODY] > 20
    assert (
        counts[sp.KIND_HIDDEN_BG] == counts[sp.KIND_BODY] + counts[sp.KIND_CARD] - counts[sp.KIND_HIDDEN_BODY]
    )
    assert ((scene.part >= 0) == np.isin(scene.kind, [sp.KIND_BODY, sp.KIND_CARD, sp.KIND_HIDDEN_BODY])).all()
    for k, op in sp.OPACITY.items():
        np.testing.assert_allclose(scene.sc.opacity[scene.kind == k], op, atol=1e-6)
    # every Gaussian projects into its own SHARP-grid cell
    u, v, z = sp.project_points(scene.cam, scene.sc.means)
    cell = np.floor(v * G / scene.cam.height).astype(int) * G + np.floor(u * G / scene.cam.width).astype(int)
    np.testing.assert_array_equal(cell, scene.cell)
    # layer 1 lies behind layer 0 of the same cell (a card ends where layer 1 starts)
    dz = z[G * G :] - z[: G * G]
    assert (dz > 0).all() and (dz[scene.kind[G * G :] != sp.KIND_HIDDEN_BODY] > 0.02).all()
    names = np.array(sp.make_body().names)
    hb = np.flatnonzero(scene.kind == sp.KIND_HIDDEN_BODY)
    behind_fore = names[scene.part[hb]][names[scene.part[hb - G * G]] == "arm_right_fore"]
    assert len(behind_fore) > 10 and np.isin(behind_fore, ["torso", "pelvis", "bag"]).all()
    assert (behind_fore == "torso").mean() > 0.5


def test_hidden_layer_and_background_colours(scene):
    names = sp.classify_rgb(scene.sc.colors)
    bgk = np.isin(scene.kind, [sp.KIND_FLOOR, sp.KIND_WALL, sp.KIND_HIDDEN_BG, sp.KIND_BG_LAYER1])
    assert np.isin(names[bgk], ["floor", "wall"]).all()
    assert np.isin(names[~bgk], list(sp.SIDES)).all()
    # the hidden wall blobs project inside the subject's matte (that is what makes them hard)
    u, v, _ = sp.project_points(scene.cam, scene.sc.means[scene.kind == sp.KIND_HIDDEN_BG])
    a = scene.alpha[
        np.clip(v.astype(int), 0, scene.cam.height - 1), np.clip(u.astype(int), 0, scene.cam.width - 1)
    ]
    assert (a >= 0.5).mean() > 0.85


def test_matte_matches_the_projection_of_the_cloud(scene):
    cam = scene.cam

    def alpha_at(mask):
        u, v, _ = sp.project_points(cam, scene.sc.means[mask])
        return scene.alpha[v.astype(int), u.astype(int)], u.astype(int), v.astype(int)

    a_body, ub, vb = alpha_at(scene.kind == sp.KIND_BODY)
    assert (a_body >= 0.5).mean() > 0.97
    a_bg, _, _ = alpha_at(np.isin(scene.kind, [sp.KIND_FLOOR, sp.KIND_WALL]))
    assert (a_bg < 0.5).mean() > 0.97
    # body Gaussians sit on the ray-cast surface (pixel-centre depth), up to the jitter
    zb = scene.sc.means[scene.kind == sp.KIND_BODY, 2]
    dz = np.abs(scene.body_depth[vb, ub] - zb)
    assert np.nanmedian(dz) < 0.01
    np.testing.assert_allclose(zb, scene.surface_z[scene.kind == sp.KIND_BODY], atol=1e-5)


def test_cards_are_ray_aligned_and_span_edge_to_background(scene):
    m = scene.kind == sp.KIND_CARD
    sc = scene.sc
    mu = sc.means[m].astype(np.float64)
    s = sc.scales[m].astype(np.float64)
    R = quat_to_rotmat(sc.quats[m])
    order = np.argsort(-s, axis=1)
    a1 = np.take_along_axis(R, order[:, None, :1], axis=2)[..., 0]
    ray = mu / np.linalg.norm(mu, axis=1, keepdims=True)
    assert (np.abs((a1 * ray).sum(1)) > 0.999).all()
    s_sorted = np.take_along_axis(s, order, axis=1)
    assert (s_sorted[:, 0] > 4 * s_sorted[:, 1]).all() and (s_sorted[:, 0] > 0.01).all()
    # mu - 2 s1 ray is the body edge, mu + 2 s1 ray the background behind it
    near = mu - 2 * s_sorted[:, :1] * ray
    np.testing.assert_allclose(near[:, 2], scene.surface_z[m], atol=1e-5)
    G = scene.cam.grid
    behind = scene.sc.means[G * G + scene.cell[m], 2]  # layer 1 of the same cell
    far = mu + 2 * s_sorted[:, :1] * ray
    np.testing.assert_allclose(far[:, 2], behind, rtol=1e-5)
    assert np.isin(scene.kind[G * G + scene.cell[m]], [sp.KIND_HIDDEN_BG, sp.KIND_HIDDEN_BODY]).all()
    # cards sit on the silhouette
    u, v, _ = sp.project_points(scene.cam, mu)
    hard = (scene.alpha >= 0.5).astype(np.uint8)
    edge = cv2.dilate(hard, np.ones((5, 5), np.uint8)) - cv2.erode(hard, np.ones((5, 5), np.uint8))
    assert edge[v.astype(int), u.astype(int)].mean() > 0.95


def test_card_options(body, cam):
    none = sp.sharp_like_scene(body, cam, cards=False)
    assert not (none.kind == sp.KIND_CARD).any()
    every = sp.sharp_like_scene(body, cam, card_frac=1.0)
    half = sp.sharp_like_scene(body, cam)
    n_all, n_half = (every.kind == sp.KIND_CARD).sum(), (half.kind == sp.KIND_CARD).sum()
    assert 0.3 * n_all < n_half < 0.7 * n_all


def test_surface_gaussians_are_flat_discs_on_the_true_normal(scene):
    m = np.isin(scene.kind, [sp.KIND_BODY, sp.KIND_WALL, sp.KIND_HIDDEN_BODY])
    sc = scene.sc
    np.testing.assert_allclose(np.linalg.norm(sc.quats, axis=1), 1.0, atol=1e-6)
    R = quat_to_rotmat(sc.quats[m])
    s = sc.scales[m]
    smallest = np.take_along_axis(R, np.argmin(s, axis=1)[:, None, None], axis=2)[..., 0]
    assert (np.abs((smallest * scene.normal[m]).sum(1)) > 0.999).all()
    assert (s.min(1) <= 0.11 * s.max(1)).all()
    # on-screen size ~0.6 cell on surfaces facing the camera
    face = m & (np.abs(scene.normal[:, 2]) > 0.95)
    f, cw = scene.cam.f, scene.cam.width / scene.cam.grid
    px = sc.scales[face, 0] * f / sc.means[face, 2]
    np.testing.assert_allclose(np.median(px), 0.6 * cw, rtol=0.05)


def test_seed_jitter_layers_and_switches(body, cam, scene):
    again = sp.sharp_like_scene(body, cam)
    np.testing.assert_array_equal(again.sc.means, scene.sc.means)
    np.testing.assert_array_equal(again.kind, scene.kind)
    other = sp.sharp_like_scene(body, cam, seed=1)
    assert not np.array_equal(other.sc.means, scene.sc.means)
    one = sp.sharp_like_scene(body, cam, layers=1)
    assert len(one.sc) == scene.cam.grid**2 and (one.layer == 0).all()
    bare = sp.sharp_like_scene(body, cam, wall=False, floor=False)
    assert not np.isin(bare.kind, [sp.KIND_FLOOR, sp.KIND_WALL, sp.KIND_HIDDEN_BG, sp.KIND_BG_LAYER1]).any()
    assert len(bare.sc) < len(scene.sc)
    assert (sp.classify_rgb(bare.rgb[bare.alpha == 0]) == "sky").all()
    no_wall = sp.sharp_like_scene(body, cam, wall=False)
    assert (no_wall.kind == sp.KIND_FLOOR).any() and not (no_wall.kind == sp.KIND_WALL).any()
    with pytest.raises(ValueError):
        sp.sharp_like_scene(body, cam, layers=3)


# --------------------------------------------------------------------------- front model
def test_front_model_maps_are_consistent(body, cam, scene):
    fm = sp.synth_front_model(body, cam)
    g = fm.grid
    assert (g.Wm, g.Hm, g.cell_px) == (96, 128, 1.25)
    shape = (g.Hm, g.Wm)
    for name in ("matte", "ring", "D_f", "D_near", "D_far", "D_ext", "A_f", "cos_theta", "dt"):
        assert getattr(fm, name).shape == shape, name
    assert fm.normals.shape == (*shape, 3) and fm.rgb.shape == (*shape, 3) and fm.rgb.dtype == np.uint8
    matte, ring = fm.matte, fm.ring
    band = matte | ring
    assert matte.dtype == bool and not (matte & ring).any() and ring.sum() > 0.2 * matte.sum()
    # the matte is the cell-centre projection of the body and agrees with the photo matte
    small = cv2.resize(scene.alpha, (g.Wm, g.Hm), interpolation=cv2.INTER_AREA) >= 0.5
    assert (small & matte).sum() / (small | matte).sum() > 0.93
    for name in ("D_f", "D_near", "D_far"):
        d = getattr(fm, name)
        assert np.isfinite(d[band]).all() and np.isnan(d[~band]).all(), name
    assert np.isfinite(fm.D_ext).all()
    np.testing.assert_array_equal(fm.D_ext[matte], fm.D_near[matte])
    assert (fm.D_near[band] <= fm.D_f[band] + 1e-6).all() and (fm.D_f[band] <= fm.D_far[band] + 1e-6).all()
    assert 2.3 < np.median(fm.D_f[matte]) < 2.5
    np.testing.assert_array_equal(fm.A_f, matte.astype(np.float32))
    n = fm.normals[matte]
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
    assert np.isnan(fm.normals[~matte]).all() and np.isnan(fm.cos_theta[~matte]).all()
    assert (fm.cos_theta[matte] > -1e-6).all() and np.median(fm.cos_theta[matte]) > 0.8
    assert (fm.dt[matte] >= 1).all() and (fm.dt[~matte] == 0).all()
    # a disocclusion edge exists: the forearm in front of the torso
    assert ((fm.D_far - fm.D_f)[matte] > 0.05).sum() > 50
    # per-Gaussian data
    sc = fm.sc
    N = len(sc)
    assert N == matte.sum() and fm.cell.shape == fm.theta_deg.shape == fm.rim.shape == (N,)
    u, v, z = sp.project_points(cam, sc.means)
    uc, vc = sp.grid_centers(g)
    np.testing.assert_allclose(u, uc.ravel()[fm.cell], atol=1e-3)
    np.testing.assert_allclose(v, vc.ravel()[fm.cell], atol=1e-3)
    np.testing.assert_allclose(z, fm.D_f.ravel()[fm.cell], atol=1e-5)
    assert matte.ravel()[fm.cell].all()
    np.testing.assert_allclose(np.cos(np.radians(fm.theta_deg)), fm.cos_theta.ravel()[fm.cell], atol=1e-4)
    np.testing.assert_array_equal(fm.rim, fm.dt.ravel()[fm.cell] <= 2.0)
    assert 0.2 < fm.rim.mean() < 0.7
    assert fm.stats["synthetic"] and fm.cam is cam
    # the ground-truth front depth agrees with the SHARP-like layer 0 on the subject
    mb = scene.kind == sp.KIND_BODY
    ub, vb, zb = sp.project_points(cam, scene.sc.means[mb])
    cells = (vb / g.cell_px).astype(int) * g.Wm + (ub / g.cell_px).astype(int)
    ok = matte.ravel()[cells]
    assert np.median(np.abs(fm.D_f.ravel()[cells[ok]] - zb[ok])) < 0.01
    half = sp.synth_front_model(body, cam, stride=2)
    assert 0.2 * N < len(half.sc) < 0.3 * N
    # the true back along the same rays
    back = sp.true_back_depth(body, cam)
    assert (np.isfinite(back) == matte).all() and (back[matte] > fm.D_f[matte]).all()
    torso = body.centers[body.part("torso")] + body.R_cb @ [0.0, -0.15, 0.0]  # chest, above the forearm
    tu, tv, _ = sp.project_points(cam, torso)
    r, c = int(tv / g.cell_px), int(tu / g.cell_px)
    assert back[r, c] - fm.D_f[r, c] == pytest.approx(0.22 * math.sqrt(1 - (0.15 / 0.36) ** 2), abs=0.01)


# --------------------------------------------------------------------------- canonical frame
@pytest.mark.parametrize("pitch", [0.0, 12.0])
def test_canon_truth_and_floor(cam, pitch):
    body = sp.make_body(pitch_deg=pitch, yaw_deg=20.0)
    cn = sp.synth_canon(body, cam, thickness=0.12)
    R = cn.R
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(cn.up_C, -R[:, 1])
    tilt = math.degrees(math.acos(float(np.clip(cn.up_C @ [0.0, -1.0, 0.0], -1, 1))))
    assert tilt == pytest.approx(pitch, abs=1e-9) and cn.pitch_deg == pitch
    assert cn.up_C[2] <= 0  # looking down: world up leans back toward the camera
    assert cn.T_prior == pytest.approx(0.12 * cn.height_s) and cn.up_source == "truth"
    assert not (cn.cut_top or cn.cut_bottom)
    # the body axis is on the orbit-axis depth, the floor is a level plane in S
    np.testing.assert_allclose(cn.to_S(body.origin)[2], 0.0, atol=1e-12)
    np.testing.assert_allclose(cn.to_C(cn.to_S(body.origin)), body.origin, atol=1e-12)
    s = sp.sharp_like_scene(body, cam)
    floor = s.sc.means[s.kind == sp.KIND_FLOOR].astype(np.float64)
    yS = cn.to_S(floor)[:, 1]
    assert np.ptp(yS) < 1e-4 and yS.mean() == pytest.approx(cn.height_s / 2 + 0.01, abs=1e-4)
    normal, _, frac = fit_plane_ransac(floor, 0.01)
    assert frac > 0.99 and abs(normal @ cn.up_C) > math.cos(math.radians(0.2))


def test_body_placement_options(cam):
    low = sp.make_body(distance=5.3, cam_height=2.32, lateral=-0.3)
    assert low.wall_z == pytest.approx(6.3)
    _, a, rgb = sp.sharp_like_cloud(low, sp.make_cam(640, 480, 65.0))
    rows, cols = np.nonzero(a >= 0.5)
    assert 290 < rows.min() < 310 and 450 < rows.max() < 470  # e.g. a box matte at rows 300-460
    assert cols.mean() < 320
    cn = sp.synth_canon(low, cam)
    assert cn.z_axis == pytest.approx(5.3) and cn.height_s == pytest.approx(sp.BODY_HEIGHT)


def test_canon_cut_flags():
    cam = sp.make_cam(120, 100, 40.0)  # same focal length range, shorter frame: head and feet cut
    cn = sp.synth_canon(sp.make_body(), cam)
    assert cn.cut_top and cn.cut_bottom


# --------------------------------------------------------------------------- orthographic views
def test_ortho_view_sides_and_mirror(body, canon):
    N, m = 200, 0.01
    views = {phi: sp.ortho_view(body, canon, phi, m, N) for phi in (0, 90, 180, 270)}
    for phi, side in ((0, "front"), (90, "left"), (180, "back"), (270, "right")):
        rgb, a, depth = views[phi]
        assert rgb.shape == (N, N, 3) and a.shape == depth.shape == (N, N)
        names = sp.classify_rgb(rgb[a == 1.0], sp.SIDES)
        u, c = np.unique(names, return_counts=True)
        assert u[np.argmax(c)] == side and _frac(names, side) > (0.75 if phi % 180 == 0 else 0.5), phi
        assert np.isnan(depth[a == 0]).all()
    # orthographic invariant: the back silhouette is the mirrored front silhouette
    a0, a180 = views[0][1] >= 0.5, views[180][1] >= 0.5
    assert (a0 & a180[:, ::-1]).sum() / (a0 | a180[:, ::-1]).sum() > 0.99
    # the raised (left) arm is on the image right from the front and on the left from behind,
    # the bag the other way round, and the bag's back is blue
    arm = body.centers[body.part("arm_left")]
    bag = body.centers[body.part("bag")] + body.R_cb @ [0.0, 0.0, 0.12]  # on the bag's back
    u0, _, _ = sp.ortho_project(canon, 0, m, N, arm)
    u180, _, _ = sp.ortho_project(canon, 180, m, N, arm)
    assert u0 > N / 2 + 20 and u180 < N / 2 - 20
    ub, vb, _ = sp.ortho_project(canon, 180, m, N, bag)
    assert ub > N / 2 + 20
    rgb180, a180f, _ = views[180]
    assert a180f[int(vb), int(ub)] == 1.0 and sp.classify_rgb(rgb180[int(vb), int(ub)]) == "back"
    ub0, _, _ = sp.ortho_project(canon, 0, m, N, bag)
    assert ub0 < N / 2 - 20
    # depth: front at phi 0 + back at phi 180 = the body's thickness there (chest: 0.19 m)
    chest = body.from_B([0.0, -1.30, 0.0])
    uc, vc, _ = sp.ortho_project(canon, 0, m, N, chest)
    i, j = int(vc), int(uc)
    thick = -views[0][2][i, j] - views[180][2][i, N - 1 - j]
    assert thick == pytest.approx(2 * 0.11 * math.sqrt(1 - (0.18 / 0.36) ** 2), abs=0.01)
    # phi = 90 is "walked to your right": the view's right is z^ (away from the photo camera)
    right, down, fwd = sp.ortho_axes(canon, 90)
    np.testing.assert_allclose(right, canon.R[:, 2], atol=1e-12)
    np.testing.assert_allclose(fwd, -canon.R[:, 0], atol=1e-12)
    np.testing.assert_allclose(np.cross(right, down), fwd, atol=1e-12)


def test_everything_is_fast(body, cam):
    t0 = time.perf_counter()
    s = sp.sharp_like_scene(body, cam)
    fm = sp.synth_front_model(body, cam)
    cn = sp.synth_canon(body, cam)
    for phi in range(0, 360, 45):
        sp.ortho_view(body, cn, phi)
    assert len(s.sc) and len(fm.sc)
    assert time.perf_counter() - t0 < 4.0
