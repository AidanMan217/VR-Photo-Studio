"""SHARP 360 front clean-up and canonical frame (geometry/splat_fusion.py, design 3.0-3.3).

Uses the analytic synthetic person (tests/geometry/synth_person.py): a SHARP-like two-layer
cloud with labelled Gaussians (visible body, hidden body behind the forearm, wall blobs
behind the body, edge cards, ...) and the exact ground-truth maps and frame.
"""

from __future__ import annotations

import dataclasses
import math
import time

import cv2
import numpy as np
import pytest
import synth_person as sp

from vrphoto.core.errors import Cancelled, InputError, VRPhotoError
from vrphoto.core.types import CancelToken, SplatCloud
from vrphoto.geometry import splat_fusion as sf
from vrphoto.geometry.splat import concat_splats, quat_to_rotmat, rotmat_to_quat
from vrphoto.geometry.splat_fusion import MouldGrid, SharpCam, canonical_frame, depth_tol, sharp_front

MOULD = 128  # mould long side for the 120 x 160 test photo (cell = 1.25 px)


@pytest.fixture(scope="module")
def body() -> sp.Body:
    return sp.make_body()


@pytest.fixture(scope="module")
def cam() -> SharpCam:
    return sp.make_cam()


@pytest.fixture(scope="module")
def scene(body, cam) -> sp.SharpScene:
    return sp.sharp_like_scene(body, cam)


@pytest.fixture(scope="module")
def front(scene) -> sf.FrontModel:
    return sharp_front(scene.sc, scene.alpha, scene.cam, scene.rgb, mould_long=MOULD)


@pytest.fixture(scope="module")
def truth(body, cam) -> sf.FrontModel:
    return sp.synth_front_model(body, cam, MOULD)


@pytest.fixture(scope="module", params=[0, 2, 5, 7], ids=lambda s: f"seed{s}")
def seeded(body, cam, request) -> tuple[sp.SharpScene, sf.FrontModel]:
    """``(scene, cleaned front)`` for a few fixture seeds (jitter and which edge cells
    become cards), so the clean-up checks do not depend on one draw."""
    scn = sp.sharp_like_scene(body, cam, seed=request.param)
    return scn, sharp_front(scn.sc, scn.alpha, scn.cam, scn.rgb, mould_long=MOULD)


def _kept(front: sf.FrontModel, n: int) -> np.ndarray:
    k = np.zeros(n, bool)
    k[front.src] = True
    return k


def _dilated(scene: sp.SharpScene) -> tuple[np.ndarray, np.ndarray]:
    """``(hard matte, matte dilated by 1 % of the long side)`` in photo pixels."""
    hard = scene.alpha >= 0.5
    r = max(2, round(0.01 * max(scene.cam.width, scene.cam.height)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return hard, cv2.dilate(hard.astype(np.uint8), ker) > 0


def _pixel_masks(scene: sp.SharpScene) -> tuple[np.ndarray, np.ndarray]:
    """Per Gaussian: centre on the hard matte / on the 1 % ring around it (photo pixels)."""
    cam = scene.cam
    hard, dil = _dilated(scene)
    u, v, _ = sp.project_points(cam, scene.sc.means)
    ok = np.isfinite(u) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    ui = np.clip(np.floor(np.nan_to_num(u)), 0, cam.width - 1).astype(int)
    vi = np.clip(np.floor(np.nan_to_num(v)), 0, cam.height - 1).astype(int)
    return ok & hard[vi, ui], ok & dil[vi, ui] & ~hard[vi, ui]


def _footprint(cam: SharpCam, sc: SplatCloud, rows: np.ndarray) -> np.ndarray:
    """Projected 2-D covariance (pinhole Jacobian) of the Gaussians ``rows``."""
    R = quat_to_rotmat(sc.quats[rows])
    S = np.asarray(sc.scales[rows], np.float64)
    C = (R * S[:, None, :] ** 2) @ np.swapaxes(R, 1, 2)
    x, y, z = np.asarray(sc.means[rows], np.float64).T
    J = np.zeros((len(rows), 2, 3))
    J[:, 0, 0] = J[:, 1, 1] = cam.f / z
    J[:, 0, 2] = -cam.f * x / z**2
    J[:, 1, 2] = -cam.f * y / z**2
    return J @ C @ np.swapaxes(J, 1, 2)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# --------------------------------------------------------------------------- SharpCam
def test_from_outputs_uses_the_workers_camera():
    cam = SharpCam.from_outputs(
        {"f_px": 1234.5, "width": 1536, "height": 1024},
        {"hfov_deg": 40.0},
        n_raw=2 * 768**2,
        shape=(512, 768),
    )
    assert (cam.f, cam.width, cam.height) == (1234.5, 1536, 1024)
    assert (cam.n_raw, cam.grid, cam.layers) == (2 * 768**2, 768, 2)


@pytest.mark.parametrize(
    ("outs", "params", "f_expected"),
    [
        ({}, {"hfov_deg": 50.0}, 400.0 / math.tan(math.radians(25.0))),  # no f_px: the requested FOV
        ({"f_px": None}, {}, 400.0 / math.tan(math.radians(32.5))),  # nothing: 65 deg default
        ({"f_px": "abc", "width": "x"}, {"hfov_deg": "nope"}, 400.0 / math.tan(math.radians(32.5))),
        ({"f_px": float("nan")}, {"hfov_deg": -3}, 400.0 / math.tan(math.radians(32.5))),
        ({"f_px": 0, "width": 0, "height": -5}, {"hfov_deg": 90.0}, 400.0),
    ],
)
def test_from_outputs_falls_back_like_subject_splats(outs, params, f_expected):
    cam = SharpCam.from_outputs(outs, params, n_raw=100, shape=(600, 800))
    assert (cam.width, cam.height) == (800, 600)  # the photo's shape
    assert cam.f == pytest.approx(f_expected, rel=1e-9)


def test_from_outputs_width_without_f_uses_the_workers_width():
    cam = SharpCam.from_outputs({"width": 1000, "height": 750}, {"hfov_deg": 60.0}, n_raw=0, shape=(300, 400))
    assert (cam.width, cam.height) == (1000, 750)
    assert cam.f == pytest.approx(500.0 / math.tan(math.radians(30.0)))


@pytest.mark.parametrize(
    ("n_raw", "grid", "layers"),
    [
        (2 * 768**2, 768, 2),
        (768**2, 768, 1),
        (3 * 768**2, 768, 3),
        (int(2 * 768**2 * 0.99), 768, 2),  # a few Gaussians dropped by the worker
        (2 * 96**2, 96, 2),
        (1000, 22, 2),
        (0, 768, 2),
    ],
)
def test_grid_detection(n_raw, grid, layers):
    cam = SharpCam.from_outputs({"f_px": 100.0}, {}, n_raw=n_raw, shape=(10, 10))
    assert (cam.grid, cam.layers) == (grid, layers)


def test_project_rays_and_sharp_cells_round_trip():
    cam = SharpCam(f=300.0, width=640, height=480, n_raw=0, grid=64)
    rng = np.random.default_rng(3)
    u = rng.uniform(0, 640, 500)
    v = rng.uniform(0, 480, 500)
    z = rng.uniform(0.5, 9.0, 500)
    P = cam.rays(u, v) * z[:, None]
    assert np.all(cam.rays(u, v)[:, 2] == 1.0)
    u2, v2, z2 = cam.project(P)
    np.testing.assert_allclose(np.c_[u2, v2, z2], np.c_[u, v, z], atol=1e-9)
    assert np.isnan(cam.project(np.array([[0.1, 0.2, 0.0], [1.0, 1.0, -2.0]]))[0]).all()  # z <= 0
    cell = cam.sharp_cell(u, v)
    np.testing.assert_array_equal(
        cell, np.floor(v * 64 / 480).astype(int) * 64 + np.floor(u * 64 / 640).astype(int)
    )
    assert cam.sharp_cell(np.array([-0.1, 640.0, np.nan]), np.array([1.0, 1.0, 1.0])).tolist() == [-1, -1, -1]
    assert cam.project(np.zeros((2, 3, 3)))[0].shape == (2, 3)  # any leading shape


# --------------------------------------------------------------------------- MouldGrid
@pytest.mark.parametrize(
    ("w", "h", "long_side", "dims"),
    [
        (120, 160, 128, (96, 128, 1.25)),
        (4000, 3000, 768, (768, 576, 4000 / 768)),
        (640, 480, 1024, (640, 480, 1.0)),  # never finer than the photo
        (3000, 4000, 512, (384, 512, 4000 / 512)),
        (1001, 7, 64, (64, 1, 1001 / 64)),
    ],
)
def test_mould_grid_for_photo(w, h, long_side, dims):
    g = MouldGrid.for_photo(w, h, long_side)
    assert (g.Wm, g.Hm) == dims[:2]
    assert g.cell_px == pytest.approx(dims[2])
    assert g == sp.mould_grid(w, h, long_side)  # the fixture's grid is the same


@pytest.mark.parametrize("bad", [(0, 10, 10), (10, 10, 0), (10, -1, 10), ("a", 10, 10)])
def test_mould_grid_rejects_bad_sizes(bad):
    with pytest.raises(InputError):
        MouldGrid.for_photo(*bad)


def test_mould_grid_round_trips():
    g = MouldGrid.for_photo(300, 200, 120)
    u, v = g.centers()
    assert u.shape == v.shape == (g.Hm, g.Wm)
    np.testing.assert_array_equal(g.cell_of(u, v), np.arange(g.Hm * g.Wm).reshape(g.Hm, g.Wm))
    # every point of a cell maps to it; outside the grid is -1
    c = g.cell_px
    np.testing.assert_array_equal(g.cell_of(u - 0.499 * c, v + 0.499 * c), g.cell_of(u, v))
    out = g.cell_of(np.array([-1e-6, g.Wm * c, 5.0, np.nan]), np.array([5.0, 5.0, g.Hm * c, 5.0]))
    assert out.tolist() == [-1, -1, -1, -1]
    # resampling: constants stay constant, channels and bool maps are handled
    img = np.full((200, 300, 3), 77, np.uint8)
    r = g.resize(img)
    assert r.shape == (g.Hm, g.Wm, 3) and r.dtype == np.uint8 and np.all(r == 77)
    one = g.resize(np.ones((200, 300, 1), np.float32))
    assert one.shape == (g.Hm, g.Wm, 1)
    m = np.zeros((200, 300), bool)
    m[:, :150] = True
    rm = g.resize(m)
    assert rm.dtype == np.float32 and rm[:, : g.Wm // 2].min() == 1.0 and rm[:, g.Wm // 2 :].max() == 0.0
    same = MouldGrid.for_photo(300, 200, 300)
    x = np.random.default_rng(0).random((200, 300)).astype(np.float32)
    np.testing.assert_array_equal(same.resize(x), x)


# --------------------------------------------------------------------------- front clean-up
def test_hidden_layer_and_wall_blobs_inside_the_matte_are_dropped(scene, front):
    kept = _kept(front, len(scene.sc))
    in_matte, _ = _pixel_masks(scene)
    bg = in_matte & scene.mask(sp.KIND_HIDDEN_BG, sp.KIND_BG_LAYER1, sp.KIND_WALL)
    assert bg.sum() > 500
    assert (bg & ~kept).sum() >= 0.95 * bg.sum()
    surface = in_matte & scene.mask(sp.KIND_BODY)
    assert surface.sum() > 500
    assert (surface & kept).sum() >= 0.98 * surface.sum()
    assert not front.stats["fallback"]
    assert front.stats["kept"] == len(front.sc) == len(front.src)
    assert np.all(np.diff(front.src) > 0)  # SHARP's row order


def test_torso_behind_the_forearm_is_kept(body, scene, front):
    kept = _kept(front, len(scene.sc))
    behind_arm = scene.mask(sp.KIND_HIDDEN_BODY) & (scene.part == body.part("torso"))
    assert behind_arm.sum() >= 15
    assert (behind_arm & kept).sum() >= 0.9 * behind_arm.sum()
    assert front.stats["hidden_kept"] >= behind_arm.sum() * 0.9


def test_cards_are_collapsed_along_their_ray(cam, seeded):
    scene, front = seeded
    cards = np.flatnonzero(scene.mask(sp.KIND_CARD))
    in_matte, ring = _pixel_masks(scene)
    rows = np.flatnonzero(np.isin(front.src, cards))
    orig = front.src[rows]
    assert len(rows) >= 0.95 * (in_matte | ring)[cards].sum()
    # same pixel, same on-screen footprint as SHARP's card ...
    u0, v0, z0 = sp.project_points(cam, scene.sc.means[orig])
    u1, v1, z1 = sp.project_points(cam, front.sc.means[rows])
    assert np.abs(u1 - u0).max() < 0.01 and np.abs(v1 - v0).max() < 0.01
    e0 = np.sqrt(np.linalg.eigvalsh(_footprint(cam, scene.sc, orig)))
    e1 = np.sqrt(np.linalg.eigvalsh(_footprint(cam, front.sc, rows)))
    assert np.abs(e1 / e0 - 1.0).max() < 0.02
    # ... but now on the subject instead of halfway to the background, and no longer a streak
    # (a card next to the forearm lands on D_near, the 5x5 nearest depth, by design: bound
    # the bulk, not the worst one)
    gap0 = z0 - scene.surface_z[orig]
    gap1 = np.abs(z1 - scene.surface_z[orig])
    assert np.median(gap0) > 0.3 and np.median(gap1) < 0.06 and np.percentile(gap1, 95) < 0.1
    s = np.sort(front.sc.scales[rows], axis=1)
    assert np.all(s[:, 2] <= 1.5 * s[:, 1] + 1e-6)
    assert front.stats["cards_collapsed"] >= len(rows)


def test_ring_keeps_wisps_at_subject_depth_and_drops_background(cam, scene, front, truth):
    kept = _kept(front, len(scene.sc))
    _, ring = _pixel_masks(scene)
    wall = ring & scene.mask(sp.KIND_WALL, sp.KIND_HIDDEN_BG)
    assert wall.sum() > 100 and (wall & kept).sum() == 0
    edge = ring & scene.mask(sp.KIND_BODY)
    assert edge.sum() >= 10 and (edge & kept).sum() >= 0.75 * edge.sum()  # coarse grid: see the wisps below

    # inject translucent hair wisps in the ring: at the subject's depth and at the wall
    hard, dil = _dilated(scene)
    rr, cc = np.nonzero(dil & ~hard)
    u, v = cc + 0.5, rr + 0.5
    z_subject = truth.D_f.ravel()[truth.grid.cell_of(u, v)]  # ground truth, extrapolated into the ring
    ok = np.isfinite(z_subject)
    u, v, z_subject = u[ok], v[ok], z_subject[ok]
    rays = sp.pixel_rays(cam, u, v)
    n = len(u)
    pts = np.r_[rays * z_subject[:, None], rays * (z_subject[:, None] + 0.5)]
    wisps = SplatCloud(
        means=pts.astype(np.float32),
        scales=np.full((2 * n, 3), 0.004, np.float32),
        quats=np.tile(np.float32([1, 0, 0, 0]), (2 * n, 1)),
        colors=np.full((2 * n, 3), 0.3, np.float32),
        opacity=np.full(2 * n, 0.4, np.float32),
    )
    sc = concat_splats([scene.sc, wisps])
    fm = sharp_front(sc, scene.alpha, scene.cam, scene.rgb, mould_long=MOULD)
    k = _kept(fm, len(sc))[len(scene.sc) :]
    assert k[:n].mean() >= 0.9  # at the subject's depth
    assert k[n:].mean() <= 0.02  # 50 cm behind it


def test_translucent_floaters_in_front_do_not_define_the_surface(seeded):
    """Front depth by transmittance: a 30 %-opaque haze 40 cm in front of the body neither
    becomes the surface nor survives (design 0.3, 'Front depth'), and it does not change
    which body Gaussians are kept."""
    scene, front = seeded
    body_rows = np.flatnonzero(scene.mask(sp.KIND_BODY))[::7]
    P = scene.sc.means[body_rows].astype(np.float64)
    haze = P * ((P[:, 2] - 0.4) / P[:, 2])[:, None]
    m = len(haze)
    fl = SplatCloud(
        means=haze.astype(np.float32),
        scales=np.full((m, 3), 0.005, np.float32),
        quats=np.tile(np.float32([1, 0, 0, 0]), (m, 1)),
        colors=np.ones((m, 3), np.float32),
        opacity=np.full(m, 0.3, np.float32),
    )
    sc = concat_splats([scene.sc, fl])
    fm = sharp_front(sc, scene.alpha, scene.cam, scene.rgb, mould_long=MOULD)
    k = _kept(fm, len(sc))
    assert k[len(scene.sc) :].sum() == 0
    k0 = _kept(front, len(scene.sc))  # without the haze
    assert (k[body_rows] == k0[body_rows]).all() and k0[body_rows].mean() > 0.97
    assert fm.stats["floaters_dropped"] >= 0.95 * m


def _tilted_card_scene(cos_tilt: tuple[float, ...]):
    """A flat subject (a plane 2 m away, a 3 m wall behind) with SHARP-like ray cards in its
    interior whose long axis is tilted off the ray (``|a1 . ray|`` = each of ``cos_tilt``).
    Returns ``(sc, alpha, cam, card rows)``."""
    W, H, G = 160, 120, 64
    cam = SharpCam(f=150.0, width=W, height=H, n_raw=2 * G * G, grid=G, layers=2)
    rows, cols = np.divmod(np.arange(G * G), G)
    u, v = (cols + 0.5) * W / G, (rows + 0.5) * H / G
    subject = (np.abs(u - W / 2) < W / 4) & (np.abs(v - H / 2) < H / 4)
    d = cam.rays(u, v)
    z0 = np.where(subject, 2.0, 3.0)
    n = G * G
    means = np.r_[d * z0[:, None], d * np.where(subject, 3.0, 3.03)[:, None]]
    scales = np.full((2 * n, 3), (0.004, 0.004, 0.0004))
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (2 * n, 1))
    # cards in every 9th interior subject cell, replacing its surface Gaussian
    inner = np.flatnonzero(subject & (np.abs(u - W / 2) < W / 5) & (np.abs(v - H / 2) < H / 5))[::9]
    rng = np.random.default_rng(1)
    cards = []
    for j, c in enumerate(inner):
        ct = cos_tilt[j % len(cos_tilt)]
        r = d[c] / np.linalg.norm(d[c])
        side = np.cross(r, rng.normal(size=3))
        side /= np.linalg.norm(side)
        a1 = ct * r + math.sqrt(1 - ct * ct) * side
        a2 = np.cross(a1, rng.normal(size=3))
        a2 /= np.linalg.norm(a2)
        s1 = 0.12
        means[c] = r * (2.0 + 0.8 * s1 * ct) / r[2]  # its near end reaches the plane
        scales[c] = (s1, 0.006, 0.003)
        quats[c] = rotmat_to_quat(np.column_stack([a1, a2, np.cross(a1, a2)])[None])[0]
        cards.append(c)
    sc = SplatCloud(
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        quats=quats.astype(np.float32),
        colors=np.full((2 * n, 3), 0.5, np.float32),
        opacity=np.full(2 * n, 0.9, np.float32),
    )
    alpha = np.zeros((H, W), np.float32)
    alpha[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0
    return sc, alpha, cam, np.array(cards)


def test_tilted_cards_keep_their_photo_footprint():
    """A card the test accepts can be 37 deg off its ray; collapsing it must keep what the
    photo's camera sees of it (the across-ray part of its long axis), not only for cards
    lying exactly on the ray."""
    tilts = (0.999, 0.95, 0.9, 0.85)
    sc, alpha, cam, cards = _tilted_card_scene(tilts)
    fm = sharp_front(sc, alpha, cam, mould_long=128)
    rows = np.flatnonzero(np.isin(fm.src, cards))
    assert len(rows) == len(cards) and fm.stats["cards_collapsed"] == len(cards)
    orig = fm.src[rows]
    u0, v0, z0 = sp.project_points(cam, sc.means[orig])
    u1, v1, z1 = sp.project_points(cam, fm.sc.means[rows])
    assert np.abs(u1 - u0).max() < 0.01 and np.abs(v1 - v0).max() < 0.01
    np.testing.assert_allclose(z1, 2.0, atol=0.01)  # slid onto the plane
    e0 = np.sqrt(np.linalg.eigvalsh(_footprint(cam, sc, orig)))
    e1 = np.sqrt(np.linalg.eigvalsh(_footprint(cam, fm.sc, rows)))
    assert np.abs(e1 / e0 - 1.0).max() < 0.02, (e1 / e0).min()
    # ... and squashed along the ray to 1.5 x its (rescaled) middle axis
    R = quat_to_rotmat(fm.sc.quats[rows])
    S = fm.sc.scales[rows].astype(np.float64)
    C = (R * S[:, None, :] ** 2) @ np.swapaxes(R, 1, 2)
    P = fm.sc.means[rows].astype(np.float64)
    r = P / np.linalg.norm(P, axis=1, keepdims=True)
    along = np.sqrt(np.einsum("ni,nij,nj->n", r, C, r))
    k = z1 / z0
    np.testing.assert_allclose(along, 1.5 * 0.006 * k, rtol=1e-3)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_a_small_full_body_figure_is_cleaned_not_filtered():
    """A person 5.5 m away in a landscape photo covers ~1.5 % of it: the clean-up keeps about
    one Gaussian per subject cell, far below 1 % of SHARP's two-layer whole-photo cloud, and
    must not fall back to the plain matte filter (which keeps wall blobs and edge streaks)."""
    b = sp.make_body(distance=5.5)
    scn = sp.sharp_like_scene(b, sp.make_cam(320, 240, 65.0, grid=256), G=256)
    assert 0.01 < (scn.alpha >= 0.5).mean() < 0.02
    fm = sharp_front(scn.sc, scn.alpha, scn.cam, scn.rgb, mould_long=256)
    assert not fm.stats["fallback"] and not fm.stats["warnings"]
    assert len(fm.sc) < 0.01 * len(scn.sc) and fm.stats["fallback_below"] < len(fm.sc)
    kept = _kept(fm, len(scn.sc))
    in_matte, _ = _pixel_masks(scn)
    assert kept[in_matte & scn.mask(sp.KIND_BODY)].mean() > 0.97
    bg = in_matte & scn.mask(sp.KIND_HIDDEN_BG, sp.KIND_BG_LAYER1, sp.KIND_WALL)
    assert kept[bg].mean() < 0.05


def test_background_close_behind_a_small_subject_is_dropped(body, cam, scene):
    """The same photo of a 34 cm figurine 0.5 m away with the wall 20 cm behind it (the whole
    scene scaled by 0.2): the gap test scales with the subject, so the wall seen around and
    through it is not trusted as front depth and its Gaussians do not survive as
    "disocclusion fill"."""
    small = dataclasses.replace(
        scene.sc, means=scene.sc.means * np.float32(0.2), scales=scene.sc.scales * np.float32(0.2)
    )
    fm = sharp_front(small, scene.alpha, scene.cam, scene.rgb, mould_long=MOULD)
    assert fm.stats["gap_test_m"] < 0.07 and not fm.stats["fallback"]
    kept = _kept(fm, len(small))
    in_matte, _ = _pixel_masks(scene)
    bg = in_matte & scene.mask(sp.KIND_HIDDEN_BG, sp.KIND_BG_LAYER1, sp.KIND_WALL)
    assert (bg & ~kept).sum() >= 0.95 * bg.sum()
    assert kept[scene.mask(sp.KIND_HIDDEN_BG)].sum() < 0.05 * scene.mask(sp.KIND_BODY).sum()
    canon = canonical_frame(fm, small, scene.alpha)
    assert canon.height_s == pytest.approx(0.2 * sp.synth_canon(body, cam).height_s, rel=0.08)


def test_front_maps_match_the_truth(front, truth):
    g = front.grid
    assert g == truth.grid
    for name in ("matte", "ring", "D_f", "D_near", "D_far", "D_ext", "A_f", "cos_theta", "dt"):
        assert getattr(front, name).shape == (g.Hm, g.Wm), name
    assert front.normals.shape == (g.Hm, g.Wm, 3) and front.rgb.shape == (g.Hm, g.Wm, 3)
    assert front.rgb.dtype == np.uint8
    iou = (front.matte & truth.matte).sum() / (front.matte | truth.matte).sum()
    assert iou > 0.97
    assert not (front.matte & front.ring).any()
    band = front.matte | front.ring
    for name in ("D_f", "D_near", "D_far"):
        d = getattr(front, name)
        assert np.isfinite(d[band]).all() and np.isnan(d[~band]).all(), name
    assert np.isfinite(front.D_ext).all()
    assert np.all(front.D_near[band] <= front.D_f[band] + 1e-6)
    assert np.all(front.D_far[band] >= front.D_f[band] - 1e-6)
    inner = front.matte & truth.matte & (truth.dt > 3)
    err = np.abs(front.D_f - truth.D_f)[inner]
    assert np.median(err) < 0.005 and np.percentile(err, 90) < 0.03
    assert 0 <= front.A_f.min() and front.A_f.max() <= 1 and np.median(front.A_f[inner]) > 0.9
    # normals: unit, facing the camera, close to the truth inside the body
    n = front.normals[front.matte]
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-4)
    assert np.isnan(front.normals[~front.matte]).all() and np.isnan(front.cos_theta[~front.matte]).all()
    assert np.all(front.cos_theta[front.matte] >= 0)
    ang = np.degrees(np.arccos(np.clip((front.normals * truth.normals).sum(-1)[inner], -1, 1)))
    assert np.median(ang) < 10.0
    np.testing.assert_allclose(front.dt, truth.dt, atol=1.01)


def test_per_gaussian_cell_angle_and_rim(cam, front):
    u, v, _ = sp.project_points(cam, front.sc.means)
    np.testing.assert_array_equal(front.cell, front.grid.cell_of(u, v))
    assert front.cell.dtype == np.int64 and front.theta_deg.dtype == np.float32
    np.testing.assert_allclose(front.theta_deg, sf.angle_deg(front.cos_theta.ravel()[front.cell]), atol=1e-4)
    thr = max(2.0, 0.004 * max(front.grid.Hm, front.grid.Wm))
    np.testing.assert_array_equal(front.rim, front.dt.ravel()[front.cell] <= thr)
    off = ~front.matte.ravel()[front.cell]  # ring Gaussians: outline band, 90 deg
    assert off.any() and front.rim[off].all() and np.all(front.theta_deg[off] == 90.0)
    assert (~front.rim).sum() > 0.3 * len(front.rim)


def test_stats_are_json_friendly(front):
    import json

    s = json.loads(json.dumps(front.stats))
    assert s["grid"] == 96 and s["layers"] == 2 and s["n_raw"] == 2 * 96**2 == s["n_expected"]
    assert s["sharp_grid_match"] is False and s["fallback"] is False and s["warnings"] == []
    for key in ("kept", "hidden_dropped", "cards_collapsed", "ring_kept", "dropped"):
        assert isinstance(s[key], int)
    assert s["kept"] + s["dropped"] == s["n_raw"]


def test_grid_is_detected_when_the_camera_does_not_know_it(scene, front):
    c0 = SharpCam(f=scene.cam.f, width=scene.cam.width, height=scene.cam.height)  # n_raw = 0
    fm = sharp_front(scene.sc, scene.alpha, c0, scene.rgb, mould_long=MOULD)
    assert (fm.cam.grid, fm.cam.n_raw) == (96, len(scene.sc))
    np.testing.assert_array_equal(fm.src, front.src)


def test_matte_at_another_resolution(scene, front):
    big_a = cv2.resize(scene.alpha, (240, 320), interpolation=cv2.INTER_LINEAR)
    big_rgb = cv2.resize(scene.rgb, (240, 320), interpolation=cv2.INTER_LINEAR)
    fm = sharp_front(scene.sc, big_a, scene.cam, big_rgb, mould_long=MOULD)
    assert fm.grid == front.grid
    common = np.intersect1d(fm.src, front.src)
    assert len(common) >= 0.97 * len(front.src)


def test_fallback_to_the_plain_matte_filter(monkeypatch, scene, front):
    monkeypatch.setattr(sf, "_FALLBACK_MIN", front.stats["kept"] + 1)
    fm = sharp_front(scene.sc, scene.alpha, scene.cam, scene.rgb, mould_long=MOULD)
    assert fm.stats["fallback"] is True and fm.stats["fallback_kind"] == "subject_filter"
    assert fm.stats["warnings"] and isinstance(fm.stats["warnings"][0], str)
    assert len(fm.sc) > len(front.sc)
    in_matte, _ = _pixel_masks(scene)
    surface = np.flatnonzero(in_matte & scene.mask(sp.KIND_BODY))
    assert np.isin(surface, fm.src).all()
    np.testing.assert_array_equal(fm.sc.means, scene.sc.means[fm.src])  # untouched rows
    assert np.isfinite(fm.D_f[fm.matte]).all() and len(fm.cell) == len(fm.rim) == len(fm.sc)
    inner = fm.matte & (fm.dt > 3)
    assert np.median(np.abs(fm.D_f - front.D_f)[inner]) < 0.01

    monkeypatch.setattr(sf, "_FALLBACK_MIN", 10**9)
    whole = sharp_front(scene.sc, scene.alpha, scene.cam, scene.rgb, mould_long=MOULD)
    assert whole.stats["fallback_kind"] == "whole_scene" and len(whole.sc) == len(scene.sc)
    assert np.isfinite(whole.D_f[whole.matte]).all()


def test_cancel_raises_cancelled(scene):
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        sharp_front(scene.sc, scene.alpha, scene.cam, scene.rgb, mould_long=MOULD, cancel=tok)
    tok2 = CancelToken()
    seen: list[float] = []

    def progress(frac: float, msg: str) -> None:
        seen.append(frac)
        if frac >= 0.3:
            tok2.cancel()

    with pytest.raises(Cancelled):
        sharp_front(
            scene.sc, scene.alpha, scene.cam, scene.rgb, mould_long=MOULD, progress=progress, cancel=tok2
        )
    assert seen and max(seen) < 1.0


def test_progress_reaches_one(scene):
    seen: list[float] = []
    sharp_front(
        scene.sc, scene.alpha, scene.cam, scene.rgb, mould_long=MOULD, progress=lambda f, m: seen.append(f)
    )
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_bad_inputs(scene):
    with pytest.raises(InputError):
        sharp_front(scene.sc, np.zeros_like(scene.alpha), scene.cam, scene.rgb, mould_long=MOULD)
    with pytest.raises(InputError):
        sharp_front(scene.sc, scene.alpha, scene.cam, scene.rgb[:10], mould_long=MOULD)
    with pytest.raises(InputError):
        sharp_front(scene.sc, scene.alpha, scene.cam, scene.rgb, mould_long=0)
    empty = SplatCloud(
        means=np.zeros((0, 3), np.float32),
        scales=np.zeros((0, 3), np.float32),
        quats=np.zeros((0, 4), np.float32),
        colors=np.zeros((0, 3), np.float32),
        opacity=np.zeros(0, np.float32),
    )
    with pytest.raises(VRPhotoError):
        sharp_front(empty, scene.alpha, scene.cam, scene.rgb, mould_long=MOULD)


def test_depth_tol():
    assert depth_tol(1.0) == 0.02 and depth_tol(5.0) == pytest.approx(0.05)
    np.testing.assert_allclose(depth_tol(np.array([0.5, 2.0, 4.0])), [0.02, 0.02, 0.04])


# --------------------------------------------------------------------------- canonical frame
def _canon_for(body: sp.Body, cam: SharpCam, **kw):
    s = sp.sharp_like_scene(body, cam, **kw)
    fm = sharp_front(s.sc, s.alpha, s.cam, s.rgb, mould_long=MOULD)
    return canonical_frame(fm, s.sc, s.alpha), fm


@pytest.mark.parametrize("pitch", [10.0, 20.0])
def test_up_from_a_tilted_floor(cam, pitch):
    b = sp.make_body(pitch_deg=pitch)
    canon, _ = _canon_for(b, cam)
    assert canon.up_source == "floor" and canon.diag["floor"]["used"]
    assert _angle(canon.up_C, b.up_C) < 1.0
    assert canon.pitch_deg == pytest.approx(pitch, abs=1.0)
    assert canon.diag["warnings"] == []


def test_up_without_a_floor_is_the_cameras(cam):
    canon, _ = _canon_for(sp.make_body(), cam, floor=False)
    assert canon.up_source == "camera"
    np.testing.assert_allclose(canon.up_C, [0.0, -1.0, 0.0], atol=1e-12)
    assert canon.pitch_deg == pytest.approx(0.0, abs=1e-9)


def test_up_from_the_front_axis_when_pitched_without_a_floor(cam):
    b = sp.make_body(pitch_deg=15.0)
    canon, _ = _canon_for(b, cam, floor=False)
    assert canon.up_source == "pca"
    assert _angle(canon.up_C, b.up_C) < 2.0  # the raised arm does not tilt it sideways


def test_frame_axes_sizes_and_axis_prior(body, cam, scene, front):
    canon = canonical_frame(front, scene.sc, scene.alpha, thickness=0.11)
    truth = sp.synth_canon(body, cam)
    R = canon.R
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(canon.up_C, -R[:, 1], atol=1e-12)
    for k in range(3):
        assert _angle(R[:, k], truth.R[:, k]) < 2.0
    assert canon.height_s == pytest.approx(truth.height_s, rel=0.08)
    assert canon.width_s == pytest.approx(truth.width_s, rel=0.15)
    assert canon.T_prior == pytest.approx(0.11 * canon.height_s)
    assert canon.z_axis == pytest.approx(truth.z_axis, abs=0.05)
    np.testing.assert_allclose(canon.c, truth.c, atol=0.06)
    np.testing.assert_allclose(canon.to_S(canon.c), 0.0, atol=1e-12)
    P = front.sc.means[:50].astype(np.float64)
    np.testing.assert_allclose(canon.to_C(canon.to_S(P)), P, atol=1e-9)
    assert not canon.cut_top and not canon.cut_bottom


@pytest.mark.parametrize(("cam_height", "top", "bottom"), [(1.4, False, True), (0.3, True, False)])
def test_cut_off_subject_is_detected(cam, cam_height, top, bottom):
    b = sp.make_body(cam_height=cam_height)
    canon, _ = _canon_for(b, cam)
    truth = sp.synth_canon(b, cam)
    assert (canon.cut_top, canon.cut_bottom) == (top, bottom) == (truth.cut_top, truth.cut_bottom)
    if bottom:  # no feet, no floor under them
        assert canon.up_source != "floor"


def test_steep_pitch_warns(monkeypatch, scene, front):
    up = np.array([0.0, -math.cos(math.radians(35)), -math.sin(math.radians(35))])
    monkeypatch.setattr(sf, "_floor_up", lambda *a: (up, {"used": True}))
    canon = canonical_frame(front, scene.sc, scene.alpha)
    assert canon.pitch_deg == pytest.approx(35.0)
    assert canon.diag["warnings"] and "35" in canon.diag["warnings"][0]


def test_canonical_frame_bad_inputs(scene, front):
    with pytest.raises(InputError):
        canonical_frame(front, scene.sc, np.zeros_like(scene.alpha))
    with pytest.raises(InputError):
        canonical_frame(front, scene.sc, scene.alpha, thickness=-1)


# --------------------------------------------------------------------------- performance
@pytest.mark.slow
def test_front_cleanup_of_a_full_size_sharp_cloud_is_fast():
    """1.2 M Gaussians (SHARP's 768 grid, two layers) in well under 30 s."""
    G, W, H, f = 768, 1536, 1152, 1300.0
    cam = SharpCam(f=f, width=W, height=H, n_raw=2 * G * G, grid=G, layers=2)
    rng = np.random.default_rng(0)
    rows, cols = np.divmod(np.arange(G * G), G)
    u = (cols + rng.uniform(0.3, 0.7, G * G)) * W / G
    v = (rows + rng.uniform(0.3, 0.7, G * G)) * H / G
    subject = (np.abs(u - W / 2) < W / 6) & (np.abs(v - H / 2) < H / 3)
    z0 = np.where(subject, 2.0 + 0.05 * np.sin(u / 50.0), 4.0)
    z1 = np.where(subject, 4.0, 4.03)
    d = cam.rays(np.r_[u, u], np.r_[v, v])
    z = np.r_[z0, z1]
    n = len(z)
    sc = SplatCloud(
        means=(d * z[:, None]).astype(np.float32),
        scales=np.abs(rng.normal(0.002, 0.0005, (n, 3))).astype(np.float32) + 1e-4,
        quats=rng.normal(size=(n, 4)).astype(np.float32),
        colors=rng.random((n, 3)).astype(np.float32),
        opacity=np.r_[np.full(G * G, 0.95), np.full(G * G, 0.8)].astype(np.float32),
    )
    alpha = np.zeros((H, W), np.float32)
    alpha[H // 2 - H // 3 : H // 2 + H // 3, W // 2 - W // 6 : W // 2 + W // 6] = 1.0
    rgb = np.zeros((H, W, 3), np.uint8)
    t0 = time.perf_counter()
    fm = sharp_front(sc, alpha, cam, rgb, mould_long=768)
    dt = time.perf_counter() - t0
    assert dt < 30.0, dt
    assert fm.stats["sharp_grid_match"] and not fm.stats["fallback"]
    on_subject = np.r_[subject, np.zeros(G * G, bool)]
    kept = _kept(fm, n)
    assert kept[on_subject].mean() > 0.98
    assert kept[G * G :].sum() < 0.01 * subject.sum()  # the wall layer behind the subject
