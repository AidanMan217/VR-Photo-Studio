"""SHARP 360 registration utilities (geometry/register.py, design 5.2 test_register.py), on
small synthetic data and the analytic person of tests/geometry/synth_person.py."""

from __future__ import annotations

import json
import math
import time

import cv2
import numpy as np
import pytest
import synth_person as sp

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, SplatCloud
from vrphoto.geometry import register as rg
from vrphoto.geometry.splat import concat_splats


# --------------------------------------------------------------------------- helpers
def _rot(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * (K @ K)


def _angle(R: np.ndarray) -> float:
    """Rotation angle of ``R`` in degrees.  atan2 of the skew part and the trace stays exact near
    0 deg, where acos((trace - 1) / 2) has a ~1e-6 deg noise floor (it failed on CI once)."""
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return math.degrees(math.atan2(0.5 * float(np.linalg.norm(w)), 0.5 * (float(np.trace(R)) - 1.0)))


def body_surface(body: sp.Body, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(P, normals, rgb)``: about ``n`` points on the outer surface of the body (the union of
    its ellipsoids) in C, roughly area-uniform, with outward normals and side colours."""
    rng = np.random.default_rng(seed)
    r = body.radii
    p = 1.6075  # Knud Thomsen's ellipsoid area approximation
    area = (
        4
        * np.pi
        * (((r[:, 0] * r[:, 1]) ** p + (r[:, 0] * r[:, 2]) ** p + (r[:, 1] * r[:, 2]) ** p) / 3) ** (1 / p)
    )
    cnt = np.maximum(1, np.round(1.4 * n * area / area.sum()).astype(int))
    pts, part = [], []
    for k in range(len(r)):
        d = rng.normal(size=(cnt[k], 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts.append(body.centers[k] + (d * r[k]) @ body.rot[k].T)
        part.append(np.full(cnt[k], k))
    P, part = np.concatenate(pts), np.concatenate(part)
    inside = np.zeros(len(P), bool)
    for k in range(len(r)):  # drop points buried inside another part
        q = (P - body.centers[k]) @ body.rot[k] / r[k]
        inside |= ((q * q).sum(1) < 1.0 - 1e-9) & (part != k)
    P, part = P[~inside], part[~inside]
    if len(P) > n:
        sel = np.sort(rng.choice(len(P), n, replace=False))
        P, part = P[sel], part[sel]
    nrm = body.normal_at(P, part)
    return P, nrm, body.colour_of(nrm)


def _cloud(P: np.ndarray, rgb: np.ndarray, size: float, opacity: float = 0.9) -> SplatCloud:
    m = len(P)
    return SplatCloud(
        means=P.astype(np.float32),
        scales=np.full((m, 3), size, np.float32),
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (m, 1)),
        colors=(np.asarray(rgb, np.float32) / 255.0).astype(np.float32),
        opacity=np.full(m, opacity, np.float32),
    )


def trellis_like_cloud(
    body: sp.Body, canon, up: str, yaw_deg: float, s: float, *, n: int = 30_000, seed: int = 0
) -> tuple[SplatCloud, np.ndarray]:
    """A TRELLIS-like Gaussian cloud of the whole body in its own unit-cube frame T: the subject
    (S coordinates) turned by ``yaw_deg`` about its vertical, its up mapped to the cloud axis
    ``up`` and shrunk by ``s`` - i.e. ``P_T = s * A^T R_y(yaw)^T R^T (P_C - c) + offset`` with
    ``A = up_rotation(up)`` - plus ``P_C`` (the truth, row for row)."""
    P, _, rgb = body_surface(body, n, seed)
    M = rg.up_rotation(up).T @ rg.yaw_rotation(yaw_deg).T @ canon.R.T  # C -> T rotation
    PT = s * (P - canon.c) @ M.T
    PT -= 0.5 * (PT.min(0) + PT.max(0))  # centred in the cube, like TRELLIS ...
    PT += np.array([0.02, -0.01, 0.03])  # ... give or take
    return _cloud(PT, rgb, 0.004), P


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
def surface(body):
    return body_surface(body, 20_000, seed=3)


# --------------------------------------------------------------------------- Nelder-Mead / Umeyama
def test_nelder_mead_solves_rosenbrock():
    def rosen(x):
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    x, fx = rg.nelder_mead(rosen, [-1.2, 1.0], 0.5, max_iter=2000, tol=1e-12)
    assert np.allclose(x, [1.0, 1.0], atol=1e-4)
    assert fx < 1e-8
    # the default budget gets close too
    x, fx = rg.nelder_mead(rosen, [-1.2, 1.0], [0.5, 0.5])
    assert np.allclose(x, [1.0, 1.0], atol=1e-2) and fx < 1e-4


def test_nelder_mead_nonfinite_values_and_cancel():
    target = np.array([0.3, -2.0, 5.0, 1.0])

    def f(x):  # NaN outside a box: treated as +inf, the simplex stays inside
        return float(((x - target) ** 2).sum()) if np.all(np.abs(x) < 10) else float("nan")

    x, fx = rg.nelder_mead(f, np.zeros(4), 1.0, max_iter=2000, tol=1e-12)
    assert np.allclose(x, target, atol=1e-4) and fx < 1e-8
    x, _ = rg.nelder_mead(lambda p: (p[0] - 3.0) ** 2, [3.0], 0.0)  # zero step -> a default one
    assert abs(x[0] - 3.0) < 1e-3
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        rg.nelder_mead(f, np.zeros(4), 1.0, cancel=tok)
    with pytest.raises(InputError):
        rg.nelder_mead(f, np.zeros(4), [1.0, 2.0])


def test_umeyama_exact_and_noisy():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(300, 3))
    R = _rot([0.3, -1.0, 0.4], 37.0)
    s, t = 1.7, np.array([0.4, -2.0, 3.0])
    dst = s * src @ R.T + t
    s1, R1, t1 = rg.umeyama(src, dst)
    assert abs(s1 - s) < 1e-10 and np.allclose(R1, R, atol=1e-10) and np.allclose(t1, t, atol=1e-10)
    noisy = dst + rng.normal(0, 1e-3, dst.shape)
    s2, R2, t2 = rg.umeyama(src, noisy)
    assert abs(s2 - s) < 1e-3 and _angle(R2 @ R.T) < 0.05 and np.abs(t2 - t).max() < 2e-3
    assert abs(np.linalg.det(R2) - 1.0) < 1e-12
    s3, R3, _ = rg.umeyama(src, dst, with_scale=False)
    assert s3 == 1.0 and _angle(R3 @ R.T) < 1e-6


def test_umeyama_weights_ignore_outliers():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(200, 3))
    R = _rot([1.0, 0.2, 0.0], -20.0)
    dst = 0.8 * src @ R.T + np.array([1.0, 2.0, 3.0])
    bad = rng.random(200) < 0.3
    dst[bad] += rng.normal(0, 2.0, (int(bad.sum()), 3))
    w = np.where(bad, 0.0, 1.0)
    s, R1, t = rg.umeyama(src, dst, weights=w)
    assert abs(s - 0.8) < 1e-9 and _angle(R1 @ R.T) < 1e-6
    # integer weights == repeated points
    w2 = rng.integers(1, 4, 200).astype(float)
    rep = np.repeat(np.arange(200), w2.astype(int))
    a = rg.umeyama(src, dst, weights=w2)
    b = rg.umeyama(src[rep], dst[rep])
    assert (
        abs(a[0] - b[0]) < 1e-9 and np.allclose(a[1], b[1], atol=1e-9) and np.allclose(a[2], b[2], atol=1e-9)
    )


def test_umeyama_never_reflects_and_validates():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(50, 3))
    s, R, _ = rg.umeyama(src, src * np.array([-1.0, 1.0, 1.0]))  # a mirror image
    assert abs(np.linalg.det(R) - 1.0) < 1e-12 and s > 0
    with pytest.raises(InputError):
        rg.umeyama(src[:2], src[:2])
    with pytest.raises(InputError):
        rg.umeyama(src, src[:10])
    s, R, t = rg.umeyama(np.ones((5, 3)), np.zeros((5, 3)))  # coincident points
    assert s == 1.0 and np.allclose(R, np.eye(3)) and np.allclose(t, -1.0)


# --------------------------------------------------------------------------- point z-buffer
def _zbuffer_brute(u, v, z, H, W, radius):
    zb = np.full((H, W), np.inf)
    idx = np.full((H, W), -1, np.int64)
    rad = np.broadcast_to(np.asarray(0.0 if radius is None else radius, float), (len(u),))
    for i in range(len(u)):
        if not (np.isfinite(u[i]) and np.isfinite(v[i]) and np.isfinite(z[i])):
            continue
        r = min(float(rad[i]), 16.0)
        x0, y0, k = math.floor(u[i]), math.floor(v[i]), math.floor(r)
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                x, y = x0 + dx, y0 + dy
                if dx * dx + dy * dy > r * r + 1e-9 or not (0 <= x < W and 0 <= y < H):
                    continue
                if z[i] < zb[y, x] or (z[i] == zb[y, x] and i < idx[y, x]):
                    zb[y, x], idx[y, x] = z[i], i
    return zb, idx


@pytest.mark.parametrize("radius", [None, 0.0, 1.0, 1.5, 2.2, "per_point"])
def test_point_zbuffer_matches_brute_force(radius):
    rng = np.random.default_rng(5)
    H, W, n = 17, 23, 260
    u = rng.uniform(-3, W + 3, n)
    v = rng.uniform(-3, H + 3, n)
    z = rng.choice(np.linspace(1.0, 3.0, 40), n)  # repeated depths: exercises the tie rule
    u[::37] = np.nan
    z[5::41] = np.inf
    rad = rng.uniform(0, 2.6, n) if radius == "per_point" else radius
    zb, idx = rg.point_zbuffer(u, v, z, (H, W), radius=rad)
    zr, ir = _zbuffer_brute(u, v, z, H, W, rad)
    assert zb.shape == (H, W) and zb.dtype == np.float64 and idx.dtype == np.int64
    assert np.array_equal(zb, zr)
    assert np.array_equal(idx, ir)


def test_point_zbuffer_edge_cases():
    zb, idx = rg.point_zbuffer([], [], [], (3, 4))
    assert np.isinf(zb).all() and (idx == -1).all()
    zb, idx = rg.point_zbuffer([1.5, 1.2], [0.5, 0.9], [2.0, 2.0], (2, 3))
    assert zb[0, 1] == 2.0 and idx[0, 1] == 0  # exact tie: lowest index
    with pytest.raises(InputError):
        rg.point_zbuffer([1.0], [1.0, 2.0], [1.0], (3, 3))
    with pytest.raises(InputError):
        rg.point_zbuffer([1.0], [1.0], [1.0], (3, 3), radius=[1.0, 2.0])


# --------------------------------------------------------------------------- silhouettes
def test_silhouette_iou():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), np.float32)
    a[2:6, 2:6] = True
    b[4:8, 2:6] = 0.7
    assert rg.silhouette_iou(a, b) == pytest.approx(8 / 24)
    assert rg.silhouette_iou(a, a) == 1.0
    assert rg.silhouette_iou(np.zeros((4, 4)), np.zeros((4, 4))) == 0.0
    with pytest.raises(InputError):
        rg.silhouette_iou(a, a[:5])


def test_contour_points_order_and_orientation():
    m = np.zeros((40, 60), np.uint8)
    m[10:30, 15:45] = 1
    pts = rg.contour_points(m, 64)
    assert pts.shape == (64, 2) and pts.dtype == np.float64
    assert pts[0, 1] == 10 and abs(pts[0, 0] - 30) <= 1  # topmost, middle of the flat top
    assert pts[1, 0] > pts[0, 0] and pts[1, 1] == 10  # clockwise on screen: along the top to +x
    x, y = pts[:, 0], pts[:, 1]
    assert (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() > 0
    step = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    perimeter = 2 * (29 + 19)  # through the outer pixel centres
    assert np.allclose(step, perimeter / 64, atol=0.6)  # equal arc length (corners cut)
    # a disc plus a smaller blob: the largest contour, starting at the top of the disc
    yy, xx = np.mgrid[0:80, 0:80]
    m2 = ((xx - 40) ** 2 + (yy - 45) ** 2 <= 20**2).astype(np.float32)
    m2[2:6, 2:6] = 1.0
    p2 = rg.contour_points(m2, 32)
    assert p2[0, 1] == 25 and abs(p2[0, 0] - 40) <= 1
    assert np.allclose(np.hypot(p2[:, 0] - 40, p2[:, 1] - 45), 20, atol=1.0)
    with pytest.raises(InputError):
        rg.contour_points(np.zeros((5, 5)), 16)
    with pytest.raises(InputError):
        rg.contour_points(m, 2)


@pytest.fixture(scope="module")
def views(body, canon):
    """Orthographic silhouettes of the body (m = 1.2 cm/px, 160 px): front and back."""
    a0 = sp.ortho_view(body, canon, 0.0, m=0.012, N=160)[1]
    a180 = sp.ortho_view(body, canon, 180.0, m=0.012, N=160)[1]
    return a0 >= 0.5, a180 >= 0.5


def _map(params, pts):
    M = rg._similarity_affine(params)
    return pts @ M[:, :2].T + M[:, 2]


def test_chamfer_similarity_recovers_scale_and_shift(views):
    src = views[0]
    ys, xs = np.nonzero(src)
    true = {"s": 1.08, "tx": 6.0, "ty": -4.0, "roll_deg": 0.0, "cx": float(xs.mean()), "cy": float(ys.mean())}
    dst = rg.warp_similarity(src, true, src.shape, interpolation=cv2.INTER_LINEAR)
    assert dst.dtype == bool and rg.silhouette_iou(src, dst) < 0.8
    params, cost = rg.chamfer_similarity_fit(src, dst)
    assert set(params) == {"s", "tx", "ty", "roll_deg", "cx", "cy"}
    assert abs(params["s"] - 1.08) < 0.01 and params["roll_deg"] == 0.0
    corners = np.array(
        [[xs.min(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()], [xs.max(), ys.min()]], float
    )
    assert np.abs(_map(params, corners) - _map(true, corners)).max() < 1.0
    assert cost < 1.0
    back = rg.warp_similarity(src, params, dst.shape)
    assert rg.silhouette_iou(back, dst) > 0.97
    # a start near the answer, with another pivot, ends in the same place
    p2, _ = rg.chamfer_similarity_fit(src, dst, init={**true, "cx": 0.0, "cy": 0.0, "tx": 0.0, "ty": 0.0})
    assert np.abs(_map(p2, corners) - _map(true, corners)).max() < 1.5


def test_chamfer_similarity_recovers_roll_within_bound(views):
    src = views[1]
    ys, xs = np.nonzero(src)
    true = {"s": 0.95, "tx": -3.0, "ty": 5.0, "roll_deg": 3.0, "cx": float(xs.mean()), "cy": float(ys.mean())}
    dst = rg.warp_similarity(src.astype(np.float32), true, src.shape) >= 0.5
    params, _ = rg.chamfer_similarity_fit(src, dst, allow_roll=True, max_roll_deg=5.0)
    assert abs(params["roll_deg"] - 3.0) < 0.5 and abs(params["s"] - 0.95) < 0.01
    corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.max()]], float)
    assert np.abs(_map(params, corners) - _map(true, corners)).max() < 1.5
    big = dict(true, roll_deg=12.0)
    dst2 = rg.warp_similarity(src, big, src.shape)
    p2, _ = rg.chamfer_similarity_fit(src, dst2, allow_roll=True, max_roll_deg=5.0)
    assert abs(p2["roll_deg"]) <= 5.0 + 1e-9
    with pytest.raises(InputError):
        rg.chamfer_similarity_fit(np.zeros_like(src), dst)


def _keystone(mask: np.ndarray, pitch_deg: float, aspect: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(warped mask, H)``: the mask as a camera tilted by ``pitch_deg`` would see it (``K R
    K^-1`` about the picture centre, f = the picture size) with ``aspect`` on x."""
    h, w = mask.shape
    f = float(max(h, w))
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
    a = math.radians(pitch_deg)
    R = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
    H = np.diag([aspect, 1.0, 1.0]) @ K @ R @ np.linalg.inv(K)
    H /= H[2, 2]
    out = cv2.warpPerspective(mask.astype(np.float32), H, (w, h), flags=cv2.INTER_LINEAR) >= 0.5
    return out, H


def test_perspective_fit_takes_out_a_camera_tilt(views):
    src = np.pad(views[1], 20)  # the back silhouette with room around it
    dst, _ = _keystone(src, 12.0, aspect=0.97)
    assert rg.silhouette_iou(src, dst) < 0.93
    H, cost, cost0 = rg.chamfer_perspective_fit(src, dst)
    assert H.shape == (3, 3) and cost < 0.5 * cost0
    fitted = cv2.warpPerspective(src.astype(np.float32), H, dst.shape[::-1]) >= 0.5
    assert rg.silhouette_iou(fitted, dst) > 0.97
    assert rg.outline_gap(fitted, dst) < 1.5 and rg.outline_gap(src, dst) > 3.0
    # the same outline: (nearly) the identity, never worse than it
    H0, c, c0 = rg.chamfer_perspective_fit(src, src)
    assert c <= c0 and np.abs(H0 - np.eye(3)).max() < 0.02
    with pytest.raises(InputError):
        rg.chamfer_perspective_fit(src, dst[:-1])
    with pytest.raises(InputError):
        rg.chamfer_perspective_fit(np.zeros_like(src), dst)


def test_perspective_fit_does_not_absorb_a_different_pose():
    """A camera turn cannot lower an arm: the arm stays where it is and the outline gap shows
    it (a general homography, shear included, would bend the body toward the arm instead)."""

    def figure(arm_deg: float) -> np.ndarray:
        m = np.zeros((240, 240), np.uint8)
        cv2.ellipse(m, (120, 140), (28, 70), 0, 0, 360, 1, -1)
        cv2.circle(m, (120, 55), 16, 1, -1)
        a = math.radians(arm_deg)
        cv2.line(m, (140, 95), (int(140 + 80 * math.cos(a)), int(95 - 80 * math.sin(a))), 1, 12)
        return m.astype(bool)

    src, dst = figure(60.0), figure(20.0)
    H, _, _ = rg.chamfer_perspective_fit(src, dst)
    fitted = cv2.warpPerspective(src.astype(np.float32), H, (240, 240)) >= 0.5
    assert rg.outline_gap(fitted, dst) > 24.0  # the arm stays put (a general homography: 13 px)
    body = np.zeros((240, 240), bool)
    body[:, :138] = True  # left of the arm: the head and body still lie on each other
    assert rg.silhouette_iou(fitted & body, dst & body) > 0.9


def test_outline_gap():
    a = np.zeros((50, 50), bool)
    a[10:40, 10:40] = True
    b = np.roll(a, 3, axis=1)
    assert rg.outline_gap(a, a) == 0.0
    assert 2.5 < rg.outline_gap(a, b, 100.0) <= 3.0
    assert rg.outline_gap(a, np.zeros_like(a)) == math.inf
    with pytest.raises(InputError):
        rg.outline_gap(a, a[:-1])


def test_warp_similarity_types():
    img = np.arange(6 * 8 * 6, dtype=np.float64).reshape(6, 8, 6)
    ident = {"s": 1.0, "tx": 0.0, "ty": 0.0, "roll_deg": 0.0, "cx": 3.0, "cy": 2.0}
    out = rg.warp_similarity(img, ident, (6, 8))
    assert out.shape == img.shape and out.dtype == np.float64 and np.allclose(out, img, atol=1e-3)
    shift = dict(ident, tx=2.0)
    rgb = np.full((6, 8, 3), 200, np.uint8)
    o2 = rg.warp_similarity(rgb, shift, (7, 9), interpolation=cv2.INTER_NEAREST, border_value=(1, 2, 3))
    assert o2.shape == (7, 9, 3) and o2.dtype == np.uint8
    assert tuple(o2[0, 0]) == (1, 2, 3) and tuple(o2[0, 4]) == (200, 200, 200)
    m = np.ones((6, 8), bool)
    assert not rg.warp_similarity(m, shift, (6, 8))[0, 0]  # bool masks: border False ...
    assert rg.warp_similarity(m, shift, (6, 8), border_value=1)[0, 0]  # ... or True
    with pytest.raises(InputError):
        rg.warp_similarity(rgb, dict(ident, s=-1.0), (6, 8))


# --------------------------------------------------------------------------- thin-plate spline
def test_tps_identity_and_affine():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 200, (40, 2))
    tps = rg.tps_fit(pts, pts)
    assert tps.max_displacement((200, 200)) < 1e-6
    A = np.array([[1.1, 0.2], [-0.1, 0.9]])
    b = np.array([5.0, -7.0])
    src = pts @ A.T + b  # output pixel q samples the source at A q + b
    tps = rg.tps_fit(src, pts, lam=0.0)
    q = rng.uniform(-50, 250, (500, 2))
    assert np.allclose(tps(q), q @ A.T + b, atol=1e-6)
    # a smooth known warp is interpolated exactly at the control points (lam = 0)
    warp = pts + np.stack([3 * np.sin(pts[:, 1] / 30), 2 * np.cos(pts[:, 0] / 40)], 1)
    tps = rg.tps_fit(warp, pts, lam=0.0)
    assert np.allclose(tps(pts), warp, atol=1e-6)
    assert 2.0 < tps.max_displacement((200, 200)) < 6.0
    with pytest.raises(InputError):
        rg.tps_fit(pts[:2], pts[:2])


def test_tps_warp_backward_map_and_lattice():
    rng = np.random.default_rng(4)
    dst = rng.uniform(0, 700, (30, 2))
    src = dst + rng.normal(0, 4, dst.shape)
    tps = rg.tps_fit(src, dst)
    ramp = np.tile(np.arange(800, dtype=np.float32), (800, 1))  # value = x coordinate
    out = rg.tps_warp(ramp, tps, (700, 720))  # > 256^2 pixels: the 4 px lattice path
    q = np.array([[100.0, 50.0], [355.0, 401.0], [690.0, 610.0], [3.0, 699.0]])
    expect = tps(q)[:, 0]
    got = out[q[:, 1].astype(int), q[:, 0].astype(int)]
    assert np.abs(got - expect).max() < 0.02
    small = rg.tps_warp(ramp[:300, :300], tps, (200, 220))  # exact path
    q2 = np.array([[10.0, 20.0], [150.0, 180.0]])
    assert np.abs(small[q2[:, 1].astype(int), q2[:, 0].astype(int)] - tps(q2)[:, 0]).max() < 1e-3
    m = rg.tps_warp(np.ones((50, 50), bool), tps, (60, 60))
    assert m.dtype == bool


def test_tps_lock_reaches_iou_098(body, canon):
    # 400 px views (the generated views are 512-1024 px): at 160 px even the exact inverse
    # warp only reaches IoU ~0.97, because resampling erodes the thin raised arm
    N = 400
    a0 = sp.ortho_view(body, canon, 0.0, m=0.005, N=N)[1]
    a180 = sp.ortho_view(body, canon, 180.0, m=0.005, N=N)[1]
    target = a0[:, ::-1] >= 0.5  # the back lock: the back view must equal the mirrored front outline
    assert rg.silhouette_iou(target, a180) > 0.99  # orthographic: exact up to sampling
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    # the "generated" back (soft alpha): same pose, bent and 2 % larger
    mx = xx + 2.5 * np.sin(yy / 57.5) - 0.02 * (xx - N / 2)
    my = (yy - N / 2) / 1.02 + N / 2 + 1.25 * np.cos(xx / 77.5)
    deformed = cv2.remap(a180, mx, my, cv2.INTER_LINEAR)
    assert rg.silhouette_iou(deformed, target) < 0.9
    tps = rg.tps_fit(rg.contour_points(deformed, 64), rg.contour_points(target, 64))
    locked = rg.tps_warp(deformed, tps, target.shape)
    assert locked.dtype == np.float32 and rg.silhouette_iou(locked, target) >= 0.98
    # the lock moves the subject itself by well under 6 % of its size
    assert tps.max_displacement(target.shape, mask=target) < 0.06 * N
    assert tps.max_displacement(target.shape) >= tps.max_displacement(target.shape, mask=target)


# --------------------------------------------------------------------------- 3D registration
def _perturb(P, centre, deg, scale, shift):
    Rp = _rot([0.3, 1.0, 0.2], deg)
    return scale * (P - centre) @ Rp.T + centre + shift, Rp


@pytest.mark.parametrize(
    "deg, scale, shift",
    [(5.0, 1.05, (0.018, -0.0144, 0.0192)), (-5.0, 0.95, (-0.018, 0.0144, -0.0192))],
)
def test_projective_icp_converges(front, canon, surface, deg, scale, shift):
    P = surface[0]
    Pp, Rp = _perturb(P, canon.c, deg, scale, np.array(shift))  # 5 deg, 5 %, 3 cm
    s, R, t, diag = rg.projective_icp(Pp, front)
    Q = s * Pp @ R.T + t
    # the documented result of the default 8 iterations (< 0.2 % scale; measured 0.10 / 0.01 %,
    # 0.016 / 0.002 deg, 0.4 mm): 3 iterations would already fail these
    assert _angle(R @ Rp) < 0.05
    assert abs(s * scale - 1.0) < 0.002
    assert np.median(np.linalg.norm(Q - P, axis=1)) < 0.001
    assert not diag["rejected"] and diag["resid_m"] < 0.001
    json.dumps(diag)


def test_projective_icp_point_metric_and_rejection(front, canon, surface):
    P = surface[0]
    Pp, _ = _perturb(P, canon.c, 3.0, 1.02, np.array([0.01, 0.0, 0.01]))
    err0 = np.median(np.linalg.norm(Pp - P, axis=1))
    s, R, t, diag = rg.projective_icp(Pp, front, metric="point")
    assert diag["metric"] == "point"
    assert np.median(np.linalg.norm(s * Pp @ R.T + t - P, axis=1)) < err0
    far, _ = _perturb(P, canon.c, 25.0, 1.0, np.zeros(3))
    s, R, t, diag = rg.projective_icp(far, front, max_rot_deg=0.5)
    assert diag["rejected"] and diag["iters"] == 1 and s == 1.0 and np.allclose(R, np.eye(3))
    with pytest.raises(InputError):
        rg.projective_icp(P[:, :2], front)
    with pytest.raises(InputError):
        rg.projective_icp(P, front, metric="nope")


def test_up_and_yaw_rotations():
    for name, axis in rg.UP_AXES.items():
        A = rg.up_rotation(name)
        assert np.allclose(A @ np.array(axis), [0.0, -1.0, 0.0]) and abs(np.linalg.det(A) - 1) < 1e-12
    Ry = rg.yaw_rotation(90.0)
    assert np.allclose(Ry @ [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]) and np.allclose(Ry @ [0, 1.0, 0], [0, 1.0, 0])
    with pytest.raises(InputError):
        rg.up_rotation("up")


def test_register_cloud_recovers_up_yaw_scale(body, front, canon):
    sc, P_true = trellis_like_cloud(body, canon, "-z", 137.0, 0.37)
    steps = []
    out, diag = rg.register_cloud_to_front(sc, front, canon, progress=lambda f, m: steps.append(f))
    assert diag["ok"] and diag["reason"] == "" and out is not None
    assert diag["up"] == "-z"
    assert abs((diag["yaw_deg"] - 137.0 + 180.0) % 360.0 - 180.0) < 0.5
    assert abs(1.0 / diag["scale"] - 0.37) / 0.37 < 0.005
    assert diag["iou"] >= 0.9 and diag["resid_m"] <= 0.005 and diag["tilt_deg"] < 0.5
    assert diag["icp"]["kept"] and [c["up"] for c in diag["candidates"]][0] == "-z"
    assert len(out) == len(sc)  # the whole cloud, every Gaussian
    err = np.linalg.norm(out.means.astype(np.float64) - P_true, axis=1)
    assert np.median(err) < 0.003 and np.percentile(err, 95) < 0.01
    assert np.allclose(out.scales, 0.004 * diag["scale"], rtol=1e-4)
    tr = diag["transform"]
    assert np.allclose(
        tr["s"] * sc.means.astype(np.float64) @ np.array(tr["R"]).T + tr["t"], out.means, atol=1e-4
    )
    assert steps and steps[-1] == pytest.approx(1.0) and all(0.0 <= f <= 1.0 for f in steps)
    json.dumps(diag)


def test_register_cloud_other_convention(body, front, canon):
    sc, P_true = trellis_like_cloud(body, canon, "+x", 250.0, 1.4, n=12_000, seed=1)
    out, diag = rg.register_cloud_to_front(sc, front, canon)
    assert diag["ok"] and diag["up"] == "+x"
    assert abs((diag["yaw_deg"] - 250.0 + 180.0) % 360.0 - 180.0) < 1.5
    assert abs(1.0 / diag["scale"] - 1.4) / 1.4 < 0.01
    assert np.median(np.linalg.norm(out.means - P_true, axis=1)) < 0.01


def test_register_cloud_tolerates_an_imperfect_shape(body, front, canon):
    # TRELLIS's body is not SHARP's: 15 % deeper, 1 cm noise, 5 % floaters, plus faint junk
    rng = np.random.default_rng(11)
    P, _, rgb = body_surface(body, 20_000, seed=4)
    PS = canon.to_S(P)
    PS[:, 2] *= 1.15
    Pd = canon.to_C(PS) + rng.normal(0.0, 0.01, P.shape)
    M = rg.up_rotation("+z").T @ rg.yaw_rotation(300.0).T @ canon.R.T
    PT = np.vstack([0.5 * (Pd - canon.c) @ M.T, rng.uniform(-0.5, 0.5, (1000, 3))])
    sc = _cloud(PT, np.vstack([rgb, np.full((1000, 3), 90)]), 0.004)
    junk = _cloud(rng.uniform(-0.5, 0.5, (6000, 3)), np.full((6000, 3), 90), 0.004, opacity=0.1)
    sc = concat_splats([sc, junk])
    out, diag = rg.register_cloud_to_front(sc, front, canon)
    assert diag["ok"] and diag["up"] == "+z" and diag["n_opaque"] == len(PT)
    assert abs((diag["yaw_deg"] - 300.0 + 180.0) % 360.0 - 180.0) < 2.0
    assert abs(1.0 / diag["scale"] - 0.5) / 0.5 < 0.03
    # its front sits on SHARP's front, so the deeper body is ~1.6 cm nearer than Pd
    d = out.means[: len(P)] - Pd
    assert np.median(np.linalg.norm(d[:, :2], axis=1)) < 0.015 and np.median(np.linalg.norm(d, axis=1)) < 0.03


def test_register_cloud_gates_reject_a_box(front, canon):
    rng = np.random.default_rng(7)
    n = 12_000
    face = rng.integers(0, 6, n)
    P = rng.uniform(-0.5, 0.5, (n, 3)) * np.array([0.8, 1.0, 0.6])
    ext = np.array([0.4, 0.5, 0.3])
    P[np.arange(n), face // 2] = np.where(face % 2 == 0, -1.0, 1.0) * ext[face // 2]  # on the faces
    out, diag = rg.register_cloud_to_front(_cloud(P, np.full((n, 3), 128), 0.01), front, canon)
    assert out is None and not diag["ok"]
    assert diag["iou"] < 0.7 and "outline" in diag["reason"]
    json.dumps(diag)


def test_register_cloud_empty_and_cancel(body, front, canon):
    sc, _ = trellis_like_cloud(body, canon, "+y", 10.0, 0.5, n=3000)
    faint = SplatCloud(sc.means, sc.scales, sc.quats, sc.colors, np.full(len(sc), 0.1, np.float32))
    out, diag = rg.register_cloud_to_front(faint, front, canon)
    assert out is None and not diag["ok"] and "empty" in diag["reason"]
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        rg.register_cloud_to_front(sc, front, canon, cancel=tok)
    bad = SplatCloud(sc.means[:, :2], sc.scales, sc.quats, sc.colors, sc.opacity)
    with pytest.raises(InputError):
        rg.register_cloud_to_front(bad, front, canon)


@pytest.mark.slow
def test_register_cloud_performance_200k(body, front, canon):
    sc, P_true = trellis_like_cloud(body, canon, "-z", 137.0, 0.37, n=200_000, seed=2)
    assert len(sc) > 190_000
    t0 = time.perf_counter()
    out, diag = rg.register_cloud_to_front(sc, front, canon)
    dt = time.perf_counter() - t0
    assert diag["ok"] and dt < 15.0
    assert np.median(np.linalg.norm(out.means - P_true, axis=1)) < 0.005


# --------------------------------------------------------------------------- colour
def _gain_offset_pairs(seed: int, noise: float, outliers: float = 0.2, n: int = 2000):
    rng = np.random.default_rng(seed)
    lin = rng.uniform(0.06, 0.8, (n, 3))  # stays inside [0, 1] after the gain / offset
    src = rg._linear_to_srgb(lin)
    dst = rg._linear_to_srgb(1.2 * lin - 0.05 + rng.normal(0, noise, (n, 3)))
    bad = rng.random(n) < outliers
    dst[bad] = rng.random((int(bad.sum()), 3))
    return src, dst


def test_color_affine_recovers_gain_offset_with_outliers():
    for seed in range(3):
        src, dst = _gain_offset_pairs(seed, noise=0.0)
        A, b, ok = rg.color_affine_fit(src, dst)
        assert ok
        assert np.abs(np.diag(A) - 1.2).max() < 0.012  # 1 %
        assert np.abs(A - np.diag(np.diag(A))).max() < 0.005
        assert np.abs(b + 0.05).max() < 0.0005  # 1 %
    src, dst = _gain_offset_pairs(9, noise=0.003)  # with colour noise
    A, b, ok = rg.color_affine_fit(src, dst)
    assert ok and np.abs(np.diag(A) - 1.2).max() < 0.012 and np.abs(b + 0.05).max() < 0.002
    pred = rg.apply_color_affine(src, A, b)
    assert pred.dtype == src.dtype and pred.shape == src.shape
    inl = np.abs(pred - dst).max(axis=1) < 0.05
    assert inl.mean() > 0.75


def test_color_affine_clamps_and_identity():
    rng = np.random.default_rng(1)
    lin = rng.uniform(0.0, 0.3, (500, 3))
    src = rg._linear_to_srgb(lin)
    A, b, ok = rg.color_affine_fit(src, rg._linear_to_srgb(3.0 * lin + 0.2))
    assert ok
    assert np.linalg.norm(A - np.eye(3)) <= 0.35 + 1e-9
    assert np.all(np.diag(A) >= 0.6 - 1e-9) and np.all(np.diag(A) <= 1.6 + 1e-9)
    assert np.all(np.abs(b) <= 0.12 + 1e-12)
    A, b, ok = rg.color_affine_fit(src, src)  # nothing to correct: does not beat identity
    assert not ok and np.array_equal(A, np.eye(3)) and np.array_equal(b, np.zeros(3))
    A, b, ok = rg.color_affine_fit(src[:5], src[:5])
    assert not ok
    with pytest.raises(InputError):
        rg.color_affine_fit(src, src[:10])


def test_apply_color_affine_in_blocks(monkeypatch):
    rng = np.random.default_rng(6)
    img = rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
    A, b = np.diag([1.1, 0.9, 1.05]), np.array([0.01, -0.02, 0.0])
    whole = rg.apply_color_affine(img, A, b)
    monkeypatch.setattr(rg, "_COLOR_CHUNK", 7)  # blocks that split rows
    assert np.array_equal(rg.apply_color_affine(img, A, b), whole)
    f = img.reshape(-1, 3) / 255.0
    assert np.allclose(rg.apply_color_affine(f, A, b), whole.reshape(-1, 3) / 255.0, atol=0.5 / 255 + 1e-6)


def test_apply_color_affine_dtypes():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    same = rg.apply_color_affine(img, np.eye(3), np.zeros(3))
    assert same.dtype == np.uint8 and same.shape == img.shape and np.abs(same.astype(int) - img).max() <= 1
    brighter = rg.apply_color_affine(img, 1.2 * np.eye(3), np.zeros(3))
    assert np.all(brighter.astype(int) >= img.astype(int) - 1) and brighter.mean() > img.mean()
    f32 = (img.reshape(-1, 3) / 255.0).astype(np.float32)
    out = rg.apply_color_affine(f32, np.eye(3), np.full(3, 0.05))
    assert out.dtype == np.float32 and out.shape == f32.shape and np.all(out >= f32 - 1e-6)
    assert out.max() <= 1.0
    with pytest.raises(InputError):
        rg.apply_color_affine(img[..., :2], np.eye(3), np.zeros(3))


def _lab_stats(img, mask):
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)[mask]
    return lab.mean(0), lab.std(0)


def test_reinhard_moves_mean_and_std_toward_reference():
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal([90, 110, 160], 12, (60, 80, 3)), 0, 255).astype(np.uint8)
    ref = np.clip(rng.normal([170, 120, 80], 25, (50, 70, 3)), 0, 255).astype(np.uint8)
    m = np.zeros((60, 80), bool)
    m[10:50, 10:70] = True
    rm = np.ones((50, 70), bool)
    mu_s, sd_s = _lab_stats(img, m)
    mu_r, sd_r = _lab_stats(ref, rm)
    full = rg.reinhard_lab(img, m, ref, rm, strength=1.0)
    mu_f, sd_f = _lab_stats(full, m)
    assert full.dtype == np.uint8 and full.shape == img.shape
    assert np.abs(mu_f - mu_r).max() < 1.5 and np.abs(sd_f - sd_r).max() < 1.5
    half = rg.reinhard_lab(img, m, ref, rm)  # strength 0.5
    mu_h, _ = _lab_stats(half, m)
    assert np.all(np.abs(mu_h - (mu_s + mu_r) / 2) < 1.5)
    assert np.abs(full[0, 0].astype(int) - img[0, 0]).max() > 5  # applied outside the mask too
    none = rg.reinhard_lab(img, m, ref, rm, strength=0.0)
    assert np.abs(none.astype(int) - img).max() <= 1
    tiny = np.zeros_like(m)
    tiny[0, :3] = True
    assert np.array_equal(rg.reinhard_lab(img, tiny, ref, rm), img)
    with pytest.raises(InputError):
        rg.reinhard_lab(img, m, ref, rm, strength=1.5)
    with pytest.raises(InputError):
        rg.reinhard_lab(img, m[:10], ref, rm)
