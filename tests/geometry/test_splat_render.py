"""CPU splat renderer and turntable preview (geometry/splat_render.py, SHARP 360 design 5.2):
analytic single Gaussians, depth order, orthographic vs telephoto, band independence, a
brute-force per-pixel reference, edge cases, and a turntable of the synthetic person
(tests/geometry/synth_person.py)."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest
import synth_person as sp

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, SplatCloud
from vrphoto.geometry import splat_render as sr
from vrphoto.geometry.splat import _sh_basis, rotmat_to_quat, splat_covariances
from vrphoto.geometry.splat_render import SplatCamera, orbit_camera, render_splats, turntable_sheet

GLTF = np.array([1.0, -1.0, -1.0])  # geometry.camera.cv_to_gltf


def cloud(means, scales, colors, opacity, quats=None, sh_rest=None) -> SplatCloud:
    means = np.asarray(means, np.float32).reshape(-1, 3)
    n = len(means)
    scales = np.broadcast_to(np.asarray(scales, np.float32), (n, 3)).copy()
    if quats is None:
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    return SplatCloud(
        means=means,
        scales=scales,
        quats=np.asarray(quats, np.float32).reshape(n, 4),
        colors=np.broadcast_to(np.asarray(colors, np.float32), (n, 3)).copy(),
        opacity=np.broadcast_to(np.asarray(opacity, np.float32), (n,)).copy(),
        sh_rest=sh_rest,
    )


def random_quats(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def random_cloud(rng: np.random.Generator, n: int, *, center=(0.0, 0.0, 3.0), spread=0.6, s=(0.01, 0.12)):
    c = np.asarray(center) + rng.uniform(-spread, spread, size=(n, 3))
    return cloud(
        c,
        np.exp(rng.uniform(np.log(s[0]), np.log(s[1]), size=(n, 3))),
        rng.random((n, 3)),
        rng.uniform(0.15, 1.0, n),
        quats=random_quats(rng, n),
    )


def brute_force(sc: SplatCloud, cam: SplatCamera, *, max_r: int = 64) -> dict[str, np.ndarray]:
    """Per-pixel reference: the 3DGS rasteriser written out literally (float64, loops)."""
    H, W = cam.height, cam.width
    cov = splat_covariances(sc)
    gs = []
    for i in range(len(sc)):
        mu = sc.means[i].astype(np.float64)
        t = cam.R @ (mu - cam.center)
        o = float(sc.opacity[i])
        if t[2] <= sr.Z_NEAR or o < sr.ALPHA_MIN:
            continue
        if cam.kind == "persp":
            u, v = cam.fx * t[0] / t[2] + cam.cx, cam.fy * t[1] / t[2] + cam.cy
            lx0, lx1 = -cam.cx / cam.fx, (W - cam.cx) / cam.fx
            ly0, ly1 = -cam.cy / cam.fy, (H - cam.cy) / cam.fy
            xz = np.clip(t[0] / t[2], lx0 - 0.15 * (lx1 - lx0), lx1 + 0.15 * (lx1 - lx0))
            yz = np.clip(t[1] / t[2], ly0 - 0.15 * (ly1 - ly0), ly1 + 0.15 * (ly1 - ly0))
            J = np.array([[cam.fx / t[2], 0, -cam.fx * xz / t[2]], [0, cam.fy / t[2], -cam.fy * yz / t[2]]])
        else:
            u, v = t[0] / cam.m + cam.cx, t[1] / cam.m + cam.cy
            J = np.array([[1 / cam.m, 0, 0], [0, 1 / cam.m, 0]])
        S2 = J @ cam.R @ cov[i] @ cam.R.T @ J.T + 0.3 * np.eye(2)
        r = min(math.ceil(3 * math.sqrt(np.linalg.eigvalsh(S2).max())), max_r)
        gs.append((t[2], u, v, np.linalg.inv(S2), r, o, sc.colors[i].astype(np.float64)))
    gs.sort(key=lambda g: g[0])
    rgb = np.zeros((H, W, 3))
    alpha = np.zeros((H, W))
    depth = np.full((H, W), np.nan)
    med = np.full((H, W), np.nan)
    for i in range(H):
        for j in range(W):
            T, c, zs = 1.0, np.zeros(3), 0.0
            for z, u, v, Q, r, o, col in gs:
                d = np.array([j + 0.5 - u, i + 0.5 - v])
                if abs(d[0]) > r or abs(d[1]) > r:
                    continue
                a = min(0.99, o * math.exp(-0.5 * d @ Q @ d))
                if a < 1 / 255:
                    continue
                t_next = T * (1 - a)
                if t_next < 1e-4:
                    break
                c += a * T * col
                zs += a * T * z
                if np.isnan(med[i, j]) and t_next <= 0.5:
                    med[i, j] = z
                T = t_next
            rgb[i, j], alpha[i, j] = c, 1 - T
            if 1 - T >= 1e-3:
                depth[i, j] = zs / (1 - T)
    return {"rgb": rgb, "alpha": alpha, "depth": depth, "median": med}


# --------------------------------------------------------------------------- cameras
def test_orbit_camera_azimuth_contract():
    target = np.array([0.2, 0.9, -2.5])
    d = 4.0
    for phi, offset in ((0, (0, 0, 1)), (90, (1, 0, 0)), (180, (0, 0, -1)), (270, (-1, 0, 0))):
        cam = orbit_camera(target, phi, distance=d, width=64, height=48, m=0.01)
        np.testing.assert_allclose(cam.center, target + d * np.array(offset), atol=1e-12)
        np.testing.assert_allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(cam.R) == pytest.approx(1.0)
        np.testing.assert_allclose(cam.R[1], [0, -1, 0], atol=1e-12)  # image down = world down
        u, v, z = cam.project(target)
        assert (u, v, z) == pytest.approx((32.0, 24.0, d))
    # phi = 0: image right is world +x (the subject's left when it faces +z)
    cam0 = orbit_camera(target, 0, distance=d, width=64, height=48, m=0.01)
    np.testing.assert_allclose(cam0.R[0], [1, 0, 0], atol=1e-12)
    # phi = 180 mirrors it
    cam180 = orbit_camera(target, 180, distance=d, width=64, height=48, m=0.01)
    np.testing.assert_allclose(cam180.R[0], [-1, 0, 0], atol=1e-12)
    # raised camera looks down; perspective camera keeps the target in the centre
    up = orbit_camera(target, 30, distance=d, width=64, height=48, kind="persp", fov_deg=50, elevation_deg=20)
    assert up.center[1] > target[1] and up.forward[1] < 0
    assert up.project(target)[:2] == pytest.approx((32.0, 24.0))
    assert up.fx == pytest.approx(32.0 / math.tan(math.radians(25.0)))


def test_orbit_camera_in_sharp_frame():
    """In SHARP's frame C (y down, the subject faces the camera at -z): phi = 90 puts the
    camera on the photo camera's right (+x), phi = 0 between subject and photo camera."""
    c = np.array([0.1, 0.2, 2.5])
    up_C, front_C = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    cam0 = orbit_camera(c, 0, up=up_C, front=front_C, distance=2.5, width=32, height=32, m=0.01)
    np.testing.assert_allclose(cam0.center, c - [0, 0, 2.5], atol=1e-12)
    np.testing.assert_allclose(cam0.R, np.eye(3), atol=1e-12)  # = the photo camera's axes
    cam90 = orbit_camera(c, 90, up=up_C, front=front_C, distance=2.5, width=32, height=32, m=0.01)
    np.testing.assert_allclose(cam90.center, c + [2.5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(cam90.R[0], [0, 0, 1], atol=1e-12)  # design: right(90) = z^


def test_camera_validation():
    with pytest.raises(InputError):
        SplatCamera(R=np.diag([1.0, 2.0, 1.0]), center=np.zeros(3), width=8, height=8, fx=10, fy=10)
    with pytest.raises(InputError):
        SplatCamera(R=np.eye(3), center=np.zeros(3), width=8, height=8, kind="ortho", m=0.0)
    with pytest.raises(InputError):
        SplatCamera(R=np.eye(3), center=np.zeros(3), width=0, height=8, fx=10, fy=10)
    with pytest.raises(InputError):
        SplatCamera(R=np.eye(3), center=np.zeros(3), width=8, height=8, kind="fisheye", fx=10, fy=10)
    with pytest.raises(InputError):
        SplatCamera(R=np.eye(3), center=np.zeros(3), width=8, height=8, fx=None, fy=10)
    with pytest.raises(InputError):
        orbit_camera((0, 0, 0), 0, distance=1, width=8, height=8, kind="persp")  # no fov
    with pytest.raises(InputError):
        orbit_camera((0, 0, 0), 0, distance=1, width=8, height=8)  # ortho without m
    with pytest.raises(InputError):
        orbit_camera((0, 0, 0), 0, up=(0, 0, 1), distance=1, width=8, height=8, m=0.1)  # front || up
    with pytest.raises(InputError):
        orbit_camera((0, 0, 0), 0, distance=None, width=8, height=8, m=0.1)
    with pytest.raises(InputError):
        orbit_camera((0, 0, 0), 0, distance=1, width=8, height=8, m=0.1, elevation_deg=90)
    cam = SplatCamera.pinhole(100.0, 40, 30)
    assert (cam.cx, cam.cy, cam.kind) == (20.0, 15.0, "persp")
    assert cam.project(np.array([0.1, -0.2, 2.0]))[:2] == pytest.approx((25.0, 5.0))
    sc = cloud([[0, 0, 2]], 0.05, 0.5, 0.9)
    with pytest.raises(InputError):
        render_splats(sc, cam, band_rows=0)
    with pytest.raises(InputError):
        render_splats(sc, cam, background=(300, 0, 0))
    with pytest.raises(InputError):
        render_splats(sc, cam, max_radius_px=0)
    with pytest.raises(InputError):
        render_splats(sc, "camera")
    with pytest.raises(InputError):
        render_splats(SplatCloud(sc.means, sc.scales[:, :2], sc.quats, sc.colors, sc.opacity), cam)


# --------------------------------------------------------------------------- analytic
def test_single_gaussian_matches_the_analytic_2d_gaussian():
    W = H = 101  # the optical axis hits the centre of pixel (50, 50)
    f, z, s, o = 200.0, 2.0, 0.08, 0.8
    cam = SplatCamera.pinhole(f, W, H)
    r = render_splats(cloud([[0, 0, z]], s, (1.0, 0.5, 0.25), o), cam)
    sigma = math.sqrt((f * s / z) ** 2 + sr.LOWPASS_PX2)  # J Sigma J^T + 0.3 I
    assert r.alpha.max() == pytest.approx(o, rel=1e-6)
    assert np.unravel_index(np.argmax(r.alpha), r.alpha.shape) == (50, 50)
    for prof in (r.alpha[50], r.alpha[:, 50]):  # log-linear fit along the row and the column
        d = np.arange(W) + 0.5 - W / 2
        keep = prof > 0.02
        k = np.polyfit(d[keep] ** 2, np.log(prof[keep]), 1)[0]
        fit = math.sqrt(-1.0 / (2.0 * k))
        assert fit == pytest.approx(sigma, rel=0.02)
        assert fit == pytest.approx(sigma, rel=1e-4)  # in fact exact
    # colour is premultiplied by alpha over a black background
    assert r.rgb[50, 50] == pytest.approx(np.rint(np.array([1.0, 0.5, 0.25]) * o * 255), abs=1)
    assert r.depth[50, 50] == pytest.approx(z)
    assert r.depth_median[50, 50] == pytest.approx(z)  # alpha 0.8 >= 0.5 at the centre
    assert np.isnan(r.depth[0, 0]) and np.isnan(r.depth_median[0, 0])


def test_anisotropic_gaussian_orthographic_matches_formula():
    m, W, H = 0.005, 80, 60
    ang = math.radians(30.0)
    rot = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
    scales = np.array([0.05, 0.02, 0.01])
    cam = SplatCamera(R=np.eye(3), center=np.zeros(3), width=W, height=H, kind="ortho", m=m)
    o = 0.7
    sc = cloud([[0.012, -0.007, 1.0]], scales, 1.0, o, quats=rotmat_to_quat(rot[None]))
    r = render_splats(sc, cam)
    S2 = rot[:2, :2] @ np.diag(scales[:2] ** 2) @ rot[:2, :2].T / m**2 + 0.3 * np.eye(2)
    Q = np.linalg.inv(S2)
    rad = math.ceil(3 * math.sqrt(np.linalg.eigvalsh(S2).max()))
    u, v = 0.012 / m + W / 2, -0.007 / m + H / 2
    jj, ii = np.meshgrid(np.arange(W) + 0.5 - u, np.arange(H) + 0.5 - v)
    q = Q[0, 0] * jj**2 + 2 * Q[0, 1] * jj * ii + Q[1, 1] * ii**2
    want = np.minimum(0.99, o * np.exp(-0.5 * q))
    want[(want < 1 / 255) | (np.abs(jj) > rad) | (np.abs(ii) > rad)] = 0.0
    np.testing.assert_allclose(r.alpha, want, atol=1e-6)
    # the long axis is along the rotated x axis: second moments give its direction and sigma
    w = r.alpha.astype(np.float64)
    cov = np.array([[(w * jj * jj).sum(), (w * jj * ii).sum()], [(w * jj * ii).sum(), (w * ii * ii).sum()]])
    ev, evec = np.linalg.eigh(cov / w.sum())
    assert abs(abs(evec[:, 1] @ rot[:2, 0]) - 1.0) < 1e-3
    assert math.sqrt(ev[1]) == pytest.approx(math.sqrt(np.linalg.eigvalsh(S2)[1]), rel=0.02)


def test_depth_order_near_occludes_far():
    cam = SplatCamera.pinhole(100.0, 41, 41)
    near = ([0, 0, 2.0], [0.06], (1.0, 0.0, 0.0), 0.9)
    far = ([0, 0, 3.0], [0.09], (0.0, 0.0, 1.0), 0.9)
    a = render_splats(cloud(*zip(near, far, strict=True)), cam)
    b = render_splats(cloud(*zip(far, near, strict=True)), cam)  # input order is irrelevant
    np.testing.assert_array_equal(a.rgb, b.rgb)
    np.testing.assert_array_equal(a.alpha, b.alpha)
    c = a.rgb[20, 20].astype(int)
    assert c == pytest.approx([0.9 * 255, 0, 0.1 * 0.9 * 255], abs=1)  # float32 opacity: +-1 level
    assert a.alpha[20, 20] == pytest.approx(0.99, abs=1e-6)
    assert a.depth_median[20, 20] == pytest.approx(2.0)
    assert a.depth[20, 20] == pytest.approx((0.9 * 2.0 + 0.09 * 3.0) / 0.99, rel=1e-6)
    # a translucent near Gaussian does not define the median depth
    thin = render_splats(cloud([[0, 0, 2.0], [0, 0, 3.0]], [[0.06], [0.09]], 1.0, [0.3, 0.9]), cam)
    assert thin.depth_median[20, 20] == pytest.approx(3.0)


def test_early_termination_like_3dgs():
    """A pixel stops before its transmittance would drop below 1e-4: behind two 0.97
    layers (T = 9e-4) a third one (T would be 2.7e-5) and everything after it is skipped."""
    cam = SplatCamera.pinhole(100.0, 21, 21)
    cols = [(1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0), (1.0, 1.0, 1.0)]
    sc = cloud([[0, 0, z] for z in (2.0, 2.5, 3.0, 3.5)], [[0.2]], cols, 0.97)
    r = render_splats(sc, cam)
    assert r.alpha[10, 10] == pytest.approx(1 - 0.03**2, abs=1e-6)
    assert r.rgb[10, 10] == pytest.approx([0.97 * 255, 0.97 * 0.03 * 255, 0], abs=1)
    assert r.depth[10, 10] == pytest.approx((0.97 * 2.0 + 0.97 * 0.03 * 2.5) / (1 - 0.03**2), rel=1e-6)


def test_spherical_harmonics_add_view_dependent_colour():
    rng = np.random.default_rng(5)
    sh = rng.normal(scale=0.1, size=(1, 3, 3)).astype(np.float32)
    sc = cloud([[0.3, -0.2, 2.0]], 0.05, (0.4, 0.5, 0.6), 0.95, sh_rest=sh)
    cam = SplatCamera.pinhole(100.0, 64, 64)
    r = render_splats(sc, cam)
    u, v, _ = cam.project(sc.means[0])
    i, j = int(v), int(u)
    d = sc.means[0] / np.linalg.norm(sc.means[0])
    want = np.clip(sc.colors[0] + _sh_basis(d[None], 3)[0] @ sh[0], 0, 1)
    a = float(r.alpha[i, j])
    np.testing.assert_allclose(r.rgb[i, j] / 255.0, want * a, atol=1.0 / 255)


# --------------------------------------------------------------------------- consistency
def test_orthographic_matches_a_far_telephoto():
    rng = np.random.default_rng(1)
    sc = random_cloud(rng, 300, center=(0.0, 0.0, 0.0), spread=0.4, s=(0.005, 0.04))
    m, D = 0.01, 1000.0
    ortho = orbit_camera((0, 0, 0), 30, distance=D, width=100, height=90, m=m)
    tele = SplatCamera(R=ortho.R, center=ortho.center, width=100, height=90, kind="persp", fx=D / m, fy=D / m)
    a = render_splats(sc, ortho)
    b = render_splats(sc, tele)
    assert np.abs(a.alpha - b.alpha).max() < 0.02
    assert np.abs(a.rgb.astype(int) - b.rgb.astype(int)).mean() < 0.5
    both = np.isfinite(a.depth) & np.isfinite(b.depth)
    assert both.sum() > 0.9 * np.isfinite(a.depth).sum()
    # the mean depth mixes Gaussians ~0.8 m apart whose weights differ by ~1 %
    assert np.median(np.abs(a.depth[both] - b.depth[both])) < 1e-3
    assert np.abs(a.depth[both] - b.depth[both]).max() < 0.1
    med = np.isfinite(a.depth_median) & np.isfinite(b.depth_median)
    assert (np.abs(a.depth_median[med] - b.depth_median[med]) < 1e-3).mean() > 0.98


def _assert_same(a: sr.SplatRender, b: sr.SplatRender) -> None:
    np.testing.assert_array_equal(a.rgb, b.rgb)
    np.testing.assert_array_equal(a.alpha, b.alpha)
    np.testing.assert_array_equal(a.depth, b.depth)  # NaNs compare equal here
    np.testing.assert_array_equal(a.depth_median, b.depth_median)


def test_band_sizes_and_runs_give_identical_images(monkeypatch):
    rng = np.random.default_rng(2)
    sc = random_cloud(rng, 1500, spread=0.5, s=(0.01, 0.08))  # dense: many pixels terminate early
    cam = SplatCamera.pinhole(60.0, 64, 48)
    ref = render_splats(sc, cam)
    assert ref.alpha.max() > 0.999  # early termination happens
    assert ref.diag["bands"] == 1
    for rows in (1, 5, 17, 48, 100):
        _assert_same(render_splats(sc, cam, band_rows=rows), ref)
    monkeypatch.setattr(sr, "_RUN_PAIRS", 300)  # many depth-ordered runs per band
    monkeypatch.setattr(sr, "_PAIR_BUDGET", 2000)  # many automatic bands
    many = render_splats(sc, cam)
    assert many.diag["bands"] > 5 and many.diag["runs"] > 20
    assert many.diag["pairs"] < ref.diag["pairs"]  # pairs at finished pixels were skipped
    _assert_same(many, ref)
    _assert_same(render_splats(sc, cam, band_rows=7), ref)


def concat(*clouds: SplatCloud) -> SplatCloud:
    return SplatCloud(
        *(
            np.concatenate([getattr(c, f) for c in clouds])
            for f in SplatCloud.__dataclass_fields__
            if f != "sh_rest"
        )
    )


@pytest.mark.parametrize("kind", ["persp", "ortho"])
def test_matches_brute_force_reference(kind):
    rng = np.random.default_rng(3 if kind == "persp" else 4)
    sc = random_cloud(rng, 20, spread=0.7, s=(0.02, 0.25))
    # plus a stack of opaque Gaussians: pixels behind it stop early (T < 1e-4)
    stack = cloud([[0.05, 0.02, z] for z in (2.6, 2.8, 3.0, 3.2)], [[0.25]], rng.random((4, 3)), 0.97)
    sc = concat(sc, stack)
    if kind == "persp":
        cam = SplatCamera(
            R=np.eye(3), center=np.zeros(3), width=40, height=32, fx=45.0, fy=50.0, cx=19.3, cy=17.1
        )
    else:
        cam = orbit_camera(
            (0, 0, 3), 20, up=(0, -1, 0), front=(0, 0, -1), distance=5, width=40, height=32, m=0.05
        )
    got = render_splats(sc, cam, band_rows=7)
    want = brute_force(sc, cam)
    assert got.diag["visible"] >= 19
    assert got.alpha.max() > 0.9998  # early termination is exercised
    np.testing.assert_allclose(got.alpha, want["alpha"], atol=1e-3)
    assert np.abs(got.rgb / 255.0 - want["rgb"]).max() <= 0.5 / 255 + 1e-3
    np.testing.assert_array_equal(np.isnan(got.depth), np.isnan(want["depth"]))
    np.testing.assert_allclose(got.depth, want["depth"], atol=1e-3)
    np.testing.assert_array_equal(np.isnan(got.depth_median), np.isnan(want["median"]))
    np.testing.assert_allclose(got.depth_median, want["median"], atol=1e-5)


def test_radius_cap_matches_brute_force():
    rng = np.random.default_rng(6)
    sc = random_cloud(rng, 6, spread=0.3, s=(0.15, 0.3))  # footprints of 10-20 px sigma
    cam = SplatCamera.pinhole(80.0, 36, 30)
    got = render_splats(sc, cam, max_radius_px=6)
    want = brute_force(sc, cam, max_r=6)
    np.testing.assert_allclose(got.alpha, want["alpha"], atol=1e-3)
    assert np.abs(got.rgb / 255.0 - want["rgb"]).max() <= 0.5 / 255 + 1e-3
    assert not np.allclose(got.alpha, render_splats(sc, cam).alpha, atol=1e-3)  # the cap matters


def test_front_untouched_from_the_photo_camera():
    """Design 5.2: the fused SHARP 360 result seen from the photo's camera looks like the
    cleaned SHARP front alone inside the eroded matte (mean difference < 2/255)."""
    from vrphoto.geometry.back_mould import shell_back
    from vrphoto.geometry.splat_fusion import canonical_frame, sharp_front
    from vrphoto.geometry.splat_merge import merge_parts

    body, cam = sp.make_body(), sp.make_cam()
    scene = sp.sharp_like_scene(body, cam)
    front = sharp_front(scene.sc, scene.alpha, scene.cam, scene.rgb, mould_long=128)
    canon = canonical_frame(front, scene.sc, scene.alpha)
    merged, _, _ = merge_parts(front, shell_back(front, canon), seam_deg=60, seed=0)
    assert len(merged) > len(front.sc)  # a back was added
    photo_cam = SplatCamera.pinhole(cam.f, cam.width, cam.height)
    alone = render_splats(front.sc, photo_cam)
    fused = render_splats(merged, photo_cam)
    inner = cv2.erode((scene.alpha >= 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert inner.sum() > 500
    diff = np.abs(alone.rgb.astype(float) - fused.rgb.astype(float)).mean(axis=2)
    assert diff[inner].mean() < 2.0
    # and the front alone reproduces the photo there (the renderer agrees with the fixture)
    photo_diff = np.abs(alone.rgb.astype(float) - scene.rgb.astype(float)).mean(axis=2)
    assert np.median(photo_diff[inner]) < 6.0


def test_prepared_cloud_renders_identically():
    rng = np.random.default_rng(8)
    sc = random_cloud(rng, 400)
    sc.sh_rest = rng.normal(scale=0.05, size=(400, 8, 3)).astype(np.float32)
    sc.opacity[:5] = 0.001  # dropped once by prepare_splats
    pre = sr.prepare_splats(sc)
    assert len(pre) == 395 and pre.n_input == 400
    for cam in (
        SplatCamera.pinhole(60.0, 48, 40),
        orbit_camera((0, 0, 3), 70, up=(0, -1, 0), front=(0, 0, -1), distance=4, width=48, height=40, m=0.03),
    ):
        a, b = render_splats(sc, cam), render_splats(pre, cam)
        _assert_same(a, b)
        assert b.diag["n"] == 400


# --------------------------------------------------------------------------- edge cases
def test_empty_and_invisible_clouds_render_the_background():
    cam = SplatCamera.pinhole(50.0, 20, 10)
    empty = SplatCloud(
        means=np.zeros((0, 3), np.float32),
        scales=np.zeros((0, 3), np.float32),
        quats=np.zeros((0, 4), np.float32),
        colors=np.zeros((0, 3), np.float32),
        opacity=np.zeros(0, np.float32),
    )
    behind = cloud([[0, 0, -1.0], [0, 0, 0.005]], 0.1, 1.0, 1.0)  # behind / at the camera plane
    faint = cloud([[0, 0, 2.0]], 0.1, 1.0, 0.003)  # opacity below 1/255
    outside = cloud([[50.0, 0, 2.0]], 0.01, 1.0, 1.0)  # far outside the frustum
    for sc in (empty, behind, faint, outside):
        r = render_splats(sc, cam, background=(10, 20, 30))
        assert r.rgb.shape == (10, 20, 3) and r.rgb.dtype == np.uint8
        assert (r.rgb == np.array([10, 20, 30], np.uint8)).all()
        assert (r.alpha == 0).all() and r.alpha.dtype == np.float32
        assert np.isnan(r.depth).all() and np.isnan(r.depth_median).all()
        assert r.diag["visible"] == 0
    with pytest.raises(InputError):
        turntable_sheet(empty)


def test_non_finite_rows_are_ignored():
    cam = SplatCamera.pinhole(100.0, 41, 41)
    good = cloud([[0, 0, 2.0]], 0.06, 1.0, 0.9)
    bad = cloud([[0, 0, 2.0], [np.nan, 0, 2.0], [0, 0, 2.0]], [[0.06], [0.06], [np.inf]], 1.0, 0.9)
    bad.quats[0] = np.nan
    a = render_splats(good, cam)
    b = render_splats(bad, cam)
    assert b.diag["visible"] == 0
    assert a.diag["visible"] == 1


def test_cancel_and_progress():
    rng = np.random.default_rng(7)
    sc = random_cloud(rng, 200)
    cam = SplatCamera.pinhole(60.0, 48, 40)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        render_splats(sc, cam, cancel=tok)
    with pytest.raises(Cancelled):
        turntable_sheet(sc, n=2, size=32, cancel=tok)
    seen: list[float] = []
    render_splats(sc, cam, band_rows=8, progress=lambda f, msg: seen.append(f))
    assert seen[0] == 0.0 and seen[-1] == 1.0 and seen == sorted(seen) and len(seen) >= 6

    calls = {"n": 0}

    class LateCancel(CancelToken):
        def check(self) -> None:  # fires on the 4th check: inside the band loop
            calls["n"] += 1
            if calls["n"] >= 4:
                raise Cancelled("stop")

    with pytest.raises(Cancelled):
        render_splats(sc, cam, band_rows=4, cancel=LateCancel())


# --------------------------------------------------------------------------- turntable
def body_cloud(body: sp.Body, spacing: float = 0.015) -> SplatCloud:
    """The synthetic person as a closed surface cloud in the glTF frame (y up, facing +z),
    coloured by side like the photo: front red, back blue, own left green, right yellow."""
    pts, nrm = [], []
    gold = math.pi * (3.0 - math.sqrt(5.0))
    for k in range(len(body.radii)):
        a = body.radii[k]
        p = 1.6075
        area = 4 * math.pi * (((a[0] * a[1]) ** p + (a[0] * a[2]) ** p + (a[1] * a[2]) ** p) / 3) ** (1 / p)
        n = max(64, int(area / spacing**2))
        i = np.arange(n) + 0.5
        zz = 1 - 2 * i / n
        rr = np.sqrt(1 - zz * zz)
        d = np.stack([rr * np.cos(gold * i), rr * np.sin(gold * i), zz], axis=1)  # Fibonacci sphere
        P = body.centers[k] + (d * a) @ body.rot[k].T
        N = (d / a) @ body.rot[k].T
        inside = np.zeros(n, bool)
        for j in range(len(body.radii)):  # drop points buried in another part
            if j != k:
                loc = (P - body.centers[j]) @ body.rot[j] / body.radii[j]
                inside |= (loc * loc).sum(1) < 0.98
        pts.append(P[~inside])
        nrm.append(N[~inside] / np.linalg.norm(N[~inside], axis=1, keepdims=True))
    P, N = np.concatenate(pts), np.concatenate(nrm)
    return cloud(P * GLTF, 0.7 * spacing, body.colour_of(N) / 255.0, 0.95)


def test_turntable_of_the_synthetic_person():
    body = sp.make_body()
    sc = body_cloud(body)
    size, n = 96, 8
    sheet = turntable_sheet(sc, n=n, size=size, cols=4, labels=False)
    assert sheet.shape == (2 * size, 4 * size, 3) and sheet.dtype == np.uint8
    tiles = [
        sheet[(k // 4) * size : (k // 4 + 1) * size, (k % 4) * size : (k % 4 + 1) * size] for k in range(n)
    ]

    def subject(tile: np.ndarray) -> np.ndarray:
        return np.abs(tile.astype(int) - 32).max(axis=2) > 40

    def majority(tile: np.ndarray) -> str:
        names, counts = np.unique(sp.classify_rgb(tile[subject(tile)], sp.SIDES), return_counts=True)
        return str(names[np.argmax(counts)])

    assert majority(tiles[0]) == "front"  # phi = 0: the photographed front
    assert majority(tiles[2]) == "left"  # phi = 90: you walked to your right -> the subject's left
    assert majority(tiles[4]) == "back"  # phi = 180
    assert majority(tiles[6]) == "right"
    for t in tiles:  # every view shows the whole person, nearly as tall as the tile
        rows = np.flatnonzero(subject(t).any(axis=1))
        assert 0.75 * size < rows[-1] - rows[0] < 0.97 * size

    def arm_reach(tile: np.ndarray) -> tuple[int, int]:
        """(columns right of the head, columns left of it) in the top 30 % of the subject."""
        mask = subject(tile)
        rows = np.flatnonzero(mask.any(axis=1))
        top = rows[0]
        head = np.flatnonzero(mask[top : top + 3].any(axis=0)).mean()
        band = mask[top : top + int(0.3 * (rows[-1] - top))]
        cols = np.flatnonzero(band.any(axis=0))
        return int(cols.max() - head), int(head - cols.min())

    right0, left0 = arm_reach(tiles[0])
    right180, left180 = arm_reach(tiles[4])
    assert right0 > 2 * left0  # the raised LEFT arm is on the image right from the front ...
    assert left180 > 2 * right180  # ... and on the image left from behind (a mirror image)
    labelled = turntable_sheet(sc, n=3, size=64, cols=2)
    assert labelled.shape == (128, 128, 3)
    assert (labelled[64:, 64:] == 32).all()  # the unused fourth tile is background


def test_turntable_ignores_far_floaters():
    body = sp.make_body()
    sc = body_cloud(body, spacing=0.03)
    rng = np.random.default_rng(9)
    n_f = 10
    floaters = cloud(rng.uniform(-30, 30, size=(n_f, 3)), 0.02, 1.0, 0.9)
    both = SplatCloud(
        means=np.concatenate([sc.means, floaters.means]),
        scales=np.concatenate([sc.scales, floaters.scales]),
        quats=np.concatenate([sc.quats, floaters.quats]),
        colors=np.concatenate([sc.colors, floaters.colors]),
        opacity=np.concatenate([sc.opacity, floaters.opacity]),
    )
    a = turntable_sheet(sc, n=1, size=64, labels=False)
    b = turntable_sheet(both, n=1, size=64, labels=False)
    rows_a = np.flatnonzero((np.abs(a.astype(int) - 32).max(axis=2) > 40).any(axis=1))
    rows_b = np.flatnonzero((np.abs(b.astype(int) - 32).max(axis=2) > 40).any(axis=1))
    assert abs((rows_b[-1] - rows_b[0]) - (rows_a[-1] - rows_a[0])) <= 2  # same framing


# --------------------------------------------------------------------------- performance
def _person_like_cloud(n: int, seed: int = 0) -> SplatCloud:
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    P = d * np.array([0.25, 0.85, 0.15]) + np.array([0.0, 0.85, 0.0]) + rng.normal(scale=0.01, size=(n, 3))
    return cloud(
        P,
        np.exp(rng.uniform(np.log(0.002), np.log(0.01), size=(n, 3))),
        rng.random((n, 3)),
        rng.uniform(0.3, 1.0, n),
        quats=random_quats(rng, n),
    )


@pytest.mark.slow
def test_300k_gaussians_at_512_px_is_fast():
    sc = _person_like_cloud(300_000)
    cam = orbit_camera((0.0, 0.85, 0.0), 30, distance=5.0, width=512, height=512, m=2.0 / 512)
    t0 = time.perf_counter()
    r = render_splats(sc, cam)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0, elapsed
    assert r.diag["visible"] == 300_000
    assert 0.08 < (r.alpha > 0.5).mean() < 0.3
