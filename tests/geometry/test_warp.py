"""Tests for geometry.warp: fixed-point backward warping (algorithm D) and hole filling."""

from __future__ import annotations

import logging
import math
import time

import cv2
import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, Intrinsics, Layer, LayeredImage, Pose, RenderResult
from vrphoto.geometry import warp
from vrphoto.geometry.warp import fill_holes, hole_fraction, render_filled, render_view


# --------------------------------------------------------------------------- helpers
def _ones(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), np.float32)


def _single(img: np.ndarray, inv_z: np.ndarray | float, K: Intrinsics, alpha=None) -> LayeredImage:
    h, w = img.shape[:2]
    if np.isscalar(inv_z):
        inv_z = np.full((h, w), float(inv_z), np.float32)
    return LayeredImage(
        [Layer(img, np.asarray(inv_z, np.float32), _ones(h, w) if alpha is None else alpha)], K
    )


def _smooth_texture(h: int, w: int, seed: int = 0, sigma: float = 2.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.random((h, w, 3)).astype(np.float32)
    tex = cv2.GaussianBlur(noise, (0, 0), sigma)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return (tex * 255.0 + 0.5).astype(np.uint8)


def _blob_image(h: int, w: int, centers: list[tuple[float, float]], sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w), np.float64)
    for cx, cy in centers:
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    img = np.clip(img * 250.0, 0, 255)
    return np.repeat(np.rint(img).astype(np.uint8)[..., None], 3, axis=2)


def _centroid(img: np.ndarray, cx: float, cy: float, r: int = 12) -> tuple[float, float]:
    g = img[..., 0].astype(np.float64)
    x0, y0 = int(round(cx)), int(round(cy))
    patch = g[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
    yy, xx = np.mgrid[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
    s = patch.sum()
    return float((patch * xx).sum() / s), float((patch * yy).sum() / s)


def _shift(img: np.ndarray, dx: float, dy: float = 0.0) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- golden tests
def test_identity_reproduces_image(two_plane_scene, psnr_fn):
    s = two_plane_scene
    li = _single(s.img, s.inv_z, s.K)
    rr = render_view(li, Pose.identity())
    assert rr.rgb.shape == s.img.shape and rr.rgb.dtype == np.uint8
    assert rr.alpha.shape == s.inv_z.shape and rr.alpha.dtype == np.float32
    assert psnr_fn(rr.rgb, s.img) > 45.0
    assert float(rr.alpha.min()) > 0.99


def test_identity_reproduces_layered_image(two_plane_scene, psnr_fn):
    s = two_plane_scene
    h, w = s.img.shape[:2]
    back = Layer(s.far_rgb, np.full((h, w), 1 / s.z_far, np.float32), _ones(h, w))
    front = Layer(s.img, s.inv_z, _ones(h, w))
    rr = render_view(LayeredImage([back, front], s.K), Pose.identity())
    assert psnr_fn(rr.rgb, s.img) > 45.0


@pytest.mark.parametrize("tx", [0.1, -0.1, 0.037])
def test_constant_depth_translation_shift_is_fB_over_Z(tx):
    """Golden: camera moves by B over a plane at Z -> content shifts by f*B/Z px."""
    h, w = 200, 320
    K = Intrinsics.from_hfov(w, h, 60.0)
    Z = 2.5
    centers = [(80.3, 60.7), (160.0, 100.2), (240.6, 140.4), (130.1, 150.9)]
    img = _blob_image(h, w, centers)
    rr = render_view(_single(img, 1.0 / Z, K), Pose.translate(tx, 0.0, 0.0))
    expected = -K.fx * tx / Z  # camera right -> content left
    ref = [_centroid(img, cx, cy) for cx, cy in centers]
    for (rx, ry), (cx, cy) in zip(ref, centers):
        mx, my = _centroid(rr.rgb, cx + expected, cy)
        assert mx - rx == pytest.approx(expected, abs=0.05)
        assert my - ry == pytest.approx(0.0, abs=0.05)


def test_constant_depth_shift_textured_matches_reference(psnr_fn):
    h, w = 180, 240
    K = Intrinsics.from_hfov(w, h, 55.0)
    Z, tx = 3.0, 0.08
    img = _smooth_texture(h, w, 4)
    rr = render_view(_single(img, 1.0 / Z, K), Pose.translate(tx))
    d = -K.fx * tx / Z
    ref = _shift(img, d)
    m = int(math.ceil(abs(d))) + 2
    assert psnr_fn(rr.rgb[:, m:-m], ref[:, m:-m]) > 40.0
    # the uncovered strip on the right is a hole, the rest is covered
    assert (rr.alpha[:, -int(abs(d)) + 1 :] < 0.5).all()
    assert (rr.alpha[:, : w - m] > 0.99).all()


def test_vertical_translation_direction():
    h, w = 200, 200
    K = Intrinsics.from_hfov(w, h, 60.0)
    centers = [(100.0, 100.0)]
    img = _blob_image(h, w, centers)
    rr = render_view(_single(img, 0.5, K), Pose.translate(0.0, 0.1, 0.0))
    # y points down: moving the camera down shifts content UP by f*B/Z
    exp = -K.fy * 0.1 * 0.5
    mx, my = _centroid(rr.rgb, 100.0, 100.0 + exp)
    assert my - 100.0 == pytest.approx(exp, abs=0.05)
    assert mx - 100.0 == pytest.approx(0.0, abs=0.05)


def test_forward_translation_magnifies():
    h, w = 200, 300
    K = Intrinsics.from_hfov(w, h, 60.0)
    Z, tz = 2.0, 0.5
    c = (K.cx + 60.0, K.cy + 30.0)
    img = _blob_image(h, w, [c])
    rr = render_view(_single(img, 1.0 / Z, K), Pose.translate(0.0, 0.0, tz))
    s = Z / (Z - tz)
    exp = (K.cx + 60.0 * s, K.cy + 30.0 * s)
    mx, my = _centroid(rr.rgb, *exp, r=16)
    assert mx == pytest.approx(exp[0], abs=0.15)
    assert my == pytest.approx(exp[1], abs=0.15)


def test_rotation_yaw_moves_content_left():
    h, w = 200, 300
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = _blob_image(h, w, [(K.cx, K.cy)])
    rr = render_view(_single(img, 1e-4, K), Pose.look(yaw_deg=5.0))
    exp = K.cx - K.fx * math.tan(math.radians(5.0))
    mx, my = _centroid(rr.rgb, exp, K.cy)
    assert mx == pytest.approx(exp, abs=0.1)
    assert my == pytest.approx(K.cy, abs=0.1)


def test_dst_K_principal_point_shift_and_scale(psnr_fn):
    h, w = 160, 200
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = _smooth_texture(h, w, 7)
    li = _single(img, 0.4, K)
    # principal point +10 px -> content moves right by 10 px
    rr = render_view(li, Pose.identity(), K.with_principal(K.cx + 10.0, K.cy))
    assert psnr_fn(rr.rgb[:, 12:], img[:, 2:-10]) > 45.0
    assert (rr.alpha[:, :9] < 0.5).all()
    # 2x larger output with doubled focal length == upscaled image
    K2 = K.scaled(2.0)
    rr2 = render_view(li, Pose.identity(), K2)
    assert rr2.rgb.shape == (2 * h, 2 * w, 3)
    ref = cv2.warpAffine(
        img,
        np.float32([[0.5, 0, 0], [0, 0.5, 0]]),
        (2 * w, 2 * h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    assert psnr_fn(rr2.rgb[4:-4, 4:-4], ref[4:-4, 4:-4]) > 40.0


def test_dst_K_overscan_canvas():
    h, w = 120, 160
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = _smooth_texture(h, w, 9)
    D = Intrinsics(K.fx, K.fy, K.cx + 20, K.cy + 10, w + 40, h + 20)
    rr = render_view(_single(img, 0.5, K), Pose.identity(), D)
    assert rr.rgb.shape == (h + 20, w + 40, 3)
    np.testing.assert_array_equal(rr.rgb[10 : 10 + h, 20 : 20 + w], img)
    assert (rr.alpha[:, :19] < 0.5).all() and (rr.alpha[:9] < 0.5).all()
    assert (rr.alpha[10 : 10 + h, 20 : 20 + w] > 0.99).all()


def test_points_behind_camera_are_dropped():
    h, w = 80, 100
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = _smooth_texture(h, w, 1)
    rr = render_view(_single(img, 0.5, K), Pose.translate(0.0, 0.0, 3.0))  # plane at 2 m is behind
    assert np.isfinite(rr.alpha).all()
    assert (rr.alpha < 0.5).all()
    assert (rr.rgb == 0).all()
    # near-plane: points closer than MIN_Z to the new camera are invalid too
    rr = render_view(_single(img, 0.5, K), Pose.translate(0.0, 0.0, 2.0 - warp.MIN_Z / 2))
    assert (rr.alpha < 0.5).all()


def test_partially_behind_camera_tilted_plane():
    h, w = 100, 160
    K = Intrinsics.from_hfov(w, h, 70.0)
    img = _smooth_texture(h, w, 2)
    # depth grows from 1 m (left) to 6 m (right); move forward 2 m -> left part behind
    z = np.linspace(1.0, 6.0, w, dtype=np.float32)[None, :].repeat(h, 0)
    rr = render_view(_single(img, 1.0 / z, K), Pose.translate(0.0, 0.0, 2.0))
    assert np.isfinite(rr.alpha).all()
    assert 0.0 < hole_fraction(rr) < 1.0


def test_nan_and_nonpositive_inverse_depth_are_safe():
    h, w = 60, 80
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = _smooth_texture(h, w, 3)
    inv = np.full((h, w), 0.5, np.float32)
    inv[10:20, 10:20] = np.nan
    inv[30:35, 40:50] = -1.0
    inv[40:45, 60:70] = np.inf
    rr = render_view(_single(img, inv, K), Pose.translate(0.03))
    assert np.isfinite(rr.alpha).all()
    assert rr.rgb.dtype == np.uint8


# --------------------------------------------------------------------------- occlusion / tears
def test_tear_becomes_transparent_no_stretching(two_plane_scene):
    """Golden: the gap opened behind the square's edge is a hole, not stretched pixels."""
    s = two_plane_scene
    li = _single(s.img, s.inv_z, s.K)
    f = s.K.fx
    for tx in (0.05, -0.05):
        rr = render_view(li, Pose.translate(tx))
        dn, df = -f * tx / s.z_near, -f * tx / s.z_far
        if tx > 0:  # square moves left more than the background -> gap at its right edge
            g0, g1 = 199 + dn, 200 + df
        else:  # gap at the left edge
            g0, g1 = 119 + df, 120 + dn
        cols = np.arange(int(math.ceil(g0 + 1.0)), int(math.floor(g1 - 1.0)) + 1)
        assert cols.size >= 3
        gap_alpha = rr.alpha[85:155][:, cols]
        assert float(gap_alpha.max()) < 0.5, "convergence alpha must drop across the tear"
        # outside the gap everything is covered
        assert float(rr.alpha[85:155, 20 : int(g0) - 1].min()) > 0.99


def test_occlusion_inside_single_layer(two_plane_scene, psnr_fn):
    """The near square slides over the far plane: its texture must win (near seed)."""
    s = two_plane_scene
    li = _single(s.img, s.inv_z, s.K)
    f = s.K.fx
    near_tex = s.img.copy()  # near texture is only defined inside the mask; compare there
    for tx in (0.05, -0.05):
        rr = render_view(li, Pose.translate(tx))
        dn = -f * tx / s.z_near
        exp = _shift(near_tex, dn)
        sq = np.zeros(s.fg_mask.shape, bool)
        x0, x1 = int(math.ceil(120 + dn)) + 1, int(math.floor(199 + dn)) - 1
        sq[82:158, x0:x1] = True
        assert psnr_fn(rr.rgb, exp, sq) > 35.0
        # the overlap zone (where the square covers formerly visible background)
        if tx > 0:
            zone = np.zeros_like(sq)
            zone[82:158, x0 : x0 + 4] = True
        else:
            zone = np.zeros_like(sq)
            zone[82:158, x1 - 4 : x1] = True
        assert psnr_fn(rr.rgb, exp, zone) > 35.0


def test_thin_foreground_pole_is_not_lost():
    h, w = 120, 200
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = np.full((h, w, 3), 40, np.uint8)
    inv = np.full((h, w), 0.1, np.float32)
    img[:, 100:103] = (255, 0, 0)
    inv[:, 100:103] = 1.0  # 1 m pole in front of a 10 m wall
    tx = 0.06
    rr = render_view(_single(img, inv, K), Pose.translate(tx))
    shift = -K.fx * tx * 1.0
    col = int(round(101 + shift))
    assert rr.rgb[60, col, 0] > 200 and rr.rgb[60, col, 1] < 50


def test_alpha_layers_composite_premultiplied():
    h, w = 40, 60
    K = Intrinsics.from_hfov(w, h, 60.0)
    back = Layer(np.full((h, w, 3), (200, 0, 0), np.uint8), np.full((h, w), 0.2, np.float32), _ones(h, w))
    fg_rgb = np.full((h, w, 3), (255, 255, 255), np.uint8)  # garbage colour where alpha == 0
    fg_rgb[:, 30:] = (0, 200, 0)
    fg_alpha = np.zeros((h, w), np.float32)
    fg_alpha[:, 30:] = 0.5
    front = Layer(fg_rgb, np.full((h, w), 0.5, np.float32), fg_alpha)
    rr = render_view(LayeredImage([back, front], K), Pose.identity())
    np.testing.assert_allclose(rr.rgb[20, 40], [100, 100, 0], atol=2)
    np.testing.assert_array_equal(rr.rgb[20, 10], [200, 0, 0])  # garbage white never bleeds
    assert rr.alpha.min() > 0.99
    # sub-pixel move: the transparent region's white must not leak into the edge
    rr = render_view(LayeredImage([back, front], K), Pose.translate(0.013))
    assert rr.rgb[20, :, 2].max() <= 2


def test_iters_parameter_and_validation(two_plane_scene):
    s = two_plane_scene
    li = _single(s.img, s.inv_z, s.K)
    rr1 = render_view(li, Pose.translate(0.02), iters=1)
    assert rr1.rgb.shape == s.img.shape
    with pytest.raises(InputError):
        render_view(li, Pose.identity(), iters=0)
    with pytest.raises(InputError):
        render_view(LayeredImage([], s.K), Pose.identity())
    with pytest.raises(InputError):
        render_view(LayeredImage([Layer(s.img[:10], s.inv_z, s.inv_z)], s.K), Pose.identity())
    with pytest.raises(InputError):
        render_view(LayeredImage([Layer(s.img, s.inv_z[:10], s.inv_z)], s.K), Pose.identity())
    with pytest.raises(InputError):
        render_view(li, Pose(np.eye(3), np.array([np.nan, 0.0, 0.0])))
    with pytest.raises(InputError):
        render_view(li, Pose(np.eye(2), np.zeros(3)))
    with pytest.raises(InputError):
        render_view(li, Pose.identity(), Intrinsics(0.0, 1.0, 0.0, 0.0, 10, 10))


def test_cancel_is_honoured(two_plane_scene):
    s = two_plane_scene
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        render_view(_single(s.img, s.inv_z, s.K), Pose.translate(0.01), cancel=tok)
    with pytest.raises(Cancelled):
        render_filled(_single(s.img, s.inv_z, s.K), Pose.translate(0.01), cancel=tok)


def test_parallel_strips_match_serial(monkeypatch, psnr_fn):
    h, w = 520, 640  # above the parallel threshold
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = _smooth_texture(h, w, 5)
    yy, xx = np.mgrid[0:h, 0:w]
    inv = (0.2 + 0.3 * ((xx // 80 + yy // 80) % 2)).astype(np.float32)
    li = _single(img, inv, K)
    pose = Pose.look(2.0, 1.0, 0.5, (0.04, -0.01, 0.02))
    par = render_view(li, pose)
    monkeypatch.setattr(warp, "_workers", lambda: 1)
    ser = render_view(li, pose)
    np.testing.assert_array_equal(par.rgb, ser.rgb)
    np.testing.assert_array_equal(par.alpha, ser.alpha)


# --------------------------------------------------------------------------- hole filling
def test_fill_holes_no_holes_returns_copy():
    rgb = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    out = fill_holes(RenderResult(rgb, np.ones((20, 30), np.float32)))
    np.testing.assert_array_equal(out, rgb)
    assert out is not rgb


def test_fill_holes_small_uses_telea(monkeypatch):
    h, w = 200, 200
    rgb = np.full((h, w, 3), 120, np.uint8)
    alpha = np.ones((h, w), np.float32)
    rgb[100, 100] = 0
    alpha[100, 100] = 0.0  # 1 px hole < 0.1 %
    calls = []
    real = cv2.inpaint

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(warp.cv2, "inpaint", spy)
    out = fill_holes(RenderResult(rgb, alpha))
    assert calls, "small holes should be inpainted with Telea"
    assert abs(int(out[100, 100, 0]) - 120) <= 2


def test_fill_holes_large_uses_push_pull(monkeypatch):
    h, w = 100, 100
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, :50] = (255, 0, 0)
    rgb[:, 50:] = (0, 0, 255)
    alpha = np.ones((h, w), np.float32)
    alpha[40:60, 40:60] = 0.0
    rgb[40:60, 40:60] = 0
    monkeypatch.setattr(warp.cv2, "inpaint", lambda *a, **k: pytest.fail("Telea used for a large hole"))
    out = fill_holes(RenderResult(rgb, alpha))
    assert (out[40:60, 40:45, 0] > 150).all()  # left of the hole -> red
    assert (out[40:60, 55:60, 2] > 150).all()  # right of the hole -> blue
    np.testing.assert_array_equal(out[alpha >= 0.5], rgb[alpha >= 0.5])


def test_fill_holes_blends_partial_coverage():
    h, w = 50, 50
    rgb = np.full((h, w, 3), 100, np.uint8)
    alpha = np.ones((h, w), np.float32)
    rgb[20:30, 20:30] = 200
    alpha[20:30, 20:30] = 0.25
    out = fill_holes(RenderResult(rgb, alpha), small_frac=0.0)
    # fill ~ 100 from the surroundings, blended with 25 % of the rendered 200
    assert 115 <= int(out[25, 25, 0]) <= 135


def test_fill_holes_all_holes_warns(caplog):
    rr = RenderResult(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.float32))
    with caplog.at_level(logging.WARNING):
        out = fill_holes(rr)
    assert (out == 0).all()
    with pytest.raises(InputError):
        fill_holes(RenderResult(np.zeros((10, 10, 3), np.uint8), np.zeros((5, 5), np.float32)))


def test_render_filled_has_no_holes(two_plane_scene):
    s = two_plane_scene
    out = render_filled(_single(s.img, s.inv_z, s.K), Pose.translate(0.05))
    assert out.shape == s.img.shape and out.dtype == np.uint8
    # the right border strip (no source) is filled, not black
    assert out[:, -2:].mean() > 20


# --------------------------------------------------------------------------- performance
@pytest.mark.slow
def test_performance_1080p_two_layers():
    h, w = 1080, 1920
    K = Intrinsics.from_hfov(w, h, 65.0)
    rng = np.random.default_rng(0)
    img = cv2.resize(rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8), (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.clip(0.3 + 0.5 * np.sin(xx / 200.0) * np.cos(yy / 150.0), 0, 1).astype(np.float32)
    d[300:700, 600:1100] = 0.95
    inv = (1 / 15 + d * (1 / 1.2 - 1 / 15)).astype(np.float32)
    li = LayeredImage([Layer(img, inv * 0.8, _ones(h, w)), Layer(img, inv, _ones(h, w))], K)
    render_view(li, Pose.translate(-0.0315))  # warm-up
    best = math.inf
    for _ in range(3):
        t = time.perf_counter()
        render_view(li, Pose.translate(-0.0315))
        best = min(best, time.perf_counter() - t)
    # target is < 0.5 s on 4 cores; generous bound to stay robust on loaded CI machines
    assert best < 1.5, f"1080p two-layer render took {best:.3f} s"
    # rotated camera (video orbit/swing paths): preconditioned steps cost a bit more
    pose = Pose.look(8.0, 2.0, 0.0, (0.04, 0.0, 0.05))
    best = math.inf
    for _ in range(2):
        t = time.perf_counter()
        rr = render_view(li, pose)
        best = min(best, time.perf_counter() - t)
    assert best < 2.5, f"rotated 1080p two-layer render took {best:.3f} s"
    assert hole_fraction(rr) < 0.2


def test_push_pull_fill_fast_large_image_path():
    h, w = 64, 4200  # long side > 2048 -> pooled path
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (250, 10, 10)
    img[:, w // 2 :] = (10, 10, 250)
    known = np.ones((h, w), bool)
    known[20:40, 1000:1100] = False  # inside the red half
    known[:, w // 2 - 50 : w // 2 + 50] = False  # straddling the boundary
    out = warp.push_pull_fill_fast(img, known)
    assert out.dtype == np.uint8 and out.shape == img.shape
    np.testing.assert_array_equal(out[known], img[known])
    assert (out[20:40, 1000:1100, 0] > 200).all()
    assert out[30, w // 2 - 45, 0] > out[30, w // 2 + 45, 0]
    # 2-D float input
    dep = np.full((h, w), 0.25, np.float32)
    dep[:, w // 2 :] = 0.5
    fd = warp.push_pull_fill_fast(dep, known)
    assert fd.dtype == np.float32
    np.testing.assert_array_equal(fd[known], dep[known])
    assert np.allclose(fd[20:40, 1000:1100], 0.25, atol=1e-3)
    # nothing to fill
    np.testing.assert_array_equal(warp.push_pull_fill_fast(img, np.ones((h, w), bool)), img)


# --------------------------------------------------------------------------- independent oracles
def _plane_oracle(img: np.ndarray, K: Intrinsics, Z: float, pose: Pose, D: Intrinsics):
    """Ray-cast a fronto-parallel textured plane at depth ``Z`` (independent of warp.py).

    Returns ``(expected_rgb, inside, reachable)``: ``inside`` = the destination ray hits the
    plane well inside the source image, ``reachable`` = it hits within one pixel of it.
    """
    u, v = np.meshgrid(np.arange(D.width, dtype=np.float64), np.arange(D.height, dtype=np.float64))
    rd = np.stack([(u - D.cx) / D.fx, (v - D.cy) / D.fy, np.ones_like(u)], -1)
    d = rd @ np.asarray(pose.R, np.float64).T  # destination rays in the source frame
    t = np.asarray(pose.t, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (Z - t[2]) / d[..., 2]
    ok = np.isfinite(lam) & (lam > 0.05)
    P = t + np.where(ok, lam, 0.0)[..., None] * d
    us = K.fx * P[..., 0] / Z + K.cx
    vs = K.fy * P[..., 1] / Z + K.cy
    h, w = img.shape[:2]
    inside = ok & (us >= 0.5) & (us <= w - 1.5) & (vs >= 0.5) & (vs <= h - 1.5)
    reach = ok & (us > -1.0) & (us < w) & (vs > -1.0) & (vs < h)
    m = lambda a: np.where(ok, a, -10.0).astype(np.float32)  # noqa: E731
    ref = cv2.remap(img, m(us), m(vs), cv2.INTER_LINEAR)
    return ref, inside, reach


@pytest.mark.parametrize(
    "ypr,t",
    [
        ((0.0, 0.0, 0.0), (0.07, -0.03, 0.0)),
        ((45.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((30.0, 15.0, 0.0), (0.0, 0.0, 0.0)),  # unpreconditioned steps fail from here on
        ((0.0, 0.0, 90.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 170.0), (0.0, 0.0, 0.0)),
        ((20.0, -10.0, 35.0), (0.0, 0.0, 0.0)),
        ((10.0, 3.0, 1.0), (0.05, 0.02, 0.1)),  # solutions at the source border
        ((15.0, 5.0, -20.0), (0.1, -0.05, 0.2)),
        ((40.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
    ],
)
def test_arbitrary_pose_matches_ray_cast_plane(ypr, t):
    """Golden: any pose / anisotropic K / different dst_K equals an analytic ray cast."""
    h, w = 150, 200
    K = Intrinsics(165.0, 150.0, 98.6, 75.6, w, h)  # anisotropic, off-centre
    D = Intrinsics(125.0, 135.0, 80.0, 70.0, 165, 135)
    img = _smooth_texture(h, w, 11, sigma=2.0)
    Z = 3.0
    pose = Pose.look(*ypr, t=t)
    rr = render_view(_single(img, 1.0 / Z, K), pose, D)
    ref, inside, reach = _plane_oracle(img, K, Z, pose, D)
    assert inside.mean() > 0.2
    missing = inside & (rr.alpha < 0.5)
    assert missing.sum() <= 0.001 * inside.sum(), f"{missing.sum()} visible pixels lost"
    assert not ((rr.alpha > 0.5) & ~reach).any(), "coverage where the plane is not visible"
    cov = inside & (rr.alpha > 0.99)
    err = np.abs(rr.rgb.astype(np.int16) - ref.astype(np.int16))[cov]
    assert err.max() <= 3 and err.mean() < 0.2


def _two_plane_oracle(s, pose: Pose, D: Intrinsics):
    """Ray-cast the two_plane_scene (square in front of the far plane), nearest hit wins."""
    h, w = s.img.shape[:2]
    K = s.K
    u, v = np.meshgrid(np.arange(D.width, dtype=np.float64), np.arange(D.height, dtype=np.float64))
    rd = np.stack([(u - D.cx) / D.fx, (v - D.cy) / D.fy, np.ones_like(u)], -1)
    d = rd @ np.asarray(pose.R, np.float64).T
    t = np.asarray(pose.t, np.float64)
    out = np.zeros(u.shape + (3,))
    valid = np.zeros(u.shape, bool)
    edge = np.zeros(u.shape, bool)
    best = np.full(u.shape, np.inf)
    for Z, tex, square in ((s.z_far, s.far_rgb, False), (s.z_near, s.img, True)):
        lam = (Z - t[2]) / d[..., 2]
        P = t + lam[..., None] * d
        us = K.fx * P[..., 0] / Z + K.cx
        vs = K.fy * P[..., 1] / Z + K.cy
        if square:  # the square covers source pixels 120..199 x 80..159
            hit = (lam > 0) & (us >= 119.5) & (us <= 199.5) & (vs >= 79.5) & (vs <= 159.5)
            core = (us >= 121) & (us <= 198) & (vs >= 81) & (vs <= 158)
            ring = (lam > 0) & (us >= 117.5) & (us <= 201.5) & (vs >= 77.5) & (vs <= 161.5)
            edge |= ring & ~core  # silhouette pixels are legitimately mixed
        else:
            hit = (lam > 0) & (us >= 0.5) & (us <= w - 1.5) & (vs >= 0.5) & (vs <= h - 1.5)
        smp = cv2.remap(tex, us.astype(np.float32), vs.astype(np.float32), cv2.INTER_LINEAR)
        sel = hit & (lam < best)
        out[sel] = smp[sel]
        best[sel] = lam[sel]
        valid |= hit
    return out, valid & ~edge


@pytest.mark.parametrize(
    "pose",
    [
        Pose.translate(0.05),
        Pose.translate(-0.08, 0.03, 0.0),
        Pose.look(15.0, 0.0, 0.0, (0.05, 0.0, 0.0)),
        Pose.look(-10.0, 8.0, 20.0, (-0.06, 0.02, 0.1)),
        Pose.look(30.0, 0.0, 0.0, (0.09, 0.0, 0.05)),
    ],
)
def test_layered_look_behind_matches_ray_cast_two_planes(two_plane_scene, psnr_fn, pose):
    """Golden: band layers (true background as 'inpainting') == ray-cast scene, any pose.

    Exercises occlusion (near seed), tears, look-behind and the rotation path together.
    """
    from vrphoto.geometry.layers import build_layers

    s = two_plane_scene

    class TrueBackground:
        def inpaint(self, img, mask):
            return s.far_rgb.copy()

    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=TrueBackground())
    rr = render_view(li, pose)
    ref, cmp = _two_plane_oracle(s, pose, s.K)
    assert cmp.mean() > 0.4
    assert int(np.count_nonzero(cmp & (rr.alpha < 0.5))) == 0, "holes inside the visible scene"
    good = cmp & (rr.alpha > 0.99)
    assert psnr_fn(rr.rgb, ref, good) > 45.0
    bad = np.abs(rr.rgb.astype(np.float64) - ref).max(-1)[good] > 30
    assert int(bad.sum()) <= 3


@pytest.mark.parametrize("tx", [0.05, -0.05])
def test_look_behind_whole_image_hole_fraction_with_overscan(two_plane_scene, tx):
    """Golden (literal): hole fraction < 0.1 % over the WHOLE frame when rendering with
    the 4 % video overscan (every output pixel then has source content)."""
    from vrphoto.geometry.layers import build_layers

    s = two_plane_scene

    class TrueBackground:
        def inpaint(self, img, mask):
            return s.far_rgb.copy()

    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.05, inpainter=TrueBackground())
    K = s.K
    D = Intrinsics(K.fx * 1.04, K.fy * 1.04, K.cx, K.cy, K.width, K.height)
    rr = render_view(li, Pose.translate(tx, 0.0, 0.0), D)
    assert hole_fraction(rr) < 0.001


def test_rotation_preconditioner_math():
    """``_Precond``: exact infinity solution + inverse Jacobian (finite differences)."""
    K = Intrinsics(300.0, 280.0, 161.0, 118.0, 320, 240)
    A = np.empty((3, 4, 2), np.float32)
    A[..., 0] = np.array([10.0, 100.0, 200.0, 300.0])[None, :]
    A[..., 1] = np.array([20.0, 120.0, 220.0])[:, None]
    ident = warp._Precond(A, warp._Motion.make(K, Pose.identity()))
    np.testing.assert_allclose(ident.p0, A, atol=1e-4)
    np.testing.assert_allclose(ident.minv.reshape(4, -1).T, np.tile([1, 0, 0, 1], (12, 1)), atol=1e-6)

    pose = Pose.look(25.0, -12.0, 30.0)
    mot = warp._Motion.make(K, pose)
    pre = warp._Precond(A, mot)

    def q_inf(p: np.ndarray) -> np.ndarray:  # rotation-only map, source-equivalent px
        return mot.project_px(p.astype(np.float32), np.zeros(len(p), np.float32)).astype(np.float64)

    p0 = pre.p0.reshape(-1, 2).astype(np.float64)
    np.testing.assert_allclose(q_inf(p0), A.reshape(-1, 2), atol=2e-3)
    eps = 0.5
    jx = (q_inf(p0 + [eps, 0]) - q_inf(p0 - [eps, 0])) / (2 * eps)
    jy = (q_inf(p0 + [0, eps]) - q_inf(p0 - [0, eps])) / (2 * eps)
    J = np.stack([jx, jy], -1)  # N x 2 x 2, columns = d/dpx, d/dpy
    minv = pre.minv.reshape(4, -1).T.reshape(-1, 2, 2)
    np.testing.assert_allclose(minv @ J, np.broadcast_to(np.eye(2), J.shape), atol=2e-3)


def test_nan_alpha_is_transparent_without_warnings():
    import warnings

    h, w = 40, 60
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = np.full((h, w, 3), 100, np.uint8)
    a = np.ones((h, w), np.float32)
    a[5:10, 5:10] = np.nan
    a[20:25, 20:25] = 3.0  # > 1 -> opaque
    a[30:35, 30:35] = -1.0  # < 0 -> transparent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rr = render_view(_single(img, 0.5, K, alpha=a), Pose.identity())
    assert rr.alpha[7, 7] == 0.0 and (rr.rgb[7, 7] == 0).all()
    assert rr.alpha[22, 22] == pytest.approx(1.0)
    assert rr.alpha[32, 32] == 0.0


def test_render_does_not_mutate_inputs(two_plane_scene):
    s = two_plane_scene
    inv = s.inv_z.copy()
    inv[0, 0] = np.nan
    alpha = np.ones(inv.shape, np.float32)
    alpha[0, 1] = 0.3
    li = _single(s.img.copy(), inv.copy(), s.K, alpha=alpha.copy())
    render_view(li, Pose.look(5.0, 0.0, 0.0, (0.03, 0.0, 0.0)))
    np.testing.assert_array_equal(li.layers[0].rgb, s.img)
    np.testing.assert_array_equal(li.layers[0].inv_z, inv)
    np.testing.assert_array_equal(li.layers[0].alpha, alpha)


def test_cancel_mid_render_in_parallel_strips():
    """Cancelling while strips run raises Cancelled, leaves no work behind, and the
    shared pool keeps working afterwards."""

    class CancelAfter(CancelToken):
        def __init__(self, n: int):
            super().__init__()
            self.n = n
            self.calls = 0

        def check(self) -> None:
            self.calls += 1
            if self.calls > self.n:
                self.cancel()
            super().check()

    h, w = 520, 640  # above the parallel threshold
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = _smooth_texture(h, w, 6)
    li = _single(img, 0.4, K)
    tok = CancelAfter(3)
    with pytest.raises(Cancelled):
        render_view(li, Pose.translate(0.02), cancel=tok)
    rr = render_view(li, Pose.translate(0.02))
    assert hole_fraction(rr) < 0.05


def test_push_pull_fast_close_to_exact_fill():
    from vrphoto.core.fill import push_pull_fill

    h, w = 300, 1500  # long side > 1024 -> pooled path
    img = _smooth_texture(h, w, 8, sigma=6.0)
    known = np.ones((h, w), bool)
    known[100:140, 300:700] = False
    known[:, 1000:1004] = False  # thin tear
    fast = warp.push_pull_fill_fast(img, known)
    exact = push_pull_fill(img, known)
    np.testing.assert_array_equal(fast[known], img[known])
    d = fast[~known].astype(np.float64) - exact[~known]
    assert 10 * math.log10(255.0**2 / max(1e-9, float((d**2).mean()))) > 35.0
