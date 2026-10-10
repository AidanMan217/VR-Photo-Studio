"""Tests for geometry.stereo: convergence, stereo pairs, layouts, wiggle (algorithm B)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, Intrinsics, Layer, LayeredImage, Pose, StereoParams
from vrphoto.geometry.layers import build_layers
from vrphoto.geometry.stereo import (
    DUBOIS_LEFT,
    LAYOUTS,
    auto_convergence,
    compose,
    eye_views,
    layout_size,
    resolve_stereo,
    stereo_pair,
    visible_inv_z,
    wiggle_frames,
)
from vrphoto.geometry.warp import render_filled

Z_NEAR, Z_FAR = 1.5, 4.0
NEAR_C = (110.0, 120.0)
FAR_C = (240.0, 110.0)


# --------------------------------------------------------------------------- helpers
def _blob_scene() -> LayeredImage:
    """Black background at Z_FAR with one blob, plus a near square (Z_NEAR) holding a blob."""
    h, w = 240, 320
    K = Intrinsics.from_hfov(w, h, 60.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    g = np.zeros((h, w))
    for cx, cy in (NEAR_C, FAR_C):
        g += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2))
    img = np.repeat(np.rint(np.clip(g * 250, 0, 255)).astype(np.uint8)[..., None], 3, axis=2)
    inv = np.full((h, w), 1.0 / Z_FAR, np.float32)
    inv[90:150, 80:140] = 1.0 / Z_NEAR
    return LayeredImage([Layer(img, inv, np.ones((h, w), np.float32))], K)


def _centroid_x(img: np.ndarray, cx: float, cy: float, r: int = 12) -> float:
    g = img[..., 0].astype(np.float64)
    x0, y0 = int(round(cx)), int(round(cy))
    patch = g[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
    xx = np.arange(x0 - r, x0 + r + 1)[None, :]
    return float((patch * xx).sum() / patch.sum())


def _locate(img: np.ndarray, tmpl: np.ndarray) -> float:
    """Sub-pixel x position of ``tmpl`` in ``img`` (template matching + parabola fit)."""
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, (x, y) = cv2.minMaxLoc(res)
    if 0 < x < res.shape[1] - 1:
        a, b, c = res[y, x - 1], res[y, x], res[y, x + 1]
        den = a - 2 * b + c
        return x + (0.5 * (a - c) / den if den != 0 else 0.0)
    return float(x)


# --------------------------------------------------------------------------- convergence
def test_auto_convergence_percentile():
    assert auto_convergence(np.full((10, 10), 0.25, np.float32)) == pytest.approx(4.0)
    z = np.linspace(1.0, 10.0, 1001)
    inv = (1.0 / z).astype(np.float32).reshape(7, 143)
    assert auto_convergence(inv) == pytest.approx(np.percentile(z, 35.0), rel=1e-3)
    assert auto_convergence(inv, pct=50.0) == pytest.approx(np.percentile(z, 50.0), rel=1e-3)
    # 35 % of pixels are nearer than the result
    zc = auto_convergence(inv)
    assert np.mean(z < zc) == pytest.approx(0.35, abs=0.01)


def test_auto_convergence_ignores_invalid_and_rejects_empty():
    inv = np.full((20, 20), 0.5, np.float32)
    inv[:5] = np.nan
    inv[5:8] = -1
    assert auto_convergence(inv) == pytest.approx(2.0)
    with pytest.raises(InputError):
        auto_convergence(np.full((4, 4), np.nan, np.float32))
    with pytest.raises(InputError):
        auto_convergence(inv, pct=120.0)


def test_visible_inv_z_uses_front_layers():
    h, w = 10, 12
    K = Intrinsics.from_hfov(w, h, 60.0)
    back = Layer(np.zeros((h, w, 3), np.uint8), np.full((h, w), 0.1, np.float32), np.ones((h, w), np.float32))
    a = np.zeros((h, w), np.float32)
    a[:, 6:] = 1.0
    front = Layer(np.zeros((h, w, 3), np.uint8), np.full((h, w), 0.7, np.float32), a)
    v = visible_inv_z(LayeredImage([back, front], K))
    assert (v[:, :6] == np.float32(0.1)).all() and (v[:, 6:] == np.float32(0.7)).all()


# --------------------------------------------------------------------------- stereo pairs
@pytest.mark.parametrize("zc", [Z_FAR, 1.0, 2.5])
def test_flat_parallax_matches_formula(zc):
    li = _blob_scene()
    f = li.K.fx
    p = StereoParams(ipd_m=0.063, strength=1.0, convergence_m=zc, max_parallax_pct=0.0)
    left, right = stereo_pair(li, p)
    B = 0.063
    for (cx, cy), Z in ((NEAR_C, Z_NEAR), (FAR_C, Z_FAR)):
        par = f * B * (1.0 / Z - 1.0 / zc)
        uL = _centroid_x(left, cx + par / 2, cy)
        uR = _centroid_x(right, cx - par / 2, cy)
        assert uL - uR == pytest.approx(par, abs=0.06)
        # symmetric about the source position (principal-point convergence)
        assert (uL + uR) / 2 == pytest.approx(cx, abs=0.06)


def test_square_crossed_when_convergence_behind_it(two_plane_scene):
    s = two_plane_scene
    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.063, inpainter=None, mode="band")
    tmpl = s.img[100:140, 140:180]  # core of the near square
    p_behind = StereoParams(convergence_m=3.0, max_parallax_pct=10.0)
    left, right = stereo_pair(li, p_behind)
    uL, uR = _locate(left, tmpl), _locate(right, tmpl)
    assert uL > uR, "square in front of the convergence plane must have crossed parallax"
    assert uL - uR == pytest.approx(s.K.fx * 0.063 * (1 / s.z_near - 1 / 3.0), abs=0.3)
    p_front = StereoParams(convergence_m=1.0, max_parallax_pct=10.0)
    left, right = stereo_pair(li, p_front)
    uL, uR = _locate(left, tmpl), _locate(right, tmpl)
    assert uL < uR, "square behind the convergence plane must have uncrossed parallax"
    # the background at the convergence distance has zero parallax
    p_bg = StereoParams(convergence_m=s.z_far, max_parallax_pct=10.0)
    left, right = stereo_pair(li, p_bg)
    bg_t = s.img[20:60, 20:60]
    assert _locate(left, bg_t) - _locate(right, bg_t) == pytest.approx(0.0, abs=0.15)


def test_parallax_clamp_respected():
    li = _blob_scene()
    f, w = li.K.fx, li.K.width
    p = StereoParams(ipd_m=0.063, strength=2.0, convergence_m=Z_FAR, max_parallax_pct=1.0)
    B, zc = resolve_stereo(li, p)
    limit = 0.01 * w
    assert zc == Z_FAR
    assert B < 0.063 * 2.0
    assert f * B * (1 / Z_NEAR - 1 / Z_FAR) == pytest.approx(limit, rel=1e-4)
    left, right = stereo_pair(li, p)
    cx, cy = NEAR_C
    par = _centroid_x(left, cx + limit / 2, cy) - _centroid_x(right, cx - limit / 2, cy)
    assert abs(par) <= limit + 0.06
    # a generous limit leaves the baseline untouched
    B2, _ = resolve_stereo(li, StereoParams(ipd_m=0.063, convergence_m=Z_FAR, max_parallax_pct=50.0))
    assert B2 == pytest.approx(0.063)
    # the clamp is measured in output pixels (dst_K)
    B3, _ = resolve_stereo(li, p, li.K.scaled(2.0))
    assert B3 == pytest.approx(B, rel=1e-6)


def test_auto_convergence_used_when_none():
    li = _blob_scene()
    B, zc = resolve_stereo(li, StereoParams(convergence_m=None, max_parallax_pct=0.0))
    assert zc == pytest.approx(auto_convergence(li.front.inv_z))
    assert zc == pytest.approx(Z_FAR)
    assert B == pytest.approx(0.063)


def test_vr_mode_parallel_eyes():
    li = _blob_scene()
    f = li.K.fx
    p = StereoParams(ipd_m=0.064, strength=3.0, convergence_m=1.0, mode="vr", max_parallax_pct=0.5)
    (pl, kl), (pr, kr) = eye_views(li, p)
    assert kl == li.K and kr == li.K
    np.testing.assert_allclose(pl.t, [-0.032, 0, 0])
    np.testing.assert_allclose(pr.t, [0.032, 0, 0])
    left, right = stereo_pair(li, p)
    for (cx, cy), Z in ((NEAR_C, Z_NEAR), (FAR_C, Z_FAR)):
        par = f * 0.064 / Z
        uL = _centroid_x(left, cx + par / 2, cy)
        uR = _centroid_x(right, cx - par / 2, cy)
        assert uL - uR == pytest.approx(par, abs=0.06)


def test_swap_eyes_and_dst_K():
    li = _blob_scene()
    p = StereoParams(convergence_m=2.0)
    left, right = stereo_pair(li, p)
    sl, sr = stereo_pair(li, StereoParams(convergence_m=2.0, swap_eyes=True))
    np.testing.assert_array_equal(sl, right)
    np.testing.assert_array_equal(sr, left)
    K2 = li.K.scaled(0.5)
    l2, r2 = stereo_pair(li, p, K2)
    assert l2.shape == (K2.height, K2.width, 3) and r2.shape == l2.shape


def test_eye_views_principal_points_and_center_pose():
    li = _blob_scene()
    K = li.K
    p = StereoParams(ipd_m=0.06, convergence_m=3.0, max_parallax_pct=0.0)
    (pl, kl), (pr, kr) = eye_views(li, p)
    shift = K.fx * 0.06 / (2 * 3.0)
    assert kl.cx == pytest.approx(K.cx - shift) and kr.cx == pytest.approx(K.cx + shift)
    assert kl.cy == K.cy and kl.fx == K.fx
    center = Pose.look(yaw_deg=90.0, t=(1.0, 2.0, 3.0))
    (pl, _), (pr, _) = eye_views(li, p, center=center)
    # yaw 90 deg: camera x axis points along source -z
    np.testing.assert_allclose(pl.t, [1.0, 2.0, 3.0 + 0.03], atol=1e-9)
    np.testing.assert_allclose(pr.t, [1.0, 2.0, 3.0 - 0.03], atol=1e-9)
    np.testing.assert_allclose(pl.R, center.R)


def test_stereo_param_validation():
    li = _blob_scene()
    with pytest.raises(InputError):
        resolve_stereo(li, StereoParams(ipd_m=0.0))
    with pytest.raises(InputError):
        resolve_stereo(li, StereoParams(convergence_m=-1.0))
    with pytest.raises(InputError):
        resolve_stereo(li, StereoParams(strength=-1.0))
    with pytest.raises(InputError):
        resolve_stereo(li, StereoParams(mode="weird"))  # type: ignore[arg-type]


def test_stereo_pair_cancel_and_progress():
    li = _blob_scene()
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        stereo_pair(li, StereoParams(), cancel=tok)
    seen = []
    stereo_pair(li, StereoParams(), progress=lambda f, m: seen.append(f))
    assert seen[-1] == 1.0


def test_zero_strength_gives_identical_eyes():
    li = _blob_scene()
    left, right = stereo_pair(li, StereoParams(strength=0.0))
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(left, li.front.rgb)


# --------------------------------------------------------------------------- layouts
@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (61, 81, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (61, 81, 3), dtype=np.uint8)
    return left, right


def test_compose_shapes_and_content(pair):
    left, right = pair
    h, w = left.shape[:2]
    sbs = compose(left, right, "sbs")
    assert sbs.shape == (h, 2 * w, 3)
    np.testing.assert_array_equal(sbs[:, :w], left)
    np.testing.assert_array_equal(sbs[:, w:], right)
    cross = compose(left, right, "cross")
    np.testing.assert_array_equal(cross[:, :w], right)
    np.testing.assert_array_equal(cross[:, w:], left)
    tb = compose(left, right, "tb")
    assert tb.shape == (2 * h, w, 3)
    np.testing.assert_array_equal(tb[:h], left)
    np.testing.assert_array_equal(tb[h:], right)
    assert compose(left, right, "half_sbs").shape == (h, 2 * (w // 2), 3)
    assert compose(left, right, "half_tb").shape == (2 * (h // 2), w, 3)
    np.testing.assert_array_equal(compose(left, right, "left"), left)
    np.testing.assert_array_equal(compose(left, right, "right"), right)
    for layout in LAYOUTS:
        out = compose(left, right, layout)
        assert out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
        assert (out.shape[1], out.shape[0]) == layout_size(w, h, layout)


def test_half_layouts_keep_eye_content():
    left = np.zeros((40, 64, 3), np.uint8)
    right = np.full((40, 64, 3), 200, np.uint8)
    hs = compose(left, right, "half_sbs")
    assert (hs[:, :32] == 0).all() and (hs[:, 32:] == 200).all()
    ht = compose(left, right, "half_tb")
    assert (ht[:20] == 0).all() and (ht[20:] == 200).all()


def test_anaglyph_channel_logic():
    white = np.full((8, 8, 3), 255, np.uint8)
    black = np.zeros((8, 8, 3), np.uint8)
    a = compose(white, black, "anaglyph")  # only the left eye lit
    assert a[0, 0, 0] > 200 and a[0, 0, 1] == 0 and a[0, 0, 2] == 0
    b = compose(black, white, "anaglyph")  # only the right eye lit
    assert b[0, 0, 0] == 0 and b[0, 0, 1] > 200 and b[0, 0, 2] > 200
    # Dubois: red row of the left matrix sums to 1.132 -> saturates
    assert DUBOIS_LEFT[0].sum() > 1.0
    # pure red in the left eye -> red; pure blue in the right eye -> blue
    red = black.copy()
    red[..., 0] = 255
    blue = black.copy()
    blue[..., 2] = 255
    c = compose(red, blue, "anaglyph")
    assert c[0, 0, 0] > 150 and c[0, 0, 2] > 200 and c[0, 0, 1] == 0
    # mid grey in both eyes stays roughly neutral-ish (green/blue from right, red from left)
    g = np.full((8, 8, 3), 128, np.uint8)
    d = compose(g, g, "anaglyph")
    assert np.all(d[0, 0] > 60) and np.all(d[0, 0] < 200)


def test_anaglyph_gray_logic(pair):
    left, right = pair
    out = compose(left, right, "anaglyph_gray")
    luma = np.array([0.299, 0.587, 0.114])
    np.testing.assert_allclose(out[..., 0], np.rint(left @ luma), atol=1)
    np.testing.assert_allclose(out[..., 1], np.rint(right @ luma), atol=1)
    np.testing.assert_array_equal(out[..., 1], out[..., 2])


def test_compose_validation(pair):
    left, right = pair
    with pytest.raises(InputError):
        compose(left, right, "diagonal")
    with pytest.raises(InputError):
        compose(left, right[:10], "sbs")
    with pytest.raises(InputError):
        compose(left[..., 0], right[..., 0], "sbs")
    with pytest.raises(InputError):
        layout_size(10, 10, "nope")


# --------------------------------------------------------------------------- wiggle
def test_wiggle_frames_path_and_pivot():
    li = _blob_scene()
    p = StereoParams(convergence_m=Z_FAR, max_parallax_pct=0.0)
    frames = wiggle_frames(li, p, n=8)
    assert len(frames) == 8 and all(fr.shape == li.front.rgb.shape for fr in frames)
    left, right = stereo_pair(li, p)
    np.testing.assert_array_equal(frames[0], left)
    np.testing.assert_array_equal(frames[4], right)
    # the convergence plane (far blob) stays still; the near blob rocks
    cxf, cyf = FAR_C
    xs_far = [_centroid_x(fr, cxf, cyf) for fr in frames]
    assert max(xs_far) - min(xs_far) < 0.06
    cxn, cyn = NEAR_C
    par = li.K.fx * 0.063 * (1 / Z_NEAR - 1 / Z_FAR)
    xs_near = [_centroid_x(fr, cxn, cyn, r=14) for fr in frames]
    assert max(xs_near) - min(xs_near) == pytest.approx(par, abs=0.1)
    # loop: frame i and n-i are mirror positions -> identical renders
    np.testing.assert_array_equal(frames[1], frames[7])


def test_wiggle_vr_params_use_flat_pivot_and_validation():
    li = _blob_scene()
    frames = wiggle_frames(li, StereoParams(mode="vr", convergence_m=Z_FAR, max_parallax_pct=0.0), n=2)
    assert len(frames) == 2
    xs = [_centroid_x(fr, *FAR_C) for fr in frames]
    assert abs(xs[0] - xs[1]) < 0.06
    with pytest.raises(InputError):
        wiggle_frames(li, StereoParams(), n=1)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        wiggle_frames(li, StereoParams(), n=4, cancel=tok)


def test_eye_render_equals_manual_render_filled():
    li = _blob_scene()
    p = StereoParams(convergence_m=2.0, max_parallax_pct=0.0)
    left, _ = stereo_pair(li, p)
    K = li.K
    kl = K.with_principal(K.cx - K.fx * 0.063 / (2 * 2.0), K.cy)
    np.testing.assert_array_equal(left, render_filled(li, Pose.translate(-0.0315), kl))
    assert math.isfinite(resolve_stereo(li, p)[1])


# --------------------------------------------------------------------------- review additions
def test_anaglyph_matches_dubois_in_linear_light():
    """Independent re-computation: sRGB -> linear, Dubois matrices, -> sRGB."""

    def to_lin(c):
        c = c / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def to_srgb(x):
        x = np.clip(x, 0.0, 1.0)
        return 255.0 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)

    ML = np.array([[0.456, 0.500, 0.176], [-0.040, -0.038, -0.016], [-0.015, -0.021, -0.005]])
    MR = np.array([[-0.043, -0.088, -0.002], [0.378, 0.734, -0.018], [-0.072, -0.113, 1.226]])
    rng = np.random.default_rng(5)
    left = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    right = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    exp = to_srgb(to_lin(left.astype(np.float64)) @ ML.T + to_lin(right.astype(np.float64)) @ MR.T)
    out = compose(left, right, "anaglyph")
    assert np.abs(out.astype(np.float64) - exp).max() <= 1.0


def test_layout_names_are_case_insensitive(pair):
    left, right = pair
    np.testing.assert_array_equal(compose(left, right, " SBS "), compose(left, right, "sbs"))
    assert layout_size(10, 8, "Half_TB") == (10, 8)


def test_parallax_clamp_counts_uncrossed_side():
    """Convergence in front of everything: the far content has the largest |parallax|."""
    li = _blob_scene()
    f, w = li.K.fx, li.K.width
    p = StereoParams(ipd_m=0.063, convergence_m=1.0, max_parallax_pct=1.0)
    B, zc = resolve_stereo(li, p)
    worst = f * B * max(abs(1 / Z_NEAR - 1 / zc), abs(1 / Z_FAR - 1 / zc))
    assert worst == pytest.approx(0.01 * w, rel=1e-4)
    assert f * B * abs(1 / Z_FAR - 1 / zc) == pytest.approx(0.01 * w, rel=1e-4)


def test_stereo_on_three_layer_cleanplate_uses_visible_depth():
    """visible_inv_z / auto convergence see the plate, not the band background."""
    h, w = 120, 160
    K = Intrinsics.from_hfov(w, h, 60.0)
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    inv = np.full((h, w), 0.25, np.float32)
    inv[80:, :60] = 1 / 1.5  # table
    alpha = np.zeros((h, w), np.float32)
    alpha[20:100, 100:130] = 1.0
    inv[alpha > 0.5] = 1 / 1.2
    li = build_layers(img, inv, K, max_baseline_m=0.1, inpainter=None, alpha=alpha, mode="cleanplate")
    assert len(li.layers) == 3
    np.testing.assert_allclose(visible_inv_z(li)[alpha > 0.5], 1 / 1.2, rtol=1e-6)
    v = visible_inv_z(li)
    np.testing.assert_allclose(v[100:, 10:50], 1 / 1.5, rtol=1e-6)  # table, not the band depth
    left, right = stereo_pair(li, StereoParams(convergence_m=4.0, max_parallax_pct=0.0))
    assert left.shape == img.shape and right.shape == img.shape
