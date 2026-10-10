"""Tests for geometry.projection: equirect / perspective / cubemap mappings, VR180 and 360."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from vrphoto.core.errors import InputError
from vrphoto.core.types import Intrinsics
from vrphoto.geometry import projection as P

RED = (255, 0, 0)


# --------------------------------------------------------------------------- helpers
def red_centroid(img: np.ndarray) -> tuple[float, float, int]:
    m = (img[..., 0] > 200) & (img[..., 1] < 70) & (img[..., 2] < 70)
    ys, xs = np.nonzero(m)
    assert m.sum() > 0, "marker not found"
    return float(xs.mean()), float(ys.mean()), int(m.sum())


def dir_image(w: int, h: int) -> np.ndarray:
    """Equirect whose pixels store their own unit view direction (float32 HxWx3)."""
    return P.equirect_dirs(w, h)


def lonlat_px_error(dirs_est: np.ndarray, w: int, h: int, sel: np.ndarray) -> float:
    """Max angular error (in equator pixels) between decoded and ideal directions."""
    ideal = P.equirect_dirs(w, h).astype(np.float64)[sel]
    est = dirs_est.astype(np.float64)[sel]
    est /= np.linalg.norm(est, axis=1, keepdims=True)
    ang = np.arctan2(np.linalg.norm(np.cross(est, ideal), axis=1), (est * ideal).sum(1))
    return float(ang.max() / (math.pi / h))


def lon_lat(d: np.ndarray) -> tuple[float, float]:
    x, y, z = (float(v) for v in d)
    return math.degrees(math.atan2(x, z)), math.degrees(math.atan2(-y, math.hypot(x, z)))


# --------------------------------------------------------------------------- equirect_dirs
def test_equirect_dirs_formula():
    w, h = 64, 32
    d = P.equirect_dirs(w, h)
    assert d.shape == (h, w, 3) and d.dtype == np.float32
    assert np.allclose(np.linalg.norm(d, axis=2), 1.0, atol=1e-5)
    for i, j in [(0, 0), (31, 15), (32, 16), (63, 31), (10, 25)]:
        lam = math.radians(-180 + (i + 0.5) / w * 360)
        phi = math.radians(90 - (j + 0.5) / h * 180)
        exp = (math.cos(phi) * math.sin(lam), -math.sin(phi), math.cos(phi) * math.cos(lam))
        assert np.allclose(d[j, i], exp, atol=1e-6)
    # top rows look up (y < 0 in the y-down cv frame), right half looks right (x > 0)
    assert (d[0, :, 1] < 0).all() and (d[-1, :, 1] > 0).all()
    assert (d[:, w // 2 :, 0] > 0).all() and (d[:, : w // 2, 0] < 0).all()


def test_equirect_dirs_partial_ranges():
    d = P.equirect_dirs(90, 45, lon_range=(-90, 90), lat_range=(-45, 45))
    lam0, phi0 = lon_lat(d[0, 0])
    assert lam0 == pytest.approx(-90 + 1.0, abs=1e-4)
    assert phi0 == pytest.approx(45 - 1.0, abs=1e-4)
    with pytest.raises(InputError):
        P.equirect_dirs(10, 10, lon_range=(10, -10))
    with pytest.raises(InputError):
        P.equirect_dirs(0, 10)


# --------------------------------------------------------------------------- equirect <-> perspective
@pytest.mark.parametrize("yaw,pitch", [(0, 0), (30, 10), (-120, -40), (175, 0), (0, 80)])
def test_equirect_persp_equirect_roundtrip(yaw, pitch):
    w, h = 1024, 512
    eq = dir_image(w, h)
    K = Intrinsics(300.0, 300.0, 319.5, 239.5, 640, 480)
    view = P.equirect_to_persp(eq, K, yaw_deg=yaw, pitch_deg=pitch)
    assert view.shape == (480, 640, 3) and view.dtype == np.float32
    back, valid = P.persp_to_equirect(view, K, w, h, lon_range=(-180, 180), yaw_deg=yaw, pitch_deg=pitch)
    assert back.shape == (h, w, 3) and valid.dtype == bool
    import cv2

    inner = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert inner.sum() > 20000
    assert lonlat_px_error(back, w, h, inner) < 0.5


def test_persp_equirect_persp_roundtrip():
    # perspective image storing its own pixel coordinates
    pw, ph = 320, 240
    K = Intrinsics.from_hfov(pw, ph, 70.0)
    uu, vv = np.meshgrid(np.arange(pw, dtype=np.float32), np.arange(ph, dtype=np.float32))
    coords = np.stack([uu, vv, np.zeros_like(uu)], -1)
    w = 2048
    eq, valid = P.persp_to_equirect(coords, K, w, w // 2, lon_range=(-180, 180), yaw_deg=20, pitch_deg=-15)
    back = P.equirect_to_persp(eq, K, yaw_deg=20, pitch_deg=-15)
    err = np.hypot(back[..., 0] - uu, back[..., 1] - vv)[4:-4, 4:-4]
    assert err.max() < 0.5


def test_marker_and_rotation_directions(checker_equirect):
    K = Intrinsics(200.0, 200.0, 199.5, 149.5, 400, 300)
    v0 = P.equirect_to_persp(checker_equirect, K)
    x, y, _ = red_centroid(v0)
    assert x == pytest.approx(199.5, abs=0.5) and y == pytest.approx(149.5, abs=0.5)
    # camera turned right by 20 deg: the marker (straight ahead) appears LEFT of centre
    v1 = P.equirect_to_persp(checker_equirect, K, yaw_deg=20)
    x1, y1, _ = red_centroid(v1)
    assert x1 == pytest.approx(199.5 - 200 * math.tan(math.radians(20)), abs=0.75)
    assert y1 == pytest.approx(149.5, abs=0.75)
    # camera pitched up: the marker appears BELOW centre
    v2 = P.equirect_to_persp(checker_equirect, K, pitch_deg=15)
    x2, y2, _ = red_centroid(v2)
    assert x2 == pytest.approx(199.5, abs=0.75)
    assert y2 == pytest.approx(149.5 + 200 * math.tan(math.radians(15)), abs=0.75)


def test_persp_to_equirect_places_photo_with_yaw_pitch():
    img = np.full((120, 160, 3), 90, np.uint8)
    img[57:63, 77:83] = RED  # centred on the principal point (79.5, 59.5)
    K = Intrinsics(150.0, 150.0, 79.5, 59.5, 160, 120)
    w, h = 720, 360  # 2 px per degree
    eq, valid = P.persp_to_equirect(img, K, w, h, lon_range=(-180, 180), yaw_deg=30, pitch_deg=20)
    x, y, _ = red_centroid(eq)
    lam = -180 + (x + 0.5) / w * 360
    phi = 90 - (y + 0.5) / h * 180
    assert lam == pytest.approx(30, abs=0.6) and phi == pytest.approx(20, abs=0.6)
    assert valid[int(round(y)), int(round(x))]
    # outside the photo is black and invalid
    assert not valid[h // 2, 0] and (eq[~valid] == 0).all()


def test_persp_to_equirect_true_angular_size():
    w_img, h_img = 400, 300
    img = np.full((h_img, w_img, 3), 200, np.uint8)
    K = Intrinsics.from_hfov(w_img, h_img, 65.0)
    n = 1800  # 10 px / deg over 180 deg
    eq, valid = P.persp_to_equirect(img, K, n, n)  # default VR180 lon range
    row = valid[n // 2]
    xs = np.nonzero(row)[0]
    assert xs.size == pytest.approx(65.0 / 180.0 * n, abs=2)
    # vertical extent on the central column = vfov
    col = valid[:, n // 2]
    assert col.sum() == pytest.approx(K.vfov_deg / 180.0 * n, abs=2)
    # nothing behind the camera
    eq360, valid360 = P.persp_to_equirect(img, K, 720, 360, lon_range=(-180, 180))
    assert not valid360[:, :180].any() and not valid360[:, -180:].any()


def test_persp_to_equirect_rescales_intrinsics_and_gray():
    img = np.full((60, 80), 77, np.uint8)  # gray image, K given for a 2x bigger version
    K = Intrinsics.from_hfov(160, 120, 60.0)
    eq, valid = P.persp_to_equirect(img, K, 360, 360)
    assert eq.shape == (360, 360)
    assert valid[:, 180].sum() == pytest.approx(K.vfov_deg / 180 * 360, abs=2)
    assert (eq[valid] == 77).all()


def test_equirect_to_persp_partial_range_is_black_outside():
    eq = np.full((200, 200, 3), 128, np.uint8)  # 180x180 coverage
    K = Intrinsics.from_hfov(200, 100, 90.0)
    view = P.equirect_to_persp(eq, K, yaw_deg=90, lon_range=(-90, 90))
    assert (view[:, :90] == 128).all()  # left half looks into the covered hemisphere
    assert (view[:, 110:] == 0).all()


def test_antialias_prefilter_avoids_moire():
    # a 1-px checker photo squeezed into a much smaller equirect must average to grey
    h, w = 1200, 1600
    yy, xx = np.mgrid[0:h, 0:w]
    img = (((xx + yy) % 2) * 255).astype(np.uint8)
    img = np.repeat(img[..., None], 3, axis=2)
    K = Intrinsics.from_hfov(w, h, 60.0)
    eq, valid = P.persp_to_equirect(img, K, 400, 400)
    vals = eq[valid].astype(np.float64)
    assert abs(vals.mean() - 127.5) < 3 and vals.std() < 12


def test_tiling_matches_untiled(monkeypatch, checker_equirect):
    K = Intrinsics.from_hfov(300, 200, 80.0)
    a = P.equirect_to_persp(checker_equirect, K, yaw_deg=10)
    monkeypatch.setattr(P, "_TILE_PX", 997)
    monkeypatch.setattr(P, "_TILE_W", 70)
    b = P.equirect_to_persp(checker_equirect, K, yaw_deg=10)
    assert np.array_equal(a, b)


def test_equirect_to_persp_wraps_longitude(checker_equirect):
    eq = checker_equirect.copy()
    w = eq.shape[1]
    eq[250:262, w - 6 :] = RED  # marker straddling the +-180 seam
    eq[250:262, :6] = RED
    K = Intrinsics(200.0, 200.0, 199.5, 149.5, 400, 300)
    view = P.equirect_to_persp(eq, K, yaw_deg=180)
    x, _, n = red_centroid(view)
    assert x == pytest.approx(199.5, abs=0.75) and n > 100


# --------------------------------------------------------------------------- cubemap
def test_cubemap_keys_orientation_and_marker(checker_equirect):
    faces = P.equirect_to_cubemap(checker_equirect, 256)
    assert list(faces) == ["+x", "-x", "+y", "-y", "+z", "-z"]
    for f in faces.values():
        assert f.shape == (256, 256, 3) and f.dtype == np.uint8
    x, y, _ = red_centroid(faces["+z"])
    assert x == pytest.approx(127.5, abs=0.5) and y == pytest.approx(127.5, abs=0.5)
    for k in ("+x", "-x", "+y", "-y", "-z"):
        m = (faces[k][..., 0] > 200) & (faces[k][..., 1] < 70)
        assert not m.any()


def test_cubemap_face_directions():
    d = dir_image(1024, 512)
    faces = P.equirect_to_cubemap(d, 64)
    axes = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
    for k, ax in axes.items():
        c = faces[k][31:33, 31:33].mean((0, 1))
        assert np.allclose(c / np.linalg.norm(c), ax, atol=0.02), k
        assert np.allclose(P.cube_face_rotation(k)[:, 2], ax)
    # documented orientation: side faces are upright (top row looks up, y < 0)
    for k in ("+x", "-x", "+z", "-z"):
        assert faces[k][0, 32, 1] < -0.5 and faces[k][-1, 32, 1] > 0.5
    # image right of the side faces
    assert faces["+z"][32, -1, 0] > 0.5
    assert faces["-z"][32, -1, 0] < -0.5
    assert faces["+x"][32, -1, 2] < -0.5
    assert faces["-x"][32, -1, 2] > 0.5
    # floor face: top edge touches the front; sky face: bottom edge touches the front
    assert faces["+y"][0, 32, 2] > 0.5 and faces["+y"][32, -1, 0] > 0.5
    assert faces["-y"][-1, 32, 2] > 0.5 and faces["-y"][32, -1, 0] > 0.5
    with pytest.raises(InputError):
        P.cube_face_rotation("+w")


def test_cubemap_roundtrip_subpixel():
    w, h = 1024, 512
    eq = dir_image(w, h)
    faces = P.equirect_to_cubemap(eq, 320)
    back = P.cubemap_to_equirect(faces, w, h)
    assert back.shape == (h, w, 3) and back.dtype == np.float32
    assert lonlat_px_error(back, w, h, np.ones((h, w), bool)) < 0.5


def test_cubemap_roundtrip_marker_and_image(checker_equirect, psnr_fn):
    faces = P.equirect_to_cubemap(checker_equirect, 384)
    back = P.cubemap_to_equirect(faces, 1024, 512)
    x, y, _ = red_centroid(back)
    assert x == pytest.approx(511.5, abs=0.5) and y == pytest.approx(255.5, abs=0.5)
    # away from the poles (where the equirect is massively oversampled) the image survives
    band = slice(512 // 6, 512 - 512 // 6)
    assert psnr_fn(back[band], checker_equirect[band]) > 24


def test_cubemap_seamless_smooth_field():
    # a smooth function of direction must stay smooth across cube edges
    w, h = 720, 360
    d = dir_image(w, h).astype(np.float64)
    field = (0.5 + 0.4 * d[..., 0] * d[..., 2] + 0.3 * d[..., 1]).astype(np.float32)
    faces = P.equirect_to_cubemap(field, 128)
    assert faces["+z"].shape == (128, 128)
    back = P.cubemap_to_equirect(faces, w, h)
    assert np.abs(back - field).max() < 2e-3


def test_cubemap_validation():
    faces = {k: np.zeros((8, 8, 3), np.uint8) for k in P.CUBE_FACES}
    del faces["-z"]
    with pytest.raises(InputError, match="missing"):
        P.cubemap_to_equirect(faces, 64, 32)
    faces["-z"] = np.zeros((8, 9, 3), np.uint8)
    with pytest.raises(InputError):
        P.cubemap_to_equirect(faces, 64, 32)
    with pytest.raises(InputError):
        P.equirect_to_cubemap(np.zeros((32, 64, 3), np.uint8), 0)


def test_push_pull_reexport():
    from vrphoto.core.fill import push_pull_fill

    assert P.push_pull_fill is push_pull_fill


# --------------------------------------------------------------------------- VR180
def _photo_with_marker(w=640, h=480):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 200 // w + 20, yy * 200 // h + 20, np.full_like(xx, 128)], -1).astype(np.uint8)
    cx, cy = w // 2, h // 2
    img[cy - 6 : cy + 6, cx - 6 : cx + 6] = RED
    K = Intrinsics(w / 2 / math.tan(math.radians(65 / 2)), w / 2 / math.tan(math.radians(65 / 2)),
                   cx - 0.5, cy - 0.5, w, h)
    return img, K


@pytest.mark.parametrize("layout", ["sbs", "tb"])
def test_vr180_layout_and_marker_centre(layout):
    img, K = _photo_with_marker()
    n = 720
    fr = P.vr180_frame(img, img, K, n, layout=layout)
    if layout == "sbs":
        assert fr.shape == (n, 2 * n, 3)
        eyes = [fr[:, :n], fr[:, n:]]
    else:
        assert fr.shape == (2 * n, n, 3)
        eyes = [fr[:n], fr[n:]]
    assert fr.dtype == np.uint8
    for e in eyes:
        x, y, _ = red_centroid(e)
        assert x == pytest.approx((n - 1) / 2, abs=1.0) and y == pytest.approx((n - 1) / 2, abs=1.0)


def test_vr180_true_angular_size_and_size_mult():
    img, K = _photo_with_marker()
    n = 900  # 5 px / deg
    fr = P.vr180_frame(img, img, K, n, fill="black")
    eye = fr[:, :n]
    span = np.count_nonzero(eye[n // 2].sum(1) > 0)
    assert span == pytest.approx(65.0 / 180.0 * n, abs=2)
    fr2 = P.vr180_frame(img, img, K, n, fill="black", size_mult=1.5)
    span2 = np.count_nonzero(fr2[n // 2, :n].sum(1) > 0)
    assert span2 == pytest.approx(97.5 / 180.0 * n, abs=2)
    fr3 = P.vr180_frame(img, img, K, n, fill="black", size_mult=0.5)
    span3 = np.count_nonzero(fr3[n // 2, :n].sum(1) > 0)
    assert span3 == pytest.approx(32.5 / 180.0 * n, abs=2)


def test_vr180_size_mult_clamped_to_150(caplog):
    img, K = _photo_with_marker()
    n = 900
    with caplog.at_level(logging.WARNING, logger="vrphoto"):
        fr = P.vr180_frame(img, img, K, n, fill="black", size_mult=5.0)
    assert any("clamped" in r.message for r in caplog.records)
    span = np.count_nonzero(fr[n // 2, :n].sum(1) > 0)
    assert span == pytest.approx(150.0 / 180.0 * n, abs=2)
    with pytest.raises(InputError):
        P.vr180_frame(img, img, K, n, size_mult=0)


def test_vr180_blur_fill_identical_in_both_eyes_and_darkened():
    img, K = _photo_with_marker()
    right = np.roll(img, 9, axis=1)  # different content in the right eye
    n = 600
    fr = P.vr180_frame(img, right, K, n, fill="blur")
    L, R = fr[:, :n].astype(int), fr[:, n:].astype(int)
    _, valid = P.persp_to_equirect(img, K, n, n)
    import cv2

    outside = cv2.erode((~valid).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert np.array_equal(L[outside], R[outside])  # the surround sits at infinity
    assert (L[valid] != R[valid]).any()
    # no black: fill continues the photo's colours, then darkens toward the frame edges
    assert L[outside].min() > 0
    ring = valid ^ cv2.dilate(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    near = L[ring].mean()
    corner = L[:20, :20].mean()
    assert corner < near
    # gaussian-blurred: no sharp structure outside the photo
    gx = np.abs(np.diff(L, axis=1)).sum(-1)[outside[:, :-1] & outside[:, 1:]]
    assert gx.max() <= 12


def test_vr180_fill_colour_and_image():
    img, K = _photo_with_marker()
    n = 256
    fr = P.vr180_frame(img, img, K, n, fill=(10, 20, 30))
    assert tuple(fr[0, 0]) == (10, 20, 30) and tuple(fr[0, n]) == (10, 20, 30)
    bg = np.zeros((64, 64, 3), np.uint8)
    bg[...] = (0, 0, 200)
    fr2 = P.vr180_frame(img, img, K, n, fill=bg)
    assert tuple(fr2[5, 5]) == (0, 0, 200)
    blk = P.vr180_frame(img, img, K, n, fill="black")
    assert tuple(blk[0, 0]) == (0, 0, 0)


def test_vr180_validation():
    img, K = _photo_with_marker(160, 120)
    with pytest.raises(InputError):
        P.vr180_frame(img, img, K, 64, layout="diagonal")
    with pytest.raises(InputError):
        P.vr180_frame(img, img, K, 64, fill="outpaint")
    with pytest.raises(InputError):
        P.vr180_frame(img, img[:, :100], K, 64)
    with pytest.raises(InputError):
        P.vr180_frame(img, img, K, 4)


def test_vr180_rescales_K_to_eye_images():
    img, K = _photo_with_marker()
    small = img[::2, ::2].copy()  # K still describes the full-size image
    n = 900
    fr = P.vr180_frame(small, small, K, n, fill="black")
    span = np.count_nonzero(fr[n // 2, :n].sum(1) > 0)
    assert span == pytest.approx(65.0 / 180.0 * n, abs=2)


def test_map_cache_and_tiled_path_agree(monkeypatch):
    img, K = _photo_with_marker(320, 240)
    P.clear_map_cache()
    a = P.vr180_frame(img, img, K, 300, fill="black")
    b = P.vr180_frame(img, img, K, 300, fill="black")  # served from the map cache
    assert np.array_equal(a, b)
    c = P.vr180_frame(img, img, K, 300, fill="black", size_mult=1.3)  # different key
    assert not np.array_equal(a, c)
    eq1, v1 = P.persp_to_equirect(img, K, 400, 400)
    monkeypatch.setattr(P, "_MAP_CACHE_MAX_PX", 0)  # force the tiled, uncached path
    monkeypatch.setattr(P, "_TILE_PX", 5000)
    eq2, v2 = P.persp_to_equirect(img, K, 400, 400)
    assert np.array_equal(v1, v2)
    assert np.abs(eq1.astype(int) - eq2.astype(int)).max() <= 1
    d = P.vr180_frame(img, img, K, 300, fill="black")
    assert np.abs(a.astype(int) - d.astype(int)).max() <= 1
    P.clear_map_cache()


def test_invalid_intrinsics_rejected():
    img = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(InputError):
        P.persp_to_equirect(img, Intrinsics(0.0, 10.0, 5.0, 5.0, 10, 10), 64, 64)
    with pytest.raises(InputError):
        P.equirect_to_persp(np.zeros((32, 64, 3), np.uint8), Intrinsics(float("nan"), 1.0, 0.0, 0.0, 8, 8))


# --------------------------------------------------------------------------- 360 from a photo
def test_pano_from_photo(photo_rgb):
    K = Intrinsics.from_hfov(320, 240, 65.0)
    w = 1440  # 4 px / deg
    pano, valid = P.pano_from_photo(photo_rgb, K, w)
    assert pano.shape == (w // 2, w, 3) and pano.dtype == np.uint8
    assert valid.shape == (w // 2, w) and valid.dtype == bool
    cols = np.nonzero(valid[w // 4])[0]
    assert cols.size == pytest.approx(65 * 4, abs=2)
    assert (cols.min() + cols.max()) / 2 == pytest.approx((w - 1) / 2, abs=1.5)
    assert pano[~valid].min() > 0  # surroundings are filled
    # seamless around the back (+-180 deg)
    seam = np.abs(pano[:, 0].astype(int) - pano[:, -1].astype(int)).max()
    inner = np.abs(np.diff(pano[:, w // 2 + 300 : w // 2 + 400].astype(int), axis=1)).max()
    assert seam <= max(3, inner + 1)
    black, vb = P.pano_from_photo(photo_rgb, K, 720, fill="black")
    import cv2

    rim = cv2.dilate(vb.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # anti-aliased edge
    assert (black[~rim] == 0).all() and vb.any()


def test_pano_from_photo_validation(photo_rgb):
    K = Intrinsics.from_hfov(320, 240, 65.0)
    with pytest.raises(InputError):
        P.pano_from_photo(photo_rgb, K, 4)
    with pytest.raises(InputError):
        P.pano_from_photo(photo_rgb, K, 256, fill="sparkles")


# --------------------------------------------------------------------------- review additions
def test_equirect_dirs_tiled_fill_matches_untiled(monkeypatch):
    a = P.equirect_dirs(97, 41, lon_range=(-100, 120), lat_range=(-60, 80))
    monkeypatch.setattr(P, "_TILE_PX", 300)  # several row tiles, last one partial
    b = P.equirect_dirs(97, 41, lon_range=(-100, 120), lat_range=(-60, 80))
    assert b.dtype == np.float32 and b.flags.c_contiguous
    assert np.array_equal(a, b)


def test_persp_to_equirect_is_not_mirrored():
    """Photo right -> canvas right (+lon), photo top -> canvas top (+lat), at true angles."""
    w, h = 400, 300
    img = np.full((h, w, 3), 90, np.uint8)
    img[40:52, 330:342] = RED  # top-right quadrant, centre (335.5, 45.5)
    K = Intrinsics(250.0, 250.0, 199.5, 149.5, w, h)
    n = 1800  # 10 px / deg
    eq, _ = P.persp_to_equirect(img, K, n, n)
    x, y, _ = red_centroid(eq)
    lam = -90 + (x + 0.5) / n * 180
    phi = 90 - (y + 0.5) / n * 180
    X, Y = (335.5 - 199.5) / 250.0, (45.5 - 149.5) / 250.0  # y-down ray (X, Y, 1)
    assert lam == pytest.approx(math.degrees(math.atan2(X, 1.0)), abs=0.15)
    assert phi == pytest.approx(math.degrees(math.atan2(-Y, math.hypot(X, 1.0))), abs=0.15)
    assert lam > 0 and phi > 0


def test_vr180_eye_order_and_orientation():
    """Left eye -> left half (sbs) / top half (tb); content keeps its orientation."""
    w, h = 320, 240
    K = Intrinsics(200.0, 200.0, 159.5, 119.5, w, h)
    left = np.full((h, w, 3), 90, np.uint8)
    right = left.copy()
    left[20:40, 260:280] = RED  # top-right of the LEFT image only
    right[200:220, 40:60] = (0, 0, 255)  # bottom-left of the RIGHT image only
    n = 400
    for layout in ("sbs", "tb"):
        fr = P.vr180_frame(left, right, K, n, fill="black", layout=layout)
        a, b = (fr[:, :n], fr[:, n:]) if layout == "sbs" else (fr[:n], fr[n:])
        x, y, _ = red_centroid(a)
        assert x > n / 2 and y < n / 2
        assert not ((b[..., 0] > 200) & (b[..., 1] < 70) & (b[..., 2] < 70)).any()
        blue = (b[..., 2] > 200) & (b[..., 0] < 70)
        ys, xs = np.nonzero(blue)
        assert xs.size and xs.mean() < n / 2 and ys.mean() > n / 2
        assert not ((a[..., 2] > 200) & (a[..., 0] < 70)).any()


def test_cube_face_rotations_match_pose_look():
    from vrphoto.core.types import Pose

    turns = {"+z": (0, 0), "-z": (180, 0), "+x": (90, 0), "-x": (-90, 0), "+y": (0, -90), "-y": (0, 90)}
    for k, (yaw, pitch) in turns.items():
        assert np.allclose(P.cube_face_rotation(k), Pose.look(yaw, pitch).R, atol=1e-12), k
        R = P.cube_face_rotation(k)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_tiny_and_odd_sizes_do_not_crash():
    one = np.full((1, 1, 3), 100, np.uint8)
    K1 = Intrinsics(1.0, 1.0, 0.0, 0.0, 1, 1)
    assert P.equirect_to_persp(one, K1).shape == (1, 1, 3)
    assert P.equirect_to_persp(one[..., 0], Intrinsics.from_hfov(5, 3, 60)).shape == (3, 5)
    eq, valid = P.persp_to_equirect(one, K1, 3, 1, lon_range=(-180, 180))
    assert eq.shape == (1, 3, 3) and valid.shape == (1, 3)
    faces = P.equirect_to_cubemap(np.full((3, 7, 3), 9, np.uint8), 1)
    assert all(f.shape == (1, 1, 3) for f in faces.values())
    assert P.cubemap_to_equirect(faces, 1, 1).shape == (1, 1, 3)
    odd = {k: np.full((3, 3), 0.25, np.float32) for k in P.CUBE_FACES}
    back = P.cubemap_to_equirect(odd, 7, 3)
    assert back.shape == (3, 7) and np.allclose(back, 0.25)
    fr = P.vr180_frame(one, one, Intrinsics.from_hfov(1, 1, 60), 9, layout="tb")
    assert fr.shape == (18, 9, 3)
    pano, v = P.pano_from_photo(np.full((7, 9, 3), 50, np.uint8), Intrinsics.from_hfov(9, 7, 60), 17)
    assert pano.shape == (8, 17, 3) and v.any()


def test_wide_canvas_beyond_remap_limit():
    """Canvases wider than cv2.remap's 32767 limit go through the tiled path."""
    img = np.full((20, 30, 3), 200, np.uint8)
    K = Intrinsics.from_hfov(30, 20, 36.0)
    w = 40000
    eq, valid = P.persp_to_equirect(img, K, w, 9, lon_range=(-180, 180))
    assert eq.shape == (9, w, 3)
    span = np.count_nonzero(valid[4])  # row 4 of 9 is the equator
    assert span == pytest.approx(36.0 / 360.0 * w, abs=2)
    assert (eq[valid] == 200).all()


def test_non_numeric_parameters_raise_input_error():
    img, K = _photo_with_marker(160, 120)
    with pytest.raises(InputError):
        P.vr180_frame(img, img, K, "big")
    with pytest.raises(InputError):
        P.vr180_frame(img, img, K, 64, size_mult="x")
    with pytest.raises(InputError):
        P.vr180_frame(img, img, K, 64, size_mult=float("nan"))
    with pytest.raises(InputError):
        P.pano_from_photo(img, K, None)
    with pytest.raises(InputError):
        P.equirect_to_cubemap(np.zeros((8, 16, 3), np.uint8), "six")
    with pytest.raises(InputError):
        P.equirect_dirs("a", 4)
    with pytest.raises(InputError):
        P.vr180_frame(img, img, None, 64)
    with pytest.raises(InputError):
        P.vr180_frame(img, img, Intrinsics(100.0, 100.0, 80.0, 60.0, 0, 120), 64)


def test_photo_dtype_and_channel_coercion():
    K = Intrinsics.from_hfov(16, 12, 60.0)
    ref = np.full((12, 16, 3), 156, np.uint8)
    pano_ref, valid = P.pano_from_photo(ref, K, 256, fill="black")
    centre = (64, 128)
    variants = {
        "uint16": np.full((12, 16, 3), 40000, np.uint16),  # 40000 / 257 = 155.6
        "float01": np.full((12, 16, 3), 156 / 255, np.float32),
        "float255": np.full((12, 16, 3), 156.0, np.float64),
        "gray": np.full((12, 16), 156, np.uint8),
        "gray1": np.full((12, 16, 1), 156, np.uint8),
        "rgba": np.dstack([ref, np.zeros((12, 16), np.uint8)]),
    }
    for name, img in variants.items():
        pano, v = P.pano_from_photo(img, K, 256, fill="black")
        assert np.array_equal(v, valid), name
        assert tuple(pano[centre]) == (156, 156, 156), name
    bad = np.full((12, 16, 3), 0.5, np.float32)
    bad[0, 0] = np.nan
    bad[1, 1] = np.inf
    pano, _ = P.pano_from_photo(bad, K, 256, fill="black")  # no warnings / garbage
    assert tuple(pano[centre]) == (128, 128, 128)
    with pytest.raises(InputError):
        P.pano_from_photo(np.zeros((12, 16, 2), np.uint8), K, 256)


def test_fill_image_variants():
    img, K = _photo_with_marker(160, 120)
    n = 128
    gray = np.full((32, 32), 77, np.uint8)
    assert tuple(P.vr180_frame(img, img, K, n, fill=gray)[2, 2]) == (77, 77, 77)
    flt = np.full((16, 16, 3), 0.5, np.float32)  # [0,1] floats are scaled, not truncated
    assert tuple(P.vr180_frame(img, img, K, n, fill=flt)[2, 2]) == (128, 128, 128)
    rgba = np.zeros((16, 16, 4), np.uint8)
    rgba[..., 1] = 99
    assert tuple(P.vr180_frame(img, img, K, n, fill=rgba)[2, 2]) == (0, 99, 0)
    with pytest.raises(InputError):
        P.vr180_frame(img, img, K, n, fill=np.zeros((1, 5), np.uint8))


def _quadrant_photo(w=320, h=240):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.where(xx < w // 2, 230, 20)
    img[..., 1] = np.where(yy < h // 2, 210, 30)
    img[..., 2] = 90
    return img, Intrinsics.from_hfov(w, h, 65.0)


def test_blur_fill_has_no_pole_pinch():
    """Equirect rows at the poles are single points on the sphere: the blurred fill must be
    (nearly) constant along them, otherwise the zenith/nadir shows a star-shaped pinch."""
    img, K = _quadrant_photo()
    pano, _ = P.pano_from_photo(img, K, 1024)
    vr = P.vr180_frame(img, img, K, 512)
    for rows in (pano[:2], pano[-2:], vr[:2, :512], vr[-2:, :512], vr[:2, 512:]):
        r = rows.astype(np.float64)
        assert r.std(axis=1).max() < 1.0
    # still a gradual, colour-continuing fill near the equator (not a flat average)
    eq_row = pano[256].astype(np.float64)
    assert np.ptp(eq_row[:, 0]) > 30
    # and still seamless in longitude
    assert np.abs(pano[:, 0].astype(int) - pano[:, -1].astype(int)).max() <= 3


def test_cubemap_antialiased_from_dense_equirect():
    h, w = 1024, 2048
    yy, xx = np.mgrid[0:h, 0:w]
    eq = np.repeat((((xx + yy) % 2) * 255).astype(np.uint8)[..., None], 3, axis=2)
    faces = P.equirect_to_cubemap(eq, 48)
    for k, f in faces.items():
        v = f.astype(np.float64)
        assert abs(v.mean() - 127.5) < 4 and v.std() < 10, k
