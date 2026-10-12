"""Tests for geometry.billboard: closed walk-around relief from a matte (algorithm I)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, DepthResult, Intrinsics, Mesh
from vrphoto.geometry.billboard import _dist, billboard, inflation_profile, local_thickness


def _person_photo(alpha: np.ndarray) -> np.ndarray:
    """Clothes colour on the body, a high-frequency 'face' on the head, grey background."""
    h, w = alpha.shape
    img = np.full((h, w, 3), 90, np.uint8)
    img[alpha > 0.5] = (40, 60, 200)
    rng = np.random.default_rng(0)
    img[32:48, 88:104] = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)  # inside the head
    return img


@pytest.fixture
def person(person_alpha):
    h, w = person_alpha.shape
    img = _person_photo(person_alpha)
    depth = DepthResult(np.full((h, w), 0.5, np.float32), "test")
    K = Intrinsics.from_hfov(w, h, 60.0)
    return img, person_alpha, depth, K


def _merged(front: Mesh, back: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Weld the two meshes on identical vertex positions -> (vertices, faces)."""
    v = np.concatenate([front.vertices, back.vertices]).astype(np.float64)
    f = np.concatenate([front.faces.astype(np.int64), back.faces.astype(np.int64) + len(front.vertices)])
    uniq, inv = np.unique(np.round(v, 7), axis=0, return_inverse=True)
    return uniq, inv.reshape(-1)[f]


def _edge_use(faces: np.ndarray) -> np.ndarray:
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return counts


def test_billboard_dimensions_and_placement(person):
    img, alpha, depth, K = person
    front, back = billboard(img, alpha, depth, K)
    v = np.concatenate([front.vertices, back.vertices])
    assert abs(v[:, 1].min()) <= 1e-4  # feet on the floor
    assert v[:, 1].max() - v[:, 1].min() == pytest.approx(1.7, abs=0.01)
    assert (v[:, 0].min() + v[:, 0].max()) / 2 == pytest.approx(0.0, abs=1e-5)
    assert (v[:, 2].min() + v[:, 2].max()) / 2 == pytest.approx(0.0, abs=1e-5)
    # width follows the silhouette aspect: 101 px wide x 229 px tall
    width = v[:, 0].max() - v[:, 0].min()
    assert width == pytest.approx(1.7 * 100 / 228, rel=0.04)
    # thickness ratio: median front-back distance = thickness * height
    th = front.vertices[:, 2] - back.vertices[:, 2]
    assert np.median(th) / 1.7 == pytest.approx(0.17, abs=0.01)
    assert th.min() >= 0


def test_billboard_is_closed_and_consistently_oriented(person):
    img, alpha, depth, K = person
    front, back = billboard(img, alpha, depth, K)
    assert np.array_equal(front.uvs, back.uvs) and np.array_equal(front.faces[:, 0], back.faces[:, 0])
    v, f = _merged(front, back)
    counts = _edge_use(f)
    assert int((counts == 1).sum()) == 0, "open boundary edges"
    assert int((counts > 2).sum()) <= 2
    # front faces look toward +z, back faces toward -z
    for m, sign in ((front, 1), (back, -1)):
        tri = m.vertices.astype(np.float64)[m.faces.astype(np.int64)]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert np.all(sign * n[:, 2] > 0)


def test_billboard_watertight_in_trimesh(person, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    img, alpha, depth, K = person
    meshes = billboard(img, alpha, depth, K)
    v, f = _merged(*meshes)
    tm = trimesh.Trimesh(v, f, process=False)  # already welded on exact positions
    assert tm.is_watertight and tm.is_winding_consistent
    assert tm.volume > 0  # outward normals


def test_billboard_textures_and_alpha_mask(person):
    img, alpha, depth, K = person
    front, back = billboard(img, alpha, depth, K, back="diffuse")
    for m in (front, back):
        assert m.alpha_mode == "MASK" and m.alpha_cutoff == 0.5 and m.texture.shape[2] == 4
        assert m.uvs.min() >= 0 and m.uvs.max() <= 1
    ft = front.texture
    # alpha follows the matte, colours outside are bled from the inside (no halo)
    assert ft[0, 0, 3] == 0 and ft[ft.shape[0] // 2, ft.shape[1] // 2, 3] == 255
    outside = ft[..., 3] == 0
    assert np.abs(ft[outside][:, :3].astype(int) - (90, 90, 90)).sum(1).mean() > 30
    # every vertex samples an opaque texel
    px = np.clip(
        (front.uvs * [ft.shape[1], ft.shape[0]] - 0.5).round().astype(int),
        0,
        [ft.shape[1] - 1, ft.shape[0] - 1],
    )
    assert np.mean(ft[px[:, 1], px[:, 0], 3] >= 128) > 0.97


def test_diffuse_back_has_no_face_and_is_darker(person):
    img, alpha, depth, K = person
    front, back = billboard(img, alpha, depth, K, back="diffuse")
    bt, ft = back.texture, front.texture
    sy, sx = bt.shape[0] / ft.shape[0], bt.shape[1] / ft.shape[1]
    # head region in crop coordinates (crop starts 2% + 2 px before the silhouette bbox)
    ys, xs = np.nonzero(alpha > 0.5)
    y0, x0 = ys.min() - (int(round(0.02 * 229)) + 2), xs.min() - (int(round(0.02 * 229)) + 2)
    fy, fx = slice(32 - y0, 48 - y0), slice(88 - x0, 104 - x0)
    by, bx = slice(int((32 - y0) * sy), int((48 - y0) * sy)), slice(int((88 - x0) * sx), int((104 - x0) * sx))
    face_front = ft[fy, fx, :3].astype(float)
    face_back = bt[by, bx, :3].astype(float)
    front_std = face_front.reshape(-1, 3).std(0).max()
    back_std = face_back.reshape(-1, 3).std(0).max()
    assert front_std > 40  # the random 'face' is on the front
    assert back_std < 0.1 * front_std  # ... but not on the back
    # torso: same clothes colour, 10 % darker
    ty, tx = (
        slice(int((100 - y0) * sy), int((110 - y0) * sy)),
        slice(int((90 - x0) * sx), int((100 - x0) * sx)),
    )
    np.testing.assert_allclose(bt[ty, tx, :3].reshape(-1, 3).mean(0), np.array([40, 60, 200]) * 0.9, atol=3)


def test_mirror_back_reuses_front_texture(person):
    img, alpha, depth, K = person
    front, back = billboard(img, alpha, depth, K, back="mirror")
    assert back.texture is front.texture
    assert np.array_equal(back.uvs, front.uvs)


def test_custom_height_and_zero_thickness(person):
    img, alpha, depth, K = person
    front, back = billboard(img, alpha, depth, K, height_m=1.2, thickness=0.0)
    v = np.concatenate([front.vertices, back.vertices])
    assert v[:, 1].max() == pytest.approx(1.2, abs=0.01) and abs(v[:, 1].min()) <= 1e-4
    np.testing.assert_allclose(front.vertices, back.vertices, atol=1e-7)


def test_thickness_scales_inflation(person):
    img, alpha, depth, K = person
    thin = billboard(img, alpha, depth, K, thickness=0.08)
    thick = billboard(img, alpha, depth, K, thickness=0.16)
    t1 = np.median(thin[0].vertices[:, 2] - thin[1].vertices[:, 2])
    t2 = np.median(thick[0].vertices[:, 2] - thick[1].vertices[:, 2])
    assert t2 / t1 == pytest.approx(2.0, rel=0.02)
    # a round cross-section: the torso centre is thicker than a leg centre
    f, b = thick
    uv = f.uvs
    d = f.vertices[:, 2] - b.vertices[:, 2]
    torso = d[np.argmin(np.hypot(uv[:, 0] - 0.5, uv[:, 1] - 0.4))]
    leg = d[np.argmin(np.hypot(uv[:, 0] - 0.38, uv[:, 1] - 0.85))]
    assert torso > 1.8 * leg


def test_depth_detail_is_bounded_to_3cm(person):
    img, alpha, depth, K = person
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    bump = 0.5 + 0.2 * np.exp(-((xx - 96) ** 2 + (yy - 100) ** 2) / (2 * 8.0**2))
    flat = billboard(img, alpha, depth, K)
    bumpy = billboard(img, alpha, DepthResult(bump.astype(np.float32), "bump"), K)
    th_flat = flat[0].vertices[:, 2] - flat[1].vertices[:, 2]
    th_bump = bumpy[0].vertices[:, 2] - bumpy[1].vertices[:, 2]
    dz = th_bump - th_flat
    assert dz.max() == pytest.approx(0.03, abs=0.004)  # nearer -> bulges toward the viewer
    assert dz.min() >= -0.0301
    peak = np.argmax(dz)
    assert np.hypot(*(bumpy[0].uvs[peak] - flat[0].uvs[peak])) == 0
    # the back only moves by the re-centring shift
    shift = bumpy[1].vertices - flat[1].vertices
    assert np.ptp(shift[:, 2]) < 1e-6 and np.abs(shift[:, :2]).max() < 1e-6


def test_specks_are_ignored_and_depth_may_differ_in_size(person):
    img, alpha, depth, K = person
    a2 = alpha.copy()
    a2[2:4, 2:4] = 1.0  # tiny speck far from the subject
    small_depth = DepthResult(cv2.resize(depth.disparity, (96, 128)), "half")
    ref = billboard(img, alpha, depth, K)
    out = billboard(img, a2, small_depth, K)
    np.testing.assert_allclose(out[0].vertices, ref[0].vertices, atol=1e-5)


def test_billboard_errors_and_cancel(person):
    img, alpha, depth, K = person
    with pytest.raises(InputError):
        billboard(img, np.zeros_like(alpha), depth, K)
    with pytest.raises(InputError):
        billboard(img, alpha, depth, K, back="glossy")
    with pytest.raises(InputError):
        billboard(img, alpha[:-1], depth, K)
    with pytest.raises(InputError):
        billboard(img, alpha, depth, K, height_m=0.0)
    with pytest.raises(InputError):
        billboard(img, alpha, depth, K, thickness=-1.0)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        billboard(img, alpha, depth, K, cancel=tok)
    seen = []
    billboard(img, alpha, depth, K, progress=lambda f, m: seen.append(f))
    assert seen[-1] == pytest.approx(1.0)


def test_large_photo_is_processed_on_a_working_grid(person_alpha):
    big_a = cv2.resize(person_alpha, (192 * 8, 256 * 8), interpolation=cv2.INTER_NEAREST)
    img = _person_photo(cv2.resize(person_alpha, (192, 256)))
    img = cv2.resize(img, (192 * 8, 256 * 8), interpolation=cv2.INTER_NEAREST)
    K = Intrinsics.from_hfov(192 * 8, 256 * 8, 60.0)
    front, back = billboard(img, big_a, DepthResult(np.full(big_a.shape, 0.5, np.float32), "t"), K)
    assert len(front.vertices) < 40_000
    assert front.texture.shape[0] > 1500  # full-resolution texture crop
    assert back.texture.shape[0] <= 1024  # diffuse back at reduced resolution
    ft = front.texture
    rim = (ft[..., 3] == 0) & (cv2.dilate((ft[..., 3] > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    assert rim.sum() > 1000
    # colours just outside the matte are bled from the inside (no grey background halo)
    assert np.mean(np.abs(ft[rim][:, :3].astype(int) - (90, 90, 90)).sum(1) > 60) > 0.9
    v = np.concatenate([front.vertices, back.vertices])
    assert v[:, 1].max() == pytest.approx(1.7, abs=0.01)


def test_local_thickness_of_a_disc_and_a_bar():
    m = np.zeros((101, 201), bool)
    cv2.circle(m.view(np.uint8), (50, 50), 30, 1, -1)
    m[45:56, 100:200] = True  # 11 px bar attached to the disc
    dt = cv2.distanceTransform(np.pad(m.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    r = local_thickness(dt, m)
    assert r[50, 50] == pytest.approx(dt[50, 50], abs=1.0)
    assert r[50, 25] >= 26  # near the disc rim the radius is still the disc's (level quantisation)
    assert r[50, 150] <= 7  # the bar keeps its own small radius
    assert np.all(r[m] >= dt[m] - 1e-6)


# --------------------------------------------------------------------------- review additions
def test_front_texture_is_not_mirrored(person):
    """Pins the facing convention: a viewer at +z looking down -z sees the photo as taken
    (the subject's image-left arm at negative x), and UVs are an exact affine function of
    the vertex x/y (u grows with x, v grows downward)."""
    img, alpha, depth, K = person
    img = img.copy()
    img[66:141, 46:65] = (255, 0, 0)  # image-left arm red
    img[66:141, 128:147] = (0, 255, 0)  # image-right arm green
    front, back = billboard(img, alpha, depth, K)
    ft = front.texture
    px = np.clip(
        (front.uvs * [ft.shape[1], ft.shape[0]] - 0.5).round().astype(int),
        0,
        [ft.shape[1] - 1, ft.shape[0] - 1],
    )
    col = ft[px[:, 1], px[:, 0], :3].astype(int)
    red = np.all(col == (255, 0, 0), axis=1)
    green = np.all(col == (0, 255, 0), axis=1)
    assert red.sum() > 20 and green.sum() > 20
    assert np.all(front.vertices[red, 0] < 0) and np.all(front.vertices[green, 0] > 0)
    A = np.column_stack([front.vertices[:, 0], front.vertices[:, 1], np.ones(len(front.vertices))])
    coef, *_ = np.linalg.lstsq(A.astype(np.float64), front.uvs.astype(np.float64), rcond=None)
    np.testing.assert_allclose(A @ coef, front.uvs, atol=1e-5)
    assert coef[0, 0] > 0 and coef[1, 1] < 0  # du/dx > 0, dv/dy < 0 (y up, v down)
    # the front surface bulges toward +z, the back toward -z
    assert np.median(front.vertices[:, 2]) > np.median(back.vertices[:, 2])


def test_billboard_input_errors_are_readable(person):
    img, alpha, depth, K = person
    with pytest.raises(InputError, match="depth"):
        billboard(img, alpha, None, K)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="subject mask"):
        billboard(img, np.float32(1.0), depth, K)  # type: ignore[arg-type]
    with pytest.raises(InputError):
        billboard(img, alpha, depth, K, height_m=float("nan"))


def test_float_and_grayscale_inputs(person):
    img, alpha, depth, K = person
    ref = billboard(img, alpha, depth, K)
    gray = billboard(img[..., 1], alpha, depth, K)
    np.testing.assert_allclose(gray[0].vertices, ref[0].vertices)
    assert gray[0].texture.shape[2] == 4
    fl = billboard(img.astype(np.float32) / 255.0, alpha.astype(bool), depth, K)
    np.testing.assert_allclose(fl[0].vertices, ref[0].vertices, atol=1e-6)


def test_inflation_profile_is_a_circular_cross_section():
    """SHARP 360's shell back reuses the billboard profile: full-grid float32 maps, 0 outside."""
    m = np.zeros((81, 121), bool)
    cv2.circle(m.view(np.uint8), (40, 40), 25, 1, -1)
    m[36:45, 66:118] = True  # thin bar: its own small radius
    prof, dt_eff = inflation_profile(m)
    assert prof.shape == dt_eff.shape == m.shape and prof.dtype == dt_eff.dtype == np.float32
    assert (prof[~m] == 0).all() and (dt_eff[~m] == 0).all()
    np.testing.assert_allclose(dt_eff, np.maximum(_dist(m) - 1.0, 0.0), atol=1e-5)  # cv2 threads: 1 ulp
    rloc = local_thickness(dt_eff, m)
    np.testing.assert_allclose(
        prof[m], np.sqrt(np.maximum(dt_eff * (2 * rloc - dt_eff), 0))[m], rtol=1e-5, atol=1e-4
    )
    assert prof[40, 40] == pytest.approx(dt_eff[40, 40], abs=1.0)  # disc centre: the full radius
    assert prof[40, 100] < 6 and prof[40, 100] < 0.3 * prof[40, 40]  # the bar stays thin
    assert prof[m & (dt_eff == 0)].max() == 0  # boundary pixels are the silhouette itself
    empty, dt0 = inflation_profile(np.zeros((5, 7), np.uint8))
    assert not empty.any() and not dt0.any()
