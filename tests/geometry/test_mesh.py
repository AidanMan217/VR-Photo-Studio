"""Tests for geometry.mesh: depth meshes, layered meshes and decimation (algorithm H)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, Intrinsics, Layer, LayeredImage, Mesh, ProgressReporter
from vrphoto.geometry import mesh as meshmod
from vrphoto.geometry.camera import gltf_to_cv
from vrphoto.geometry.mesh import (
    BACKDROP_PUSH,
    MeshError,
    decimate,
    depth_mesh,
    layered_meshes,
    vertex_normals,
)

HAS_FS = meshmod._fast_simplification() is not None


@pytest.fixture(params=["qem", "grid"])
def simplify_mode(request, monkeypatch):
    """Run a test with quadric simplification (when installed) and with the plain grid."""
    if request.param == "qem":
        if not HAS_FS:
            pytest.skip("fast-simplification not installed")
    else:
        monkeypatch.setattr(meshmod, "_fast_simplification", lambda: None)
    return request.param


def _smooth_depth(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (1.0 / (2.0 + 0.5 * np.sin(xx / 40.0) * np.cos(yy / 30.0))).astype(np.float32)


def _project_uv(
    m: Mesh, K: Intrinsics, eye=(0.0, 0.0, 0.0), scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Project glTF-frame vertices (seen from ``eye``) into the source camera -> (uv, z)."""
    p = gltf_to_cv((m.vertices.astype(np.float64) - np.asarray(eye)) / scale)
    u = K.fx * p[:, 0] / p[:, 2] + K.cx
    v = K.fy * p[:, 1] / p[:, 2] + K.cy
    return np.column_stack([(u + 0.5) / K.width, (v + 0.5) / K.height]), p[:, 2]


def _face_spans(m: Mesh, z_split: float) -> int:
    z = -m.vertices[:, 2].astype(np.float64)
    fz = z[m.faces.astype(np.int64)]
    return int(((fz.max(1) > z_split) & (fz.min(1) < z_split)).sum())


def _components(faces: np.ndarray, n: int) -> np.ndarray:
    """Vertex-connected component size (in faces) for every face (simple union-find)."""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in faces:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(f[0]) for f in faces])
    _, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    return counts[inv]


# --------------------------------------------------------------------------- depth_mesh
@pytest.mark.parametrize("target", [3000, 20000, 60000])
def test_face_count_within_10pct_of_target_for_smooth_depth(simplify_mode, target):
    h, w = 240, 320
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = np.full((h, w, 3), 128, np.uint8)
    m = depth_mesh(img, _smooth_depth(h, w), K, target_faces=target)
    assert abs(len(m.faces) - target) <= 0.1 * target
    assert m.vertices.dtype == np.float32 and m.faces.dtype == np.uint32 and m.uvs.dtype == np.float32
    assert m.faces.max() < len(m.vertices)


def test_no_triangle_spans_the_square_edge(two_plane_scene, simplify_mode):
    s = two_plane_scene
    for target in (2000, 20000, 80000):
        m = depth_mesh(s.img, s.inv_z, s.K, target_faces=target)
        assert _face_spans(m, 2.5) == 0, f"target {target}: a triangle bridges the depth edge"
        z = -m.vertices[:, 2]
        assert np.any(np.isclose(z, s.z_near, atol=1e-4)) and np.any(np.isclose(z, s.z_far, atol=1e-4))
        # every vertex lies exactly on one of the two planes
        assert np.all(np.isclose(z, s.z_near, atol=1e-4) | np.isclose(z, s.z_far, atol=1e-4))


def test_vertices_project_to_their_uvs_and_cover_the_photo(two_plane_scene, simplify_mode):
    s = two_plane_scene
    m = depth_mesh(s.img, s.inv_z, s.K, target_faces=10000)
    uv, z = _project_uv(m, s.K)
    assert np.all(z > 0)
    np.testing.assert_allclose(uv, m.uvs, atol=1e-5)
    np.testing.assert_allclose(m.uvs.min(0), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(m.uvs.max(0), [1.0, 1.0], atol=1e-6)
    assert m.texture is s.img and m.alpha_mode == "OPAQUE" and m.name == "layer"


def test_faces_are_counter_clockwise_toward_the_camera(two_plane_scene):
    s = two_plane_scene
    m = depth_mesh(s.img, s.inv_z, s.K, target_faces=5000)
    v = m.vertices.astype(np.float64)[m.faces.astype(np.int64)]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    centre = v.mean(axis=1)
    # the camera is at the origin: normals point back toward it
    assert np.all(np.einsum("ij,ij->i", n, -centre) > 0)


def test_cut_rule_uses_parallax_pixels(two_plane_scene):
    s = two_plane_scene
    h, w = s.inv_z.shape
    # a step of 1 px of parallax at 0.1 m (below cut_px=2) must stay connected
    step = 1.0 / (s.K.fx * 0.1)
    inv = np.where(s.fg_mask, 0.25 + step, 0.25).astype(np.float32)
    m = depth_mesh(s.img, inv, s.K, target_faces=8000)
    z_split = 0.5 * (1 / 0.25 + 1 / (0.25 + step))
    assert _face_spans(m, z_split) > 0
    # the same step with a tighter cut (0.5 px) is cut
    m2 = depth_mesh(s.img, inv, s.K, target_faces=8000, cut_px=0.5)
    assert _face_spans(m2, z_split) == 0
    # a larger reference baseline turns the same step into a cut as well
    m3 = depth_mesh(s.img, inv, s.K, target_faces=8000, ref_baseline_m=0.3)
    assert _face_spans(m3, z_split) == 0


def test_alpha_drops_vertices_and_makes_a_mask_texture(two_plane_scene, simplify_mode):
    s = two_plane_scene
    h, w = s.inv_z.shape
    alpha = np.ones((h, w), np.float32)
    alpha[:, : w // 2] = 0.0
    m = depth_mesh(s.img, s.inv_z, s.K, alpha=alpha, target_faces=10000, name="subject")
    uv, _ = _project_uv(m, s.K)
    u_px = uv[:, 0] * w - 0.5
    assert u_px.min() >= w // 2 - 0.5 - 1e-3
    assert m.texture.shape == (h, w, 4) and m.alpha_mode == "MASK" and m.name == "subject"
    assert m.texture[0, 0, 3] == 0 and m.texture[0, -1, 3] == 255


def test_rgba_input_uses_its_alpha(two_plane_scene):
    s = two_plane_scene
    h, w = s.inv_z.shape
    rgba = np.dstack([s.img, np.where(s.fg_mask, 255, 0).astype(np.uint8)])
    m = depth_mesh(rgba, s.inv_z, s.K, target_faces=10000)
    assert np.allclose(-m.vertices[:, 2], s.z_near, atol=1e-4)  # only the square survives
    assert m.alpha_mode == "MASK"


def test_small_components_are_removed(two_plane_scene, monkeypatch):
    monkeypatch.setattr(meshmod, "_fast_simplification", lambda: None)
    s = two_plane_scene
    inv = np.full(s.inv_z.shape, 0.25, np.float32)
    inv[20:25, 20:25] = 1.0  # tiny floater: 32 faces at stride 1
    inv[100:200, 150:300] = 1.0  # large near object
    m = depth_mesh(s.img, inv, s.K, target_faces=300_000)
    z = -m.vertices[:, 2]
    uv, _ = _project_uv(m, s.K)
    near = np.isclose(z, 1.0, atol=1e-4)
    px = uv[near] * [s.K.width, s.K.height] - 0.5
    assert near.any()
    assert px[:, 0].min() >= 149 and px[:, 1].min() >= 99  # the floater is gone
    sizes = _components(m.faces.astype(np.int64), len(m.vertices))
    assert sizes.min() >= meshmod._min_component_faces(300_000) == 50


def test_invalid_depth_is_not_meshed(two_plane_scene):
    s = two_plane_scene
    inv = s.inv_z.copy()
    inv[:, :100] = np.nan
    inv[:20] = 0.0
    m = depth_mesh(s.img, inv, s.K, target_faces=10000)
    assert np.isfinite(m.vertices).all()
    uv, _ = _project_uv(m, s.K)
    px = uv * [s.K.width, s.K.height] - 0.5
    assert px[:, 0].min() >= 99 and px[:, 1].min() >= 19
    with pytest.raises(MeshError):
        depth_mesh(s.img, np.full_like(inv, np.nan), s.K)


def test_depth_mesh_validates_inputs(two_plane_scene):
    s = two_plane_scene
    with pytest.raises(InputError):
        depth_mesh(s.img, s.inv_z[:-1], s.K)
    with pytest.raises(InputError):
        depth_mesh(s.img, s.inv_z, s.K.scaled(0.5))
    with pytest.raises(InputError):
        depth_mesh(s.img, s.inv_z, s.K, alpha=np.ones((3, 3), np.float32))
    with pytest.raises(InputError):
        depth_mesh(s.img, s.inv_z, s.K, target_faces=1)
    with pytest.raises(InputError):
        depth_mesh(s.img, s.inv_z, s.K, cut_px=0.0)
    with pytest.raises(InputError):
        depth_mesh(s.img, s.inv_z, s.K, ref_baseline_m=-1.0)
    with pytest.raises(InputError):
        depth_mesh(s.img[..., 0:2], s.inv_z, s.K)


def test_depth_mesh_progress_and_cancel(two_plane_scene):
    s = two_plane_scene
    seen: list[float] = []
    depth_mesh(s.img, s.inv_z, s.K, target_faces=4000, progress=ProgressReporter(lambda f, m: seen.append(f)))
    assert seen and seen[-1] == pytest.approx(1.0) and all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        depth_mesh(s.img, s.inv_z, s.K, cancel=tok)


def test_float_image_and_tiny_image_are_accepted():
    K = Intrinsics.from_hfov(4, 3, 60.0)
    img = np.random.default_rng(0).random((3, 4, 3)).astype(np.float32)
    m = depth_mesh(img, np.full((3, 4), 0.5, np.float32), K, target_faces=1000)
    assert len(m.faces) == 2 * 3 * 2 and m.texture.dtype == np.uint8
    with pytest.raises(InputError):
        depth_mesh(
            np.zeros((1, 5, 3), np.uint8), np.ones((1, 5), np.float32), Intrinsics.from_hfov(5, 1, 60.0)
        )


# --------------------------------------------------------------------------- layered_meshes
def _two_layer(s, alpha_front=None) -> LayeredImage:
    """Band-mode style layers: a 15 px ring inside the square edge is pushed to the far plane."""
    import cv2

    band = s.fg_mask & ~(cv2.erode(s.fg_mask.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
    bg_inv = np.where(band, 1.0 / s.z_far, s.inv_z).astype(np.float32)
    bg_rgb = np.where(band[..., None], s.far_rgb, s.img)
    ones = np.ones(s.inv_z.shape, np.float32)
    front_a = ones if alpha_front is None else alpha_front
    return LayeredImage([Layer(bg_rgb, bg_inv, ones.copy()), Layer(s.img, s.inv_z, front_a)], s.K)


def test_layered_meshes_structure_and_backdrop_behind(two_plane_scene):
    s = two_plane_scene
    li = _two_layer(s)
    meshes = layered_meshes(li, target_faces=20000)
    assert [m.name for m in meshes] == ["background", "foreground"]
    bg, fg = meshes
    assert bg.texture is li.layers[0].rgb and fg.texture is li.layers[1].rgb
    eye = (0.0, 1.5, 0.0)
    uv_f, _ = _project_uv(fg, s.K, eye)
    np.testing.assert_allclose(uv_f, fg.uvs, atol=1e-5)
    uv_b, z_b = _project_uv(bg, s.K, eye)
    np.testing.assert_allclose(uv_b, bg.uvs, atol=1e-5)
    # background = detail (pushed 0.2 %) + backdrop (pushed 1 %); nothing in front of the far plane
    # except the detail outside the band, which coincides (pushed back) with the front.
    bg_depth = 1.0 / li.layers[0].inv_z
    px = np.clip(
        np.rint(uv_b * [s.K.width, s.K.height] - 0.5).astype(int), 0, [s.K.width - 1, s.K.height - 1]
    )
    ref = bg_depth[px[:, 1], px[:, 0]]
    assert np.all(z_b >= ref * 1.0019)
    backdrop = np.isclose(z_b / ref, BACKDROP_PUSH, rtol=1e-4) | (z_b > ref * 1.005)
    assert backdrop.sum() > 50
    # backdrop surface is behind the background layer everywhere (dense sampling of its triangles)
    f = bg.faces.astype(np.int64)
    zf = z_b[f]
    bd_faces = f[np.all(zf >= ref[f] * 1.005, axis=1)]
    assert len(bd_faces) > 100
    rng = np.random.default_rng(0)
    r1, r2 = np.sqrt(rng.random((len(bd_faces), 8))), rng.random((len(bd_faces), 8))
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
    V = gltf_to_cv(bg.vertices.astype(np.float64) - eye)
    P = (
        V[bd_faces[:, 0]][:, None] * w0[..., None]
        + V[bd_faces[:, 1]][:, None] * w1[..., None]
        + V[bd_faces[:, 2]][:, None] * w2[..., None]
    ).reshape(-1, 3)
    u = np.clip(np.rint(s.K.fx * P[:, 0] / P[:, 2] + s.K.cx).astype(int), 0, s.K.width - 1)
    v = np.clip(np.rint(s.K.fy * P[:, 1] / P[:, 2] + s.K.cy).astype(int), 0, s.K.height - 1)
    assert np.all(P[:, 2] >= bg_depth[v, u] - 1e-6)
    # the front mesh is the cut mesh of the front layer
    assert _face_spans(fg, 2.5) == 0
    assert np.any(np.isclose(ref, s.z_near)) and np.any(np.isclose(ref, s.z_far))


def test_layered_background_detail_follows_the_hallucinated_region(two_plane_scene):
    s = two_plane_scene
    li = _two_layer(s)
    bg = layered_meshes(li, target_faces=20000)[0]
    uv, z = _project_uv(bg, s.K, (0, 1.5, 0))
    px = uv * [s.K.width, s.K.height] - 0.5
    detail = np.isclose(
        z,
        1.0
        / li.layers[0].inv_z[
            np.clip(np.rint(px[:, 1]).astype(int), 0, s.K.height - 1),
            np.clip(np.rint(px[:, 0]).astype(int), 0, s.K.width - 1),
        ]
        * meshmod.DETAIL_PUSH,
        rtol=1e-5,
    )
    assert detail.sum() > 100
    # detail vertices stay near the hallucinated ring (dilated by one grid cell of the
    # background grid, which is coarser than the front's: BACKGROUND_DENSITY)
    pd = px[detail]
    stride = float(np.median(np.diff(np.unique(np.round(pd[:, 0])))))
    assert stride >= np.sqrt((s.K.width - 1) * (s.K.height - 1) * 2 / 20000.0)
    assert pd[:, 0].min() > 120 - 2 * stride and pd[:, 0].max() < 200 + 2 * stride
    assert pd[:, 1].min() > 80 - 2 * stride and pd[:, 1].max() < 160 + 2 * stride
    core = (
        (pd[:, 0] > 135 + 2 * stride)
        & (pd[:, 0] < 185 - 2 * stride)
        & (pd[:, 1] > 95 + 2 * stride)
        & (pd[:, 1] < 145 - 2 * stride)
    )
    assert not core.any()  # the untouched core of the square is not duplicated


def test_layered_anchor_places_image_centre_and_keeps_perspective(two_plane_scene):
    s = two_plane_scene
    li = _two_layer(s)
    meshes = layered_meshes(li, target_faces=20000, anchor_distance_m=2.0, eye_height_m=1.6)
    fg = meshes[-1]
    scale = 2.0 / s.z_near  # the central window sees the near square
    eye = (0.0, 1.6, 0.0)
    uv, z = _project_uv(fg, s.K, eye, scale)
    np.testing.assert_allclose(uv, fg.uvs, atol=1e-5)  # still a perspective reconstruction from the eye
    assert np.all(np.isclose(z, s.z_near, atol=1e-4) | np.isclose(z, s.z_far, atol=1e-4))
    # the near square (seen at the image centre) is now a plane at z = -anchor that the
    # optical axis (0, eye_height, -t) crosses: the image centre is at (0, eye_height, -anchor)
    near = np.isclose(fg.vertices[:, 2], -2.0, atol=1e-4)
    assert near.sum() >= 4
    nv = fg.vertices[near]
    assert nv[:, 0].min() < 0 < nv[:, 0].max() and nv[:, 1].min() < 1.6 < nv[:, 1].max()
    for m in meshes:
        assert np.all(m.vertices[:, 2] < 0)  # everything in front of the viewer


def test_single_layer_gives_backdrop_and_foreground(two_plane_scene):
    s = two_plane_scene
    li = LayeredImage([Layer(s.img, s.inv_z, np.ones(s.inv_z.shape, np.float32))], s.K)
    meshes = layered_meshes(li, target_faces=16000)
    assert [m.name for m in meshes] == ["backdrop", "foreground"]
    bd = meshes[0]
    assert abs(len(bd.faces) - 1000) <= 150  # target/16 for a full frame, uncut
    assert bd.texture is meshes[1].texture


def test_cleanplate_style_front_matte(two_plane_scene):
    s = two_plane_scene
    alpha = s.fg_mask.astype(np.float32)
    li = _two_layer(s, alpha_front=alpha)
    bg, fg = layered_meshes(li, target_faces=20000)
    assert fg.alpha_mode == "MASK" and fg.texture.shape[2] == 4
    assert np.allclose(-fg.vertices[:, 2], s.z_near, atol=1e-4)  # only the subject
    # the background covers the whole frame where the front is transparent
    uv, _ = _project_uv(bg, s.K, (0, 1.5, 0))
    assert uv[:, 0].min() < 0.02 and uv[:, 0].max() > 0.98


def test_layered_meshes_validation_and_cancel(two_plane_scene):
    s = two_plane_scene
    li = _two_layer(s)
    with pytest.raises(InputError):
        layered_meshes(LayeredImage([], s.K))
    with pytest.raises(InputError):
        layered_meshes(li, anchor_distance_m=0.0)
    bad = LayeredImage([Layer(s.img, s.inv_z[:10], np.ones((10, 320), np.float32))], s.K)
    with pytest.raises(InputError):
        layered_meshes(bad)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        layered_meshes(li, cancel=tok)
    seen = []
    layered_meshes(li, target_faces=5000, progress=lambda f, m: seen.append(f))
    assert seen[-1] == pytest.approx(1.0)


def _total_faces(meshes: list[Mesh]) -> int:
    return int(sum(len(m.faces) for m in meshes))


def _full_band(s) -> LayeredImage:
    """Like the default look-behind band: the background differs from the front on
    (almost) every pixel (98 % in real photos)."""
    bg_inv = (s.inv_z * 0.9).astype(np.float32)  # pushed back everywhere
    bg_rgb = np.clip(s.img.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    ones = np.ones(s.inv_z.shape, np.float32)
    return LayeredImage([Layer(bg_rgb, bg_inv, ones.copy()), Layer(s.img, s.inv_z, ones.copy())], s.K)


@pytest.mark.parametrize("target", [8000, 20000])
def test_layered_total_faces_follow_the_detail_setting(simplify_mode, two_plane_scene, target):
    """E2E-6: "Detail" (target_faces) is the size of the whole scene, not per layer; a
    full look-behind band used to give ~2x the triangles shown in the GUI."""
    s = two_plane_scene
    meshes = layered_meshes(_full_band(s), target_faces=target)
    total = _total_faces(meshes)
    assert 0.75 * target <= total <= 1.05 * target, total
    bg, fg = meshes
    assert len(fg.faces) > len(bg.faces) - target // 16  # the front keeps the larger share


def test_layered_front_keeps_most_of_the_budget_when_little_is_hallucinated(simplify_mode, two_plane_scene):
    s = two_plane_scene
    target = 20000
    bg, fg = layered_meshes(_two_layer(s), target_faces=target)  # thin ring only
    total = len(bg.faces) + len(fg.faces)
    assert total <= 1.05 * target
    if simplify_mode == "qem":
        assert len(fg.faces) >= 0.7 * target


def test_single_layer_and_cleanplate_totals_stay_within_budget(simplify_mode, two_plane_scene):
    s = two_plane_scene
    one = LayeredImage([Layer(s.img, s.inv_z, np.ones(s.inv_z.shape, np.float32))], s.K)
    assert _total_faces(layered_meshes(one, target_faces=16000)) <= 1.02 * 16000
    clean = _two_layer(s, alpha_front=s.fg_mask.astype(np.float32))
    total = _total_faces(layered_meshes(clean, target_faces=20000))
    assert 0.5 * 20000 <= total <= 1.05 * 20000, total


def test_anchor_scale_matches_layered_meshes(two_plane_scene):
    """F3: the pipeline sizes the look-behind band for real head movement with the same
    scale layered_meshes applies (band built for max_baseline / s)."""
    s = two_plane_scene
    li = _two_layer(s)
    sc = meshmod.anchor_scale(s.inv_z, s.K, 2.0)
    assert sc == pytest.approx(2.0 / s.z_near)  # the central window sees the near square
    fg = layered_meshes(li, target_faces=6000, anchor_distance_m=2.0, eye_height_m=0.0)[-1]
    z_scene = -fg.vertices[:, 2].astype(np.float64)
    assert np.isclose(z_scene, s.z_near * sc, atol=1e-4).any()
    assert np.all(
        np.isclose(z_scene, s.z_near * sc, atol=1e-4) | np.isclose(z_scene, s.z_far * sc, atol=1e-4)
    )
    # clean-plate: the matte decides which pixels count
    alpha = np.zeros(s.inv_z.shape, np.float32)
    alpha[:, :40] = 1.0  # only far pixels are opaque -> window empty -> whole-frame fallback
    assert meshmod.anchor_scale(s.inv_z, s.K, 2.0, alpha=alpha) == pytest.approx(2.0 / s.z_far)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(InputError):
            meshmod.anchor_scale(s.inv_z, s.K, bad)
    with pytest.raises(InputError):
        meshmod.anchor_scale(s.inv_z[:10], s.K, 2.0)
    with pytest.raises(MeshError):
        meshmod.anchor_scale(np.zeros_like(s.inv_z), s.K, 2.0)


# --------------------------------------------------------------------------- decimate
@pytest.mark.skipif(not HAS_FS, reason="fast-simplification not installed")
def test_decimate_depth_mesh_reprojects_uvs(two_plane_scene, monkeypatch):
    s = two_plane_scene
    monkeypatch.setattr(meshmod, "_fast_simplification", lambda: None)
    dense = depth_mesh(s.img, s.inv_z, s.K, target_faces=60000)
    monkeypatch.undo()
    out = decimate(dense, 8000)
    assert abs(len(out.faces) - 8000) <= 800
    uv, z = _project_uv(out, s.K)
    np.testing.assert_allclose(uv, out.uvs, atol=1e-4)
    assert _face_spans(out, 2.5) == 0
    assert out.texture is dense.texture and out.name == dense.name


@pytest.mark.skipif(not HAS_FS, reason="fast-simplification not installed")
def test_decimate_arbitrary_uvs_uses_nearest_source_vertex():
    import trimesh

    sph = trimesh.creation.icosphere(subdivisions=4)
    rng = np.random.default_rng(0)
    uvs = rng.random((len(sph.vertices), 2)).astype(np.float32)
    m = Mesh(
        sph.vertices.astype(np.float32),
        sph.faces.astype(np.uint32),
        uvs,
        np.zeros((8, 8, 3), np.uint8),
        normals=sph.vertex_normals.astype(np.float32),
    )
    out = decimate(m, 1000)
    assert abs(len(out.faces) - 1000) <= 100
    # every output UV is one of the input UVs
    key_in = {tuple(x) for x in np.round(uvs, 6)}
    assert all(tuple(x) in key_in for x in np.round(out.uvs, 6))
    assert out.normals is not None
    np.testing.assert_allclose(np.linalg.norm(out.normals, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(out.vertices, axis=1), 1.0, atol=0.05)


def test_decimate_without_library_returns_input(two_plane_scene, monkeypatch, caplog):
    s = two_plane_scene
    m = depth_mesh(s.img, s.inv_z, s.K, target_faces=5000)
    monkeypatch.setattr(meshmod, "_fast_simplification", lambda: None)
    with caplog.at_level(logging.WARNING):
        out = decimate(m, 100)
    assert out is m
    assert "not installed" in caplog.text
    assert decimate(m, 10**6) is m
    with pytest.raises(InputError):
        decimate(m, 0)


def test_vertex_normals_unit_and_oriented():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], np.float32)
    f = np.array([[0, 1, 2]], np.uint32)
    n = vertex_normals(v, f)
    np.testing.assert_allclose(n[:3], [[0, 0, 1]] * 3, atol=1e-6)
    np.testing.assert_allclose(n[3], [0, 0, 1])  # isolated vertex


# --------------------------------------------------------------------------- review additions
def _screen_signed_area(m: Mesh, K: Intrinsics) -> np.ndarray:
    """Signed area of every face in source-camera pixels (u right, v down); < 0 = front-facing."""
    p = gltf_to_cv(m.vertices.astype(np.float64))
    u = K.fx * p[:, 0] / p[:, 2] + K.cx
    v = K.fy * p[:, 1] / p[:, 2] + K.cy
    f = m.faces.astype(np.int64)
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    return (u[b] - u[a]) * (v[c] - v[a]) - (v[b] - v[a]) * (u[c] - u[a])


@pytest.mark.parametrize("noise", [0.0, 0.05])
def test_every_face_is_front_facing_in_screen_space(simplify_mode, noise):
    """No fold-overs, also after quadric simplification of a noisy depth map."""
    h, w = 120, 160
    K = Intrinsics.from_hfov(w, h, 60.0)
    inv = _smooth_depth(h, w) + np.random.default_rng(0).normal(0, noise, (h, w)).astype(np.float32)
    m = depth_mesh(np.zeros((h, w, 3), np.uint8), inv, K, target_faces=5000)
    area = _screen_signed_area(m, K)
    assert np.all(area < 0), f"{int((area >= 0).sum())} folded faces"
    assert len(m.faces) >= 0.85 * 5000


def test_anisotropic_off_centre_intrinsics_round_trip(two_plane_scene, simplify_mode):
    """UVs stay the exact source projection for fx != fy and an off-centre principal point."""
    s = two_plane_scene
    K = Intrinsics(s.K.fx * 1.1, s.K.fy * 0.9, s.K.cx + 13.0, s.K.cy - 7.5, s.K.width, s.K.height)
    m = depth_mesh(s.img, s.inv_z, K, target_faces=8000)
    uv, z = _project_uv(m, K)
    np.testing.assert_allclose(uv, m.uvs, atol=1e-5)
    assert np.all(np.isclose(z, s.z_near, atol=1e-4) | np.isclose(z, s.z_far, atol=1e-4))
    np.testing.assert_allclose(m.uvs.min(0), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(m.uvs.max(0), [1.0, 1.0], atol=1e-6)
    # the principal ray maps to the -z axis: a vertex at (cx, cy) would have x = y = 0
    p = gltf_to_cv(m.vertices.astype(np.float64))
    left = (K.fx * p[:, 0] / p[:, 2] + K.cx) < K.cx
    assert np.all(m.vertices[left, 0] < 0) and np.all(m.vertices[~left, 0] >= -1e-6)
    up = (K.fy * p[:, 1] / p[:, 2] + K.cy) < K.cy  # image rows above the principal point
    assert np.all(m.vertices[up, 1] > 0)  # ... are above the optical axis (glTF y up)


@pytest.mark.parametrize("size", [(2, 2), (3, 2), (2, 5), (7, 3)])
def test_tiny_images_are_meshed(size):
    w, h = size
    K = Intrinsics.from_hfov(w, h, 60.0)
    m = depth_mesh(np.zeros((h, w, 3), np.uint8), np.full((h, w), 0.5, np.float32), K, target_faces=1000)
    assert len(m.faces) == 2 * (w - 1) * (h - 1)
    np.testing.assert_allclose(m.uvs.min(0), 0.0, atol=1e-6)
    np.testing.assert_allclose(m.uvs.max(0), 1.0, atol=1e-6)


def test_matte_made_only_of_small_pieces_is_kept(two_plane_scene, monkeypatch):
    """The debris filter never deletes everything (a small subject is not 'debris')."""
    monkeypatch.setattr(meshmod, "_fast_simplification", lambda: None)
    s = two_plane_scene
    alpha = np.zeros(s.inv_z.shape, np.float32)
    alpha[10:14, 10:14] = 1.0  # 18 faces at stride 1 (< 50)
    m = depth_mesh(s.img, s.inv_z, s.K, alpha=alpha, target_faces=300_000)
    assert 0 < len(m.faces) < 50
    assert m.alpha_mode == "MASK"


def test_target_faces_accepts_integral_floats(two_plane_scene):
    s = two_plane_scene
    a = depth_mesh(s.img, s.inv_z, s.K, target_faces=4000.0)
    b = depth_mesh(s.img, s.inv_z, s.K, target_faces=np.int64(4000))
    assert len(a.faces) == len(b.faces)
    for bad in (4000.5, float("nan"), True, "4000"):
        with pytest.raises(InputError):
            depth_mesh(s.img, s.inv_z, s.K, target_faces=bad)
    with pytest.raises(InputError):
        depth_mesh(s.img, s.inv_z, s.K, cut_px="2")  # type: ignore[arg-type]


def test_uint16_image_is_rescaled_not_saturated(two_plane_scene):
    s = two_plane_scene
    img16 = s.img.astype(np.uint16) * 257
    m = depth_mesh(img16, s.inv_z, s.K, target_faces=2000)
    np.testing.assert_array_equal(m.texture, s.img)


def test_inverse_depth_sign_larger_is_nearer(two_plane_scene):
    """Pins the convention: the square (larger inv_z) is nearer the camera (less negative z)."""
    s = two_plane_scene
    m = depth_mesh(s.img, s.inv_z, s.K, target_faces=8000)
    uv, _ = _project_uv(m, s.K)
    px = np.clip(np.rint(uv * [s.K.width, s.K.height] - 0.5).astype(int), 0, [s.K.width - 1, s.K.height - 1])
    on_square = s.fg_mask[px[:, 1], px[:, 0]]
    assert np.all(m.vertices[on_square, 2] > m.vertices[~on_square, 2].max())
    np.testing.assert_allclose(-m.vertices[on_square, 2], s.z_near, atol=1e-4)
