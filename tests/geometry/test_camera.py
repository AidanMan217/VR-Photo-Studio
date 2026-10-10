"""Tests for geometry.camera: (un)projection, frame conversion, intrinsics estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vrphoto.core.errors import InputError, VRPhotoError
from vrphoto.core.types import DepthResult, Intrinsics
from vrphoto.geometry.camera import (
    cv_to_gltf,
    estimate_intrinsics,
    gltf_to_cv,
    pixel_grid,
    project,
    unproject,
)


@pytest.fixture
def K() -> Intrinsics:
    return Intrinsics(fx=300.0, fy=310.0, cx=159.5, cy=119.0, width=320, height=240)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unproject_project_roundtrip(K, dtype):
    rng = np.random.default_rng(3)
    z = rng.uniform(0.5, 20.0, size=(K.height, K.width)).astype(dtype)
    P = unproject(K, z)
    assert P.shape == (K.height, K.width, 3)
    assert P.dtype == dtype
    np.testing.assert_allclose(P[..., 2], z, rtol=1e-6)
    uv = project(K, P)
    assert uv.shape == (K.height, K.width, 2)
    u, v = pixel_grid(K, np.float64)
    tol = 2e-3 if dtype == np.float32 else 1e-9
    np.testing.assert_allclose(uv[..., 0], np.broadcast_to(u, z.shape), atol=tol)
    np.testing.assert_allclose(uv[..., 1], np.broadcast_to(v, z.shape), atol=tol)


def test_unproject_known_values(K):
    z = np.full((K.height, K.width), 2.0, np.float32)
    P = unproject(K, z)
    # pixel (row 119, col 159.5 is between pixels) -> use pixel (119, 159): X = (159-159.5)/300*2
    np.testing.assert_allclose(P[119, 159], [(159 - 159.5) / 300.0 * 2, 0.0, 2.0], atol=1e-6)
    # y down: a pixel below the principal point has positive Y
    assert P[200, 100, 1] > 0
    # x right: a pixel right of the principal point has positive X
    assert P[10, 300, 0] > 0


def test_project_conventions_and_invalid(K):
    P = np.array(
        [
            [0.0, 0.0, 2.0],  # on axis -> principal point
            [1.0, 0.5, 2.0],  # right and down
            [0.0, 0.0, 0.0],  # on the camera plane -> NaN
            [1.0, 1.0, -3.0],  # behind -> NaN
            [0.0, 0.0, 5e-7],  # below the 1e-6 threshold -> NaN
        ]
    )
    uv = project(K, P)
    np.testing.assert_allclose(uv[0], [K.cx, K.cy])
    np.testing.assert_allclose(uv[1], [K.cx + 300.0 * 0.5, K.cy + 310.0 * 0.25])
    assert np.isnan(uv[2:]).all()
    assert uv.dtype == np.float64


def test_project_preserves_batch_shape(K):
    P = np.ones((4, 5, 6, 3), np.float32)
    assert project(K, P).shape == (4, 5, 6, 2)
    assert project(K, P).dtype == np.float32


def test_unproject_shape_mismatch(K):
    with pytest.raises(InputError):
        unproject(K, np.ones((10, 10), np.float32))
    with pytest.raises(InputError):
        unproject(K, np.ones((K.height, K.width, 1), np.float32))


def test_project_bad_shape(K):
    with pytest.raises(InputError):
        project(K, np.ones((5, 2)))


def test_cv_gltf_conversion():
    P = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    G = cv_to_gltf(P)
    np.testing.assert_array_equal(G, [[1.0, -2.0, -3.0], [-1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(gltf_to_cv(G), P)
    # camera looks down +z in cv == -z in glTF; image "down" (+y cv) is -y in glTF
    assert cv_to_gltf(np.array([0.0, 0.0, 5.0]))[2] == -5.0
    assert cv_to_gltf(np.array([0.0, 1.0, 0.0]))[1] == -1.0
    f32 = np.ones((7, 3), np.float32)
    assert cv_to_gltf(f32).dtype == np.float32
    assert cv_to_gltf(np.ones((2, 3), np.int64)).dtype == np.float32
    with pytest.raises(InputError):
        cv_to_gltf(np.ones((3, 2)))


def test_estimate_intrinsics_priority():
    shape = (480, 640, 3)
    moge = Intrinsics.from_hfov(320, 240, 50.0)  # model ran at half resolution
    depth = DepthResult(np.zeros((480, 640), np.float32), "moge2", intrinsics=moge)

    k = estimate_intrinsics(shape, hfov_deg=80.0, exif_hfov=70.0, depth=depth)
    assert k.hfov_deg == pytest.approx(80.0)
    k = estimate_intrinsics(shape, exif_hfov=70.0, depth=depth)
    assert k.hfov_deg == pytest.approx(50.0)
    assert (k.width, k.height) == (640, 480)  # rescaled to the image
    assert k.cx == pytest.approx(320.0)
    k = estimate_intrinsics(shape, exif_hfov=70.0, depth=DepthResult(np.zeros((480, 640)), "da2"))
    assert k.hfov_deg == pytest.approx(70.0)
    k = estimate_intrinsics(shape)
    assert k.hfov_deg == pytest.approx(65.0)
    assert k == Intrinsics.from_hfov(640, 480, 65.0)
    k = estimate_intrinsics((480, 640), default_hfov=55.0)
    assert k.hfov_deg == pytest.approx(55.0)


def test_estimate_intrinsics_portrait_and_square_pixels():
    k = estimate_intrinsics((1000, 600), hfov_deg=45.0)
    assert (k.width, k.height) == (600, 1000)
    assert k.fx == k.fy
    assert k.fx == pytest.approx(300.0 / math.tan(math.radians(22.5)))


def test_estimate_intrinsics_invalid_values():
    with pytest.raises(InputError):
        estimate_intrinsics((480, 640), hfov_deg=0.0)
    with pytest.raises(InputError):
        estimate_intrinsics((480, 640), hfov_deg=200.0)
    with pytest.raises(VRPhotoError):
        estimate_intrinsics((480,))
    with pytest.raises(InputError):
        estimate_intrinsics((0, 640))
    # implausible EXIF is ignored, not fatal
    k = estimate_intrinsics((480, 640), exif_hfov=float("nan"))
    assert k.hfov_deg == pytest.approx(65.0)
    k = estimate_intrinsics((480, 640), exif_hfov=0.2)
    assert k.hfov_deg == pytest.approx(65.0)
    # broken model intrinsics fall through to EXIF
    bad = DepthResult(np.zeros((480, 640)), "m", intrinsics=Intrinsics(0.0, 0.0, 0.0, 0.0, 640, 480))
    assert estimate_intrinsics((480, 640), exif_hfov=60.0, depth=bad).hfov_deg == pytest.approx(60.0)


def test_image_up_is_gltf_up_and_forward_is_minus_z(K):
    """Pins the frame chain: a pixel ABOVE the principal point (smaller v, y-down cv
    frame) ends up ABOVE the camera (+y) in glTF, and depth goes to -z."""
    z = np.full((K.height, K.width), 2.0, np.float32)
    P = unproject(K, z)
    G = cv_to_gltf(P)
    top, bottom = G[5, 160], G[230, 160]
    assert top[1] > 0 > bottom[1]
    assert (G[..., 2] == -2.0).all()
    right = G[120, 310]
    assert right[0] > 0  # x stays right
    # left eye convention: a camera at x = -B/2 sees a point straight ahead to its right
    uv = project(K, (P[119, 160] - np.array([-0.032, 0.0, 0.0]))[None])
    assert uv[0, 0] > 160


def test_estimate_intrinsics_rejects_non_numbers():
    with pytest.raises(InputError):
        estimate_intrinsics((480, 640), hfov_deg="wide")  # type: ignore[arg-type]
    with pytest.raises(InputError):
        estimate_intrinsics(None)  # type: ignore[arg-type]
    with pytest.raises(InputError):
        estimate_intrinsics((480, 640), default_hfov="x")  # type: ignore[arg-type]
    # unparsable EXIF is ignored like an implausible one
    k = estimate_intrinsics((480, 640), exif_hfov="n/a")  # type: ignore[arg-type]
    assert k.hfov_deg == pytest.approx(65.0)
    # numpy scalars and shapes are fine
    k = estimate_intrinsics(np.zeros((48, 64, 3)).shape, hfov_deg=np.float32(70.0))
    assert (k.width, k.height) == (64, 48) and k.hfov_deg == pytest.approx(70.0, abs=1e-4)


def test_default_fov_applies_to_the_long_side_for_portrait_photos():
    """R7: an EXIF-less portrait photo is ~65 deg TALL (like the same camera held
    sideways), not 65 deg wide x 87 deg tall."""
    land = estimate_intrinsics((800, 1200, 3))
    port = estimate_intrinsics((1200, 800, 3))
    assert land.hfov_deg == pytest.approx(65.0)
    assert port.vfov_deg == pytest.approx(65.0)
    assert port.hfov_deg == pytest.approx(land.vfov_deg)  # rotated landscape camera
    assert port.fx == port.fy == pytest.approx(land.fx)  # same focal length in pixels
    assert (port.cx, port.cy, port.width, port.height) == (400.0, 600.0, 800, 1200)
    sq = estimate_intrinsics((500, 500))
    assert sq.hfov_deg == pytest.approx(65.0) and sq.vfov_deg == pytest.approx(65.0)
    # explicit and EXIF values stay horizontal fields of view of the displayed image
    assert estimate_intrinsics((1200, 800), hfov_deg=50.0).hfov_deg == pytest.approx(50.0)
    assert estimate_intrinsics((1200, 800), exif_hfov=50.0).hfov_deg == pytest.approx(50.0)
    assert estimate_intrinsics((1200, 800), default_hfov=55.0).vfov_deg == pytest.approx(55.0)
