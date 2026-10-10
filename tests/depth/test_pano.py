"""Tests for depth.pano.estimate_equirect with fake perspective estimators."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InferenceError, InputError
from vrphoto.core.types import CancelToken, DepthResult, ProgressReporter
from vrphoto.depth.pano import estimate_equirect
from vrphoto.geometry.projection import CUBE_FACES, equirect_dirs

W, H = 1024, 512


def smooth_truth() -> np.ndarray:
    """A smooth, non-trivial function of direction in [0, 1]."""
    d = equirect_dirs(W, H).astype(np.float64)
    lam = np.arctan2(d[..., 0], d[..., 2])
    t = 0.5 + 0.3 * d[..., 0] + 0.2 * d[..., 1] * d[..., 2] + 0.1 * np.sin(3 * lam) * np.sqrt(1 - d[..., 1] ** 2)
    return (t - t.min()) / (t.max() - t.min())


def encode(v: np.ndarray) -> np.ndarray:
    return np.repeat(np.rint(v * 255).astype(np.uint8)[..., None], 3, axis=2)


class ImageFake:
    """Disparity = per-call random affine of the red channel (a radial quantity)."""

    id = "fake-img"

    def __init__(self, out_size: int | None = None):
        self.calls: list[np.ndarray] = []
        self.out_size = out_size

    def estimate(self, img, *, progress=None, cancel=None):
        self.calls.append(img)
        rng = np.random.default_rng(len(self.calls))
        a, b = rng.uniform(0.5, 2.0), rng.uniform(-0.3, 0.3)
        d = (a * img[..., 0].astype(np.float32) / 255.0 + b).astype(np.float32)
        if self.out_size:
            import cv2

            d = cv2.resize(d, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)
        if progress is not None:
            progress(1.0, "face")
        return DepthResult(disparity=d, model_id="x")


class PlanarFake:
    """Behaves like a real perspective estimator: inverse PLANAR depth (1/z), min-max normalised.

    The equirect red channel encodes inverse distance; the fake converts it to 1/z with the
    known face FOV (inverse distance / cos(theta))."""

    id = "fake-z"

    def __init__(self, fov_deg: float):
        self.fov = fov_deg

    def estimate(self, img, *, progress=None, cancel=None):
        n = img.shape[0]
        f = (n / 2) / math.tan(math.radians(self.fov) / 2)
        c = (n - 1) / 2
        a = (np.arange(n) - c) / f
        cos_t = 1 / np.sqrt(1 + a[None, :] ** 2 + a[:, None] ** 2)
        inv_dist = img[..., 0].astype(np.float64) / 255.0 * 0.9 + 0.1
        inv_z = inv_dist / cos_t
        d = (inv_z - inv_z.min()) / (inv_z.max() - inv_z.min())
        return DepthResult(disparity=d.astype(np.float32), model_id="x")


def affine_error(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    sel = (est > 0.002) & (est < 0.998)  # ignore percentile-clipped extremes
    A = np.stack([est[sel], np.ones(sel.sum())], 1).astype(np.float64)
    coef = np.linalg.lstsq(A, truth[sel], rcond=None)[0]
    err = np.zeros_like(truth)
    err[sel] = A @ coef - truth[sel]
    return err


def seam_columns() -> list[int]:
    # cube edges of the side faces on the equator: lon = -135, -45, 45, 135 deg
    return [int(round((lon + 180) / 360 * W)) for lon in (-135, -45, 45, 135)]


def test_seamless_with_per_face_affine():
    truth = smooth_truth()
    fake = ImageFake()
    res = estimate_equirect(encode(truth), fake, face=128)
    assert isinstance(res, DepthResult)
    assert res.model_id == "pano:fake-img"
    d = res.disparity
    assert d.shape == (H, W) and d.dtype == np.float32
    assert d.min() >= 0.0 and d.max() <= 1.0 and d.max() - d.min() > 0.9
    assert len(fake.calls) == 6 and all(c.shape == (128, 128, 3) for c in fake.calls)
    err = affine_error(d.astype(np.float64), truth)
    assert np.abs(err).max() < 0.03
    # no jumps across face seams: neighbour differences there look like everywhere else
    dx = np.abs(np.diff(d.astype(np.float64), axis=1))
    typical = np.percentile(dx[H // 4 : 3 * H // 4], 99.5)
    rows = slice(H // 4, 3 * H // 4)
    for c in seam_columns():
        assert dx[rows, c - 3 : c + 3].max() < max(0.01, 2.5 * typical)
    dy = np.abs(np.diff(d.astype(np.float64), axis=0))
    typical_y = np.percentile(dy, 99.5)
    for r in (H // 4, 3 * H // 4):  # lat +-45 deg: top/bottom face edges near lon 0
        assert dy[r - 3 : r + 3, W // 2 - 50 : W // 2 + 50].max() < max(0.01, 2.5 * typical_y)


def test_unaligned_faces_fail_the_checks(monkeypatch):
    """The checks above are sensitive: skipping the affine alignment breaks them."""
    import vrphoto.depth.pano as pano_mod

    monkeypatch.setattr(pano_mod, "_fit", lambda model, samples, area, ref: (np.tile([1.0, 0.0], (6, 1)), 1.0))
    truth = smooth_truth()
    res = estimate_equirect(encode(truth), ImageFake(), face=128)
    assert np.abs(affine_error(res.disparity.astype(np.float64), truth)).max() > 0.05


def test_planar_depth_estimator_uses_radial_model(caplog):
    d = equirect_dirs(W, H).astype(np.float64)
    lo = np.array([-2.0, -1.2, -4.0])
    hi = np.array([3.0, 1.5, 2.5])  # off-centre box room
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hi = np.where(d > 0, hi / np.where(d > 0, d, 1), np.inf)
        t_lo = np.where(d < 0, lo / np.where(d < 0, d, 1), np.inf)
    dist = np.minimum(t_hi, t_lo).min(-1)
    inv = 1 / dist
    invn = (inv - inv.min()) / (inv.max() - inv.min())
    with caplog.at_level(logging.INFO, logger="vrphoto"):
        res = estimate_equirect(encode(invn), PlanarFake(100.0), face=128, overlap_deg=10.0)
    assert any("radial alignment" in r.message for r in caplog.records)
    err = affine_error(res.disparity.astype(np.float64), invn)
    assert np.abs(err).max() < 0.03
    # nearest wall (floor below the viewer) is brighter than the far wall behind
    assert res.disparity[H - 5, W // 2] > res.disparity[H // 2, 5]


def test_faces_are_rendered_in_cube_order(checker_equirect):
    fake = ImageFake()
    estimate_equirect(checker_equirect, fake, face=96)
    plusz = fake.calls[CUBE_FACES.index("+z")]
    m = (plusz[..., 0] > 200) & (plusz[..., 1] < 70)
    ys, xs = np.nonzero(m)
    assert xs.mean() == pytest.approx(47.5, abs=0.75) and ys.mean() == pytest.approx(47.5, abs=0.75)
    for k in ("+x", "-x", "+y", "-y", "-z"):
        f = fake.calls[CUBE_FACES.index(k)]
        assert not ((f[..., 0] > 200) & (f[..., 1] < 70)).any()


def test_estimator_output_size_and_nans_handled():
    truth = smooth_truth()

    class NanFake(ImageFake):
        def estimate(self, img, *, progress=None, cancel=None):
            r = super().estimate(img)
            r.disparity[:3, :3] = np.nan
            return r

    res = estimate_equirect(encode(truth), ImageFake(out_size=50), face=100)
    assert res.disparity.shape == (H, W)
    assert np.abs(affine_error(res.disparity.astype(np.float64), truth)).max() < 0.05
    res2 = estimate_equirect(encode(truth), NanFake(), face=64)
    assert np.isfinite(res2.disparity).all()


def test_zero_overlap_still_covers_sphere():
    truth = smooth_truth()
    res = estimate_equirect(encode(truth), ImageFake(), face=64, overlap_deg=0.0)
    assert np.isfinite(res.disparity).all() and res.disparity.std() > 0.1


def test_validation_and_errors():
    eq = np.zeros((100, 300, 3), np.uint8)
    with pytest.raises(InputError, match="2:1"):
        estimate_equirect(eq, ImageFake())
    eq2 = np.zeros((64, 128, 3), np.uint8)
    with pytest.raises(InputError):
        estimate_equirect(eq2, ImageFake(), overlap_deg=90)
    with pytest.raises(InputError):
        estimate_equirect(eq2, ImageFake(), face=4)

    class Broken:
        id = "broken"

        def estimate(self, img, *, progress=None, cancel=None):
            return DepthResult(disparity=np.full(img.shape[:2], np.nan, np.float32), model_id="b")

    with pytest.raises(InferenceError):
        estimate_equirect(eq2, Broken(), face=32)


def test_progress_and_cancel():
    truth = smooth_truth()
    seen: list[float] = []
    estimate_equirect(encode(truth), ImageFake(), face=48, progress=ProgressReporter(lambda f, m: seen.append(f)))
    assert seen[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))

    tok = CancelToken()

    class CancelAfterTwo(ImageFake):
        def estimate(self, img, *, progress=None, cancel=None):
            if len(self.calls) == 1:
                tok.cancel()
            return super().estimate(img)

    fake = CancelAfterTwo()
    with pytest.raises(Cancelled):
        estimate_equirect(encode(truth), fake, face=48, cancel=tok)
    assert len(fake.calls) == 2


# --------------------------------------------------------------------------- review additions
def test_input_dtypes_and_channels_are_coerced():
    """float [0,1], uint16, gray and RGBA panoramas give the same faces as uint8 RGB."""
    truth = smooth_truth()
    ref_img = encode(truth)
    ref = ImageFake()
    r0 = estimate_equirect(ref_img, ref, face=48)
    variants = {
        "float01": ref_img.astype(np.float32) / 255.0,
        "uint16": ref_img.astype(np.uint16) * 257,
        "gray": ref_img[..., 0],
        "rgba": np.dstack([ref_img, np.full(ref_img.shape[:2], 255, np.uint8)]),
    }
    for name, img in variants.items():
        fake = ImageFake()
        r = estimate_equirect(img, fake, face=48)
        for a, b in zip(ref.calls, fake.calls):
            assert b.dtype == np.uint8 and b.shape == (48, 48, 3), name
            assert np.abs(a.astype(int) - b.astype(int)).max() <= 1, name
        assert np.abs(r.disparity - r0.disparity).max() < 0.02, name


def test_tiny_panorama_still_aligns():
    """The alignment grid does not depend on the equirect size (8x16 used to fail)."""
    rng = np.random.default_rng(3)
    res = estimate_equirect(rng.integers(0, 255, (8, 16, 3), dtype=np.uint8), ImageFake(), face=16)
    assert res.disparity.shape == (8, 16) and np.isfinite(res.disparity).all()


def test_estimator_may_return_bare_array_and_params_validated():
    class ArrayFake:
        id = "arr"

        def estimate(self, img, *, progress=None, cancel=None):
            return img[..., 0].astype(np.float32) / 255.0

    truth = smooth_truth()
    res = estimate_equirect(encode(truth), ArrayFake(), face=64)
    assert res.model_id == "pano:arr"
    assert np.abs(affine_error(res.disparity.astype(np.float64), truth)).max() < 0.05
    eq = np.zeros((64, 128, 3), np.uint8)
    with pytest.raises(InputError):
        estimate_equirect(eq, ArrayFake(), face="large")
    with pytest.raises(InputError):
        estimate_equirect(eq, ArrayFake(), overlap_deg="some")
    with pytest.raises(InputError):
        estimate_equirect(eq, ArrayFake(), overlap_deg=float("nan"))
    with pytest.raises(InputError):
        estimate_equirect([[0]], ArrayFake())


def test_large_panorama_prefiltered_once_and_faces_unchanged(monkeypatch):
    """Faces rendered from a once-prefiltered source equal direct per-face renders."""
    import vrphoto.depth.pano as pano_mod
    from vrphoto.geometry import projection as proj

    rng = np.random.default_rng(5)
    eq = rng.integers(0, 255, (1024, 2048, 3), dtype=np.uint8)
    calls = []
    real = proj._fit_equirect_source

    def counting(src, max_ppr, span):
        out = real(src, max_ppr, span)
        calls.append((src.shape, out.shape))
        return out

    monkeypatch.setattr(pano_mod, "_fit_equirect_source", counting)
    monkeypatch.setattr(proj, "_fit_equirect_source", counting)
    fake = ImageFake()
    estimate_equirect(eq, fake, face=32)
    resized = [c for c in calls if c[0] != c[1]]
    assert len(resized) == 1  # the 2048-wide panorama is area-downscaled exactly once
    fov = 90.0 + max(10.0, 1.0, 4.0 * 90.0 / 32)
    K = proj._face_K(32, fov)
    for k, img in zip(CUBE_FACES, fake.calls):
        direct = proj._equirect_to_persp_R(eq, K, proj.cube_face_rotation(k))
        assert np.array_equal(img, direct), k
