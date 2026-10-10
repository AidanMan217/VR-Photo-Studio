"""Tests for geometry.ods: omnidirectional stereo from a mono equirect + distance map."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, ProgressReporter
from vrphoto.geometry import ods as O

RED = (255, 0, 0)
GREEN = (0, 200, 0)


def checker(h: int, w: int, cell: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((xx // cell) + (yy // cell)) % 2 * 200 + 30).astype(np.uint8)
    return np.stack([c, c, c], -1)


def red_mask(img: np.ndarray) -> np.ndarray:
    return (img[..., 0] > 200) & (img[..., 1] < 70) & (img[..., 2] < 70)


def weighted_centroid_x(img: np.ndarray, rows: slice) -> float:
    """Sub-pixel x centroid of the red marker (redness-weighted)."""
    sub = img[rows].astype(np.float64)
    wgt = np.clip(sub[..., 0] - sub[..., 1], 0, None)
    wgt[wgt < 60] = 0
    xs = np.arange(img.shape[1])[None, :]
    return float((wgt * xs).sum() / wgt.sum())


@pytest.mark.parametrize("D", [0.5, 1.0, 3.0])
def test_point_straight_ahead_shift(D):
    h, w = 1024, 2048
    eq = checker(h, w)
    eq[h // 2 - 6 : h // 2 + 6, w // 2 - 6 : w // 2 + 6] = RED  # centred on lon 0, lat 0
    dist = np.full((h, w), D, np.float32)
    ipd = 0.064
    left, right = O.ods_pair(eq, dist, ipd)
    assert left.shape == eq.shape and right.shape == eq.shape
    assert left.dtype == np.uint8 and right.dtype == np.uint8
    expected = math.asin((ipd / 2) / D) * w / (2 * math.pi)
    rows = slice(h // 2 - 4, h // 2 + 4)
    x0 = (w - 1) / 2
    xl = weighted_centroid_x(left, rows)
    xr = weighted_centroid_x(right, rows)
    assert xl - x0 == pytest.approx(expected, abs=1.0)  # RIGHT in the left eye
    assert xr - x0 == pytest.approx(-expected, abs=1.0)  # LEFT in the right eye


def test_displacement_formula_and_signs():
    h, w = 180, 360
    D = 2.0
    dist = np.full((h, w), D, np.float32)
    dx, dy = O.ods_displacement(dist, 0.064)
    assert dx.dtype == np.float32 and dx.shape == (h, w)
    r = 0.032
    for j in (90, 60, 120, 30):
        phi = math.radians(90 - (j + 0.5) / h * 180)
        wgt = min(1.0, max(0.0, (75 - abs(math.degrees(phi))) / 30))
        rr = r * wgt
        rho, y = D * math.cos(phi), D * math.sin(phi)
        exp_dx = math.asin(min(rr / rho, 0.99)) * w / (2 * math.pi)
        exp_phi = math.atan2(y, math.sqrt(max(rho * rho - rr * rr, 1e-12)))
        exp_dy = -(exp_phi - phi) * h / math.pi
        assert dx[j, 17] == pytest.approx(exp_dx, abs=1e-4)
        assert dy[j, 17] == pytest.approx(exp_dy, abs=1e-4)
    assert (dx >= 0).all()
    assert (dy[: h // 2 - 1][dx[: h // 2 - 1] > 0] < 0).all()  # upper hemisphere moves up
    assert (dy[h // 2 + 1 :][dx[h // 2 + 1 :] > 0] > 0).all()  # lower hemisphere moves down
    # pole fade: zero beyond 75 deg, full parallax below 45 deg
    lat = 90 - (np.arange(h) + 0.5) / h * 180
    assert (dx[np.abs(lat) >= 75] == 0).all() and (dy[np.abs(lat) >= 75] == 0).all()
    assert (dx[np.abs(lat) < 45] > 0).all()


def test_poles_have_zero_parallax():
    h, w = 256, 512
    rng = np.random.default_rng(0)
    eq = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    dist = rng.uniform(0.8, 5.0, (h, w)).astype(np.float32)
    left, right = O.ods_pair(eq, dist, 0.064)
    lat = 90 - (np.arange(h) + 0.5) / h * 180
    pole = np.abs(lat) > 76
    assert np.array_equal(left[pole], eq[pole])
    assert np.array_equal(right[pole], eq[pole])
    assert not np.array_equal(left[np.abs(lat) < 30], eq[np.abs(lat) < 30])


def test_custom_pole_fade():
    dist = np.full((90, 180), 1.0, np.float32)
    dx, _ = O.ods_displacement(dist, 0.064, pole_fade=(10.0, 20.0))
    lat = 90 - (np.arange(90) + 0.5) / 90 * 180
    assert (dx[np.abs(lat) >= 20] == 0).all() and (dx[np.abs(lat) < 10] > 0).all()
    with pytest.raises(InputError):
        O.ods_displacement(dist, 0.064, pole_fade=(50.0, 40.0))


def test_zero_ipd_and_infinite_distance_are_identity():
    h, w = 64, 128
    eq = checker(h, w, 8)
    left, right = O.ods_pair(eq, np.full((h, w), 2.0, np.float32), 0.0)
    assert np.array_equal(left, eq) and np.array_equal(right, eq)
    far = np.full((h, w), np.inf, np.float32)
    far[:, :10] = 0.0  # invalid -> treated as infinitely far
    far[:, 10:20] = np.nan
    left, right = O.ods_pair(eq, far, 0.064)
    assert np.array_equal(left, eq) and np.array_equal(right, eq)


def _near_box_scene(h=512, w=1024, D_near=0.6):
    eq = checker(h, w, 16)
    dist = np.full((h, w), 20.0, np.float32)
    r0, r1, c0, c1 = 200, 312, 450, 575
    dist[r0:r1, c0:c1] = D_near
    eq[r0:r1, c0:c1] = GREEN
    return eq, dist, (r0, r1, c0, c1)


def test_disocclusion_filled_from_background():
    eq, dist, (r0, r1, c0, c1) = _near_box_scene()
    w = eq.shape[1]
    left, right = O.ods_pair(eq, dist, 0.064)
    shift = math.asin(0.032 / 0.6) * w / (2 * math.pi)  # ~8.7 px
    rows = slice(r0 + 10, r1 - 10)
    green_l = (left[rows, :, 1] > 150) & (left[rows, :, 0] < 60)
    green_r = (right[rows, :, 1] > 150) & (right[rows, :, 0] < 60)
    # the box moved right in the left eye and left in the right eye
    cl = np.nonzero(green_l.any(0))[0]
    cr = np.nonzero(green_r.any(0))[0]
    assert cl.min() == pytest.approx(c0 + shift, abs=2) and cr.max() == pytest.approx(c1 - 1 - shift, abs=2)
    # the uncovered strip (left of the box in the left eye) is background, not green smear
    gap_l = left[rows, c0 + 1 : int(c0 + shift) - 1].astype(int)
    gap_r = right[rows, int(c1 - shift) + 1 : c1 - 1].astype(int)
    for gap in (gap_l, gap_r):
        assert gap.size > 0
        greenish = gap[..., 1] - (gap[..., 0] + gap[..., 2]) / 2
        assert greenish.max() < 25
        assert np.abs(gap[..., 0] - gap[..., 2]).max() < 25


class RecordingInpainter:
    id = "fake"

    def __init__(self):
        self.calls = []

    def inpaint(self, img, mask, *, progress=None, cancel=None):
        self.calls.append((img.shape, mask.copy()))
        out = img.copy()
        out[mask] = (255, 0, 255)
        return out


def test_inpainter_object_and_callable_used_for_holes():
    eq, dist, (r0, r1, c0, c1) = _near_box_scene()
    inp = RecordingInpainter()
    left, right = O.ods_pair(eq, dist, 0.064, inpainter=inp)
    assert len(inp.calls) == 2
    shape, mask = inp.calls[0]
    assert shape[0] == eq.shape[0] and shape[1] > eq.shape[1]  # wrap-padded
    assert mask.dtype == bool and mask.any()
    rows = slice(r0 + 10, r1 - 10)
    magenta = (left[rows, :, 0] > 200) & (left[rows, :, 2] > 200) & (left[rows, :, 1] < 60)
    assert magenta[:, c0 : c0 + 8].any()
    # pixels far from any disocclusion are untouched by the inpainter
    assert not ((left[:, :300, 0] > 200) & (left[:, :300, 1] < 60) & (left[:, :300, 2] > 200)).any()

    seen = []

    def fn(img, mask):
        seen.append(mask.sum())
        out = img.copy()
        out[mask] = 7
        return out

    O.ods_pair(eq, dist, 0.064, inpainter=fn)
    assert len(seen) == 2 and all(s > 0 for s in seen)


def test_longitude_wrap_at_seam():
    h, w = 512, 1024
    eq = checker(h, w)
    eq[h // 2 - 6 : h // 2 + 6, :6] = RED  # marker straddles the +-180 deg seam
    eq[h // 2 - 6 : h // 2 + 6, w - 6 :] = RED
    D = 0.5
    left, right = O.ods_pair(eq, np.full((h, w), D, np.float32), 0.064)
    shift = math.asin(0.032 / D) * w / (2 * math.pi)
    rows = slice(h // 2 - 4, h // 2 + 4)
    for img, sign in ((left, 1), (right, -1)):
        rolled = np.roll(img, w // 2, axis=1)  # bring the seam to the middle
        x = weighted_centroid_x(rolled, rows)
        assert x - (w - 1) / 2 == pytest.approx(sign * shift, abs=1.0)
        assert not red_mask(img[:, 40 : w - 40]).any()


def test_vertical_parallax_toward_pole_is_equal_in_both_eyes():
    h, w = 720, 1440
    eq = checker(h, w)
    j = int(round((90 - 30) / 180 * h))  # latitude ~30 deg up
    eq[j - 5 : j + 5, w // 2 - 5 : w // 2 + 5] = RED
    D = 0.15
    left, right = O.ods_pair(eq, np.full((h, w), D, np.float32), 0.064)
    yl = np.nonzero(red_mask(left))[0].mean()
    yr = np.nonzero(red_mask(right))[0].mean()
    y0 = np.nonzero(red_mask(eq))[0].mean()
    phi = math.radians(90 - (y0 + 0.5) / h * 180)
    rho, y = D * math.cos(phi), D * math.sin(phi)
    phi2 = math.atan2(y, math.sqrt(rho * rho - 0.032**2))
    expected = (phi2 - phi) * h / math.pi  # rows upward
    assert y0 - yl == pytest.approx(expected, abs=1.0) and expected > 2
    assert yr == pytest.approx(yl, abs=0.5)


def test_distance_map_resized_and_validation():
    h, w = 128, 256
    eq = checker(h, w, 8)
    small = np.full((32, 64), 1.5, np.float32)
    left, right = O.ods_pair(eq, small, 0.064)
    assert left.shape == eq.shape
    with pytest.raises(InputError):
        O.ods_pair(eq.astype(np.float32), small)
    with pytest.raises(InputError):
        O.ods_pair(eq, small, -0.1)
    with pytest.raises(InputError):
        O.ods_pair(eq, np.zeros((32, 64), np.float32))
    with pytest.raises(InputError):
        O.ods_pair(eq, small, pole_fade=(80.0, 70.0))


def test_progress_and_cancel():
    h, w = 128, 256
    eq = checker(h, w, 8)
    dist = np.full((h, w), 1.0, np.float32)
    seen: list[float] = []
    O.ods_pair(eq, dist, progress=ProgressReporter(lambda f, m: seen.append(f)))
    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        O.ods_pair(eq, dist, cancel=tok)


# --------------------------------------------------------------------------- review additions
def _room_distance(h: int, w: int) -> np.ndarray:
    """Distance to the walls of an off-centre box room (smooth, no disocclusions)."""
    from vrphoto.geometry.projection import equirect_dirs

    d = equirect_dirs(w, h).astype(np.float64)
    lo, hi = np.array([-3.0, -1.4, -4.0]), np.array([2.5, 1.6, 3.5])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hi = np.where(d > 0, hi / np.where(d > 0, d, 1), np.inf)
        t_lo = np.where(d < 0, lo / np.where(d < 0, d, 1), np.inf)
    return np.minimum(t_hi, t_lo).min(-1).astype(np.float32)


def test_backward_warp_inverts_algorithm_g_everywhere():
    """Decode, for every output pixel, which source pixel the warp picked and check that
    algorithm G maps that source pixel back onto the output pixel (fixed point solved)."""
    h, w = 256, 512
    yy, xx = np.mgrid[0:h, 0:w]
    ang = 2 * np.pi * (xx + 0.5) / w
    eq = np.stack([128 + 120 * np.sin(ang), 128 + 120 * np.cos(ang), (yy + 0.5) * 255.0 / h], -1)
    eq = np.rint(eq).astype(np.uint8)  # source column on a circle (R, G), source row in B
    dist = _room_distance(h, w) * 0.25  # 0.35..1.3 m: strong parallax (up to ~6 px)
    dx, dy = O.ods_displacement(dist, 0.064)
    left, right = O.ods_pair(eq, dist, 0.064)
    rows = slice(int(h * 0.1), int(h * 0.9))
    for out, e in ((left, 1.0), (right, -1.0)):
        o = out.astype(np.float64)
        sx = np.mod(np.arctan2(o[..., 0] - 128, o[..., 1] - 128) * w / (2 * np.pi) - 0.5, w)
        sy = o[..., 2] * h / 255.0 - 0.5
        ix = np.clip(np.rint(sx).astype(int) % w, 0, w - 1)
        iy = np.clip(np.rint(sy).astype(int), 0, h - 1)
        fx = sx + e * dx[iy, ix]  # forward-map the decoded source pixel
        fy = sy + dy[iy, ix]
        ex = np.abs(np.mod(fx - xx + w / 2, w) - w / 2)[rows]
        ey = np.abs(fy - yy)[rows]
        assert np.median(ex) < 0.35 and np.percentile(ex, 99) < 1.0  # unwarped input: ~4.3 / 7.6
        assert np.median(ey) < 0.6 and np.percentile(ey, 99) < 1.5
    assert dx[rows].max() > 5  # the test exercises real parallax


def test_ipd_validation():
    dist = np.full((16, 32), 1.0, np.float32)
    eq = checker(16, 32, 4)
    for bad in ("wide", None, float("nan"), -0.01, 1.5):
        with pytest.raises(InputError):
            O.ods_pair(eq, dist, bad)
        with pytest.raises(InputError):
            O.ods_displacement(dist, bad)
    dx, _ = O.ods_displacement(dist[..., None], 0.064)  # HxWx1 accepted
    assert dx.shape == (16, 32)


def test_distance_map_resize_wraps_longitude():
    """A low-res distance map is upsampled on inverse distance with longitude wrap, so the
    +-180 deg seam (column 0 vs W-1) blends both sides instead of replicating one."""
    small = np.full((8, 16), 2.0, np.float32)
    small[:, 0] = 0.5  # near column at the left edge of the seam
    d = O._prepare_distance(small, 32, 64, min_m=0.064)
    assert d.shape == (32, 64) and d.dtype == np.float32
    # output column 63 is small x = 15.375: 5/8 of column 15 + 3/8 of (wrapped) column 0;
    # a replicate border would leave it at the far 2.0 m -> parallax jump at the seam
    assert d[16, 63] == pytest.approx(1.0 / (0.625 / 2.0 + 0.375 / 0.5), rel=1e-3)
    # output column 0 is small x = -0.375: 3/8 of (wrapped) column 15 + 5/8 of column 0
    assert d[16, 0] == pytest.approx(1.0 / (0.375 / 2.0 + 0.625 / 0.5), rel=1e-3)
    assert d[16, 32] == pytest.approx(2.0)


def test_far_seam_coordinates_wrap(monkeypatch):
    """The fixed point keeps x inside [0, W): coordinates never leave the int16 range that
    OpenCV 4.x remap saturates to, even when content crosses the seam."""
    import cv2

    seen = []
    real = cv2.remap

    def spy(src, mx, my, *a, **k):
        if mx.dtype == np.float32:
            seen.append((float(mx.min()), float(mx.max()), src.shape[1]))
        return real(src, mx, my, *a, **k)

    monkeypatch.setattr(O.cv2, "remap", spy)
    h, w = 64, 128
    O.ods_pair(checker(h, w, 8), np.full((h, w), 0.2, np.float32), 0.064)
    assert seen
    assert all(lo >= 0.0 and hi < sw for lo, hi, sw in seen)


def test_float_distance_dtypes_and_lists():
    h, w = 32, 64
    eq = checker(h, w, 8)
    ref = O.ods_pair(eq, np.full((h, w), 1.5, np.float32))
    for dist in (np.full((h, w), 1.5, np.float64), np.full((h, w), 1.5, np.float16),
                 [[1.5] * w] * h):
        out = O.ods_pair(eq, dist)
        assert np.array_equal(out[0], ref[0]) and np.array_equal(out[1], ref[1])
    with pytest.raises(InputError):
        O.ods_pair(eq, [["x"] * w] * h)


def test_fill_disocclusions_empty_and_all_masked():
    h, w = 16, 32
    img = checker(h, w, 4)
    invd = np.full((h, w), 0.5, np.float32)
    same = O._fill_disocclusions(img, np.ones((h, w), np.float32), invd, None, None)
    assert same is img  # nothing to fill: untouched
    out = O._fill_disocclusions(img, np.zeros((h, w), np.float32), invd, None, None)
    assert out.shape == img.shape and out.dtype == np.uint8  # all hole: no crash
    calls = []
    O._fill_disocclusions(img, np.zeros((h, w), np.float32), invd,
                          lambda im, m: calls.append(1) or im, None)
    assert not calls  # nothing known to inpaint from -> inpainter not called
