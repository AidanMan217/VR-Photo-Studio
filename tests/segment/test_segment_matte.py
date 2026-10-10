"""clean_matte / refine_matte behaviour."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vrphoto.core.errors import InputError
from vrphoto.segment.matte import (
    clean_matte,
    feather_mask,
    fill_mask_holes,
    largest_component,
    refine_matte,
    to_alpha,
)


def _blob_alpha() -> np.ndarray:
    a = np.zeros((120, 160), np.float32)
    a[20:100, 30:110] = 0.9  # main subject (80x80)
    a[5:15, 140:150] = 0.8  # small speckle
    a[50:54, 60:64] = 0.1  # small hole inside the subject
    return a


def test_returns_float32_in_range():
    out = clean_matte(_blob_alpha())
    assert out.dtype == np.float32
    assert out.shape == (120, 160)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_threshold_and_largest_only():
    a = _blob_alpha()
    out = clean_matte(a, feather_px=0)
    assert out[60, 70] == 1.0
    assert out[10, 145] == 0.0  # speckle removed
    assert set(np.unique(out)) <= {0.0, 1.0}
    keep = clean_matte(a, feather_px=0, largest_only=False)
    assert keep[10, 145] == 1.0


def test_threshold_value_respected():
    a = np.zeros((50, 50), np.float32)
    a[10:40, 10:40] = 0.4
    assert clean_matte(a, threshold=0.5).max() == 0.0
    assert clean_matte(a, threshold=0.3, feather_px=0)[25, 25] == 1.0


def test_small_hole_filled_large_gap_kept():
    a = _blob_alpha()
    out = clean_matte(a, feather_px=0)
    assert out[52, 62] == 1.0  # tiny hole filled
    nofill = clean_matte(a, feather_px=0, fill_holes=False)
    assert nofill[52, 62] == 0.0

    # ring with a big interior gap (like an arm against the body): kept by default
    ring = np.zeros((200, 200), np.float32)
    cv2.circle(ring, (100, 100), 80, 1.0, -1)
    cv2.circle(ring, (100, 100), 30, 0.0, -1)
    kept = clean_matte(ring, feather_px=0)
    assert kept[100, 100] == 0.0
    filled = clean_matte(ring, feather_px=0, max_hole_frac=None)
    assert filled[100, 100] == 1.0


def test_hole_touching_border_is_not_a_hole():
    m = np.zeros((40, 40), bool)
    m[:, 10:30] = True
    m[0:10, 18:22] = False  # notch open to the top border
    out = fill_mask_holes(m, None)
    assert not out[2, 20]


def test_feather_ramp_centred_on_boundary():
    a = np.zeros((100, 100), np.float32)
    a[:, 50:] = 1.0
    out = clean_matte(a, feather_px=4, largest_only=False)
    row = out[50]
    assert row[0] == 0.0 and row[-1] == 1.0
    assert np.all(np.diff(row) >= -1e-6)  # monotone
    # pixels either side of the edge straddle 0.5 symmetrically
    assert abs((row[49] + row[50]) - 1.0) < 1e-5
    assert row[49] < 0.5 < row[50]
    # ramp width ~ 2 * feather
    soft = np.count_nonzero((row > 0.0) & (row < 1.0))
    assert 6 <= soft <= 9
    # the 0.5 iso-line stays on the binary boundary
    assert np.array_equal(out >= 0.5, a >= 0.5)


def test_feather_zero_is_binary():
    out = feather_mask(np.eye(10, dtype=bool), 0)
    assert out.dtype == np.float32
    assert set(np.unique(out)) <= {0.0, 1.0}


def test_empty_and_full():
    assert clean_matte(np.zeros((30, 40), np.float32)).max() == 0.0
    full = clean_matte(np.ones((30, 40), np.float32))
    assert full.min() == 1.0


def test_accepts_uint8_and_bool():
    a8 = (_blob_alpha() * 255).astype(np.uint8)
    np.testing.assert_allclose(clean_matte(a8), clean_matte(_blob_alpha()))
    ab = _blob_alpha() > 0.5
    assert clean_matte(ab, feather_px=0)[60, 70] == 1.0
    assert to_alpha(np.array([[0, 255]], np.uint8)).tolist() == [[0.0, 1.0]]


def test_invalid_inputs():
    with pytest.raises(InputError):
        clean_matte(np.zeros((4, 4, 3), np.float32))
    with pytest.raises(InputError):
        clean_matte(np.zeros((4, 4), np.float32), threshold=2.0)
    with pytest.raises(InputError):
        clean_matte([[0.0]])  # type: ignore[arg-type]


def test_largest_component_helper():
    m = np.zeros((20, 20), bool)
    m[1:3, 1:3] = True
    m[10:18, 10:18] = True
    out = largest_component(m)
    assert out[12, 12] and not out[1, 1]


def test_refine_matte_snaps_to_edges(person_alpha):
    h, w = person_alpha.shape
    img = np.full((h, w, 3), 30, np.uint8)
    img[person_alpha > 0.5] = (220, 200, 180)
    coarse = cv2.resize(cv2.resize(person_alpha, (w // 8, h // 8), interpolation=cv2.INTER_AREA), (w, h))
    refined = refine_matte(coarse, img, radius=4)
    assert refined.dtype == np.float32 and refined.shape == (h, w)
    gt = person_alpha > 0.5
    wrong_coarse = np.count_nonzero((coarse > 0.5) != gt)
    wrong_refined = np.count_nonzero((refined > 0.5) != gt)
    assert wrong_refined < 0.75 * wrong_coarse
    assert refined.min() >= 0.0 and refined.max() <= 1.0
    with pytest.raises(InputError):
        refine_matte(coarse, img[:10])


def test_fast_guided_filter_matches_exact():
    """Large images use the subsampled guided filter; it must agree with the exact one."""
    from vrphoto.segment.matte import guided_filter_gray

    h, w = 600, 800
    yy, xx = np.mgrid[0:h, 0:w]
    disc = np.hypot(xx - 400, yy - 300) < 200
    gray = (disc * 0.8 + 0.1).astype(np.float32)
    coarse = cv2.resize(
        cv2.resize(disc.astype(np.float32), (50, 38), interpolation=cv2.INTER_AREA), (w, h)
    ).astype(np.float32)
    exact = guided_filter_gray(gray, coarse, 12, 1e-3, max_px=None)
    fast = guided_filter_gray(gray, coarse, 12, 1e-3, max_px=120_000)  # k = 2
    assert fast.shape == exact.shape and fast.dtype == np.float32
    assert float(np.abs(exact - fast).mean()) < 2e-3
    assert float(np.abs(np.clip(fast, 0, 1) - disc).mean()) <= float(np.abs(coarse - disc).mean())
