"""inpaint_by_crops / plan_crops."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InferenceError, InputError
from vrphoto.core.types import CancelToken, ProgressReporter
from vrphoto.inpaint.crops import inpaint_by_crops, plan_crops

FILL = np.array([10, 200, 90], np.uint8)


class Recorder:
    """fn that paints holes with FILL and records what it was given."""

    def __init__(self, offset: int = 0):
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.offset = offset

    def __call__(self, crop, mask):
        self.calls.append((crop.copy(), mask.copy()))
        out = np.clip(crop.astype(np.int16) + self.offset, 0, 255).astype(np.uint8)
        out[mask] = FILL
        return out


def _inside(box, outer):
    return box[0] >= outer[0] and box[1] >= outer[1] and box[2] <= outer[2] and box[3] <= outer[3]


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ----------------------------------------------------------------------------- plan_crops
def test_plan_single_component_square_with_margin():
    m = np.zeros((600, 800), bool)
    m[280:320, 380:440] = True  # 60 wide, 40 high
    (box,) = plan_crops(m)
    x0, y0, x1, y1 = box
    assert x1 - x0 == y1 - y0 == 128  # 2 * 60 < min 128
    assert _inside((380, 280, 440, 320), box)
    m2 = np.zeros((600, 800), bool)
    m2[200:300, 300:450] = True  # 150 wide -> 300 px crop
    (box2,) = plan_crops(m2)
    assert box2[2] - box2[0] == 300 and box2[3] - box2[1] == 300
    assert _inside((300, 200, 450, 300), box2)


def test_plan_near_border_is_clamped():
    m = np.zeros((400, 500), bool)
    m[0:10, 0:12] = True  # top-left corner
    m[390:400, 480:500] = True  # bottom-right corner
    boxes = plan_crops(m)
    assert len(boxes) == 2
    for b in boxes:
        assert b[0] >= 0 and b[1] >= 0 and b[2] <= 500 and b[3] <= 400
        assert b[2] - b[0] == b[3] - b[1] == 128
    assert (0, 0, 128, 128) in boxes
    assert (372, 272, 500, 400) in boxes


def test_plan_far_components_separate_near_components_merge():
    m = np.zeros((1000, 1000), bool)
    m[100:110, 100:110] = True
    m[800:810, 800:810] = True
    assert len(plan_crops(m)) == 2
    m[100:110, 150:160] = True  # close to the first one -> merged
    boxes = plan_crops(m)
    assert len(boxes) == 2
    assert any(_inside((100, 100, 160, 110), b) for b in boxes)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not _overlap(a, b)


def test_plan_chain_merge():
    m = np.zeros((800, 1600), bool)
    for x in range(100, 1500, 100):  # a row of dots 100 px apart: crops overlap pairwise
        m[400:405, x : x + 5] = True
    boxes = plan_crops(m, whole_frac=1.01)
    assert len(boxes) == 1
    assert _inside((100, 400, 1405, 405), boxes[0])


def test_plan_every_masked_pixel_in_exactly_one_crop():
    rng = np.random.default_rng(3)
    m = np.zeros((700, 900), bool)
    for _ in range(25):
        y, x = rng.integers(0, 690), rng.integers(0, 890)
        s = rng.integers(1, 40)
        m[y : y + s, x : x + s] = True
    boxes = plan_crops(m, whole_frac=1.01)
    cover = np.zeros(m.shape, np.int32)
    for x0, y0, x1, y1 in boxes:
        cover[y0:y1, x0:x1] += 1
    assert cover.max() <= 1  # crops never overlap
    assert np.all(cover[m] == 1)


def test_plan_large_mask_uses_whole_image():
    m = np.zeros((300, 400), bool)
    m[20:280, 30:370] = True
    assert plan_crops(m) == [(0, 0, 400, 300)]


def test_plan_small_image_whole():
    m = np.zeros((60, 100), bool)
    m[20:30, 40:50] = True
    assert plan_crops(m) == [(0, 0, 100, 60)]


def test_plan_empty():
    assert plan_crops(np.zeros((10, 10), bool)) == []


def test_plan_many_speckles_is_fast():
    rng = np.random.default_rng(0)
    m = np.zeros((2000, 3000), bool)
    ys, xs = rng.integers(0, 2000, 3000), rng.integers(0, 3000, 3000)
    m[ys, xs] = True
    t = time.perf_counter()
    boxes = plan_crops(m)
    assert time.perf_counter() - t < 3.0
    assert boxes  # (whole image, most likely)


# ----------------------------------------------------------------------------- inpaint_by_crops
def test_fills_multiple_components_including_border(textured_img):
    m = np.zeros(textured_img.shape[:2], bool)
    m[0:12, 0:15] = True  # corner
    m[100:130, 140:170] = True  # middle
    m[230:240, 300:320] = True  # opposite corner
    fn = Recorder()
    out = inpaint_by_crops(textured_img, m, fn, size=64, feather=0, whole_frac=1.01, max_downscale=None)
    assert out.dtype == np.uint8 and out.shape == textured_img.shape
    assert np.array_equal(out[~m], textured_img[~m])  # unmasked bit-identical
    assert len(fn.calls) == len(plan_crops(m, whole_frac=1.01))
    for crop, mk in fn.calls:
        assert crop.shape == (64, 64, 3) and crop.dtype == np.uint8
        assert mk.shape == (64, 64) and mk.dtype == np.bool_ and mk.any()
    # deep inside every hole the fill colour arrives (edges are resampled)
    centres = (
        (slice(4, 7), slice(5, 9)),
        (slice(105, 125), slice(145, 165)),
        (slice(234, 237), slice(306, 314)),
    )
    for ys, xs in centres:
        np.testing.assert_allclose(out[ys, xs].reshape(-1, 3).mean(0), FILL, atol=12)


def test_unmasked_bit_identical_with_feather(textured_img):
    m = np.zeros(textured_img.shape[:2], bool)
    cv2.circle(m.view(np.uint8), (160, 120), 30, 1, -1)
    m = m.view(bool)
    out = inpaint_by_crops(textured_img, m, Recorder(offset=40), size=128, feather=6)
    assert np.array_equal(out[~m], textured_img[~m])
    assert not np.array_equal(out[m], textured_img[m])


def test_fn_gets_prefilled_holes_and_conservative_mask():
    img = np.full((300, 300, 3), 20, np.uint8)
    m = np.zeros((300, 300), bool)
    m[140:147, 100:211] = True  # thin band, 111 px wide -> 2.2x downscale onto 100 px
    img[m] = 255  # garbage (e.g. the foreground) inside the hole
    fn = Recorder()
    inpaint_by_crops(img, m, fn, size=100, feather=0, whole_frac=1.01, max_downscale=None)
    ((crop, mk),) = fn.calls
    assert crop.max() <= 30  # garbage never reaches the model, not even blurred
    # every original hole pixel maps onto a masked model pixel
    (box,) = plan_crops(m, whole_frac=1.01)
    x0, y0, x1, y1 = box
    side = x1 - x0
    ys, xs = np.nonzero(m[y0:y1, x0:x1])
    sy = np.clip(((ys + 0.5) * 100 / side).astype(int), 0, 99)
    sx = np.clip(((xs + 0.5) * 100 / side).astype(int), 0, 99)
    assert mk[sy, sx].all()


def test_upscaled_crop_gets_prefilled_holes():
    img = np.full((300, 300, 3), 20, np.uint8)
    m = np.zeros((300, 300), bool)
    m[150:160, 150:160] = True
    img[m] = 255
    fn = Recorder()
    out = inpaint_by_crops(img, m, fn, size=256, feather=0)  # 128 px crop enlarged to 256
    ((crop, mk),) = fn.calls
    assert crop.shape == (256, 256, 3)
    assert crop.max() <= 30
    assert mk.sum() >= 4 * 10 * 10  # hole scaled by 2 in each direction (conservatively)
    assert np.array_equal(out[~m], img[~m])


def test_non_square_crop_is_padded(textured_img):
    img = textured_img[:60, :100].copy()  # smaller than min crop -> whole (non-square) image
    m = np.zeros((60, 100), bool)
    m[25:35, 45:60] = True
    fn = Recorder()
    out = inpaint_by_crops(img, m, fn, size=64, feather=0)
    ((crop, mk),) = fn.calls
    assert crop.shape == (64, 64, 3)
    assert np.array_equal(out[~m], img[~m])
    np.testing.assert_allclose(out[28:32, 48:57].reshape(-1, 3).mean(0), FILL, atol=10)


def test_seam_blending_matches_surroundings():
    img = np.full((200, 200, 3), 100, np.uint8)
    m = np.zeros((200, 200), bool)
    m[70:130, 70:130] = True

    def shifted(crop, mask):  # model output with a global colour shift
        return np.full_like(crop, 140)

    hard = inpaint_by_crops(img, m, shifted, size=128, feather=0, whole_frac=1.01)
    soft = inpaint_by_crops(img, m, shifted, size=128, feather=6, whole_frac=1.01)
    assert hard[70, 100, 0] == 140  # seam: 40 levels jump at the border
    assert abs(int(soft[70, 100, 0]) - 100) <= 4  # border pixel meets the surroundings
    assert soft[100, 100, 0] == 140  # interior keeps the model output
    assert int(soft[73, 100, 0]) - 100 > 0  # ramps over the feather width
    assert np.all(np.diff(soft[70:80, 100, 0].astype(int)) >= 0)


def test_whole_image_mode_single_call(textured_img):
    m = np.zeros(textured_img.shape[:2], bool)
    m[10:230, 10:310] = True
    fn = Recorder()
    out = inpaint_by_crops(textured_img, m, fn, size=64)
    assert len(fn.calls) == 1
    assert np.array_equal(out[~m], textured_img[~m])


def test_empty_mask_returns_copy(textured_img):
    out = inpaint_by_crops(textured_img, np.zeros(textured_img.shape[:2], bool), Recorder())
    assert out is not textured_img
    assert np.array_equal(out, textured_img)


def test_mask_dtypes_and_validation(textured_img):
    m8 = np.zeros(textured_img.shape[:2], np.uint8)
    m8[100:110, 100:110] = 255
    out = inpaint_by_crops(textured_img, m8, Recorder(), size=64, feather=0)
    assert np.array_equal(out[m8 == 0], textured_img[m8 == 0])
    np.testing.assert_allclose(out[103:107, 103:107].reshape(-1, 3).mean(0), FILL, atol=10)
    with pytest.raises(InputError):
        inpaint_by_crops(textured_img, m8[:10], Recorder())
    with pytest.raises(InputError):
        inpaint_by_crops(textured_img.astype(np.float32), m8, Recorder())
    with pytest.raises(InputError):
        inpaint_by_crops(textured_img, m8, Recorder(), size=4)


def test_bad_fn_results(textured_img):
    m = np.zeros(textured_img.shape[:2], bool)
    m[50:60, 50:60] = True
    with pytest.raises(InferenceError):
        inpaint_by_crops(textured_img, m, lambda c, k: c[:10], size=64)
    with pytest.raises(InferenceError):
        inpaint_by_crops(textured_img, m, lambda c, k: None, size=64)  # type: ignore[arg-type,return-value]
    # float output is accepted and converted
    out = inpaint_by_crops(textured_img, m, lambda c, k: c.astype(np.float32) * 0 + 77.4, size=64, feather=0)
    assert out[55, 55, 0] == 77


def test_cancel_and_progress(textured_img):
    m = np.zeros(textured_img.shape[:2], bool)
    m[10:20, 10:20] = True
    m[200:210, 280:290] = True
    seen: list[float] = []
    inpaint_by_crops(
        textured_img, m, Recorder(), size=64, progress=ProgressReporter(lambda f, s: seen.append(f))
    )
    assert seen[0] == 0.0 and seen[-1] == pytest.approx(1.0)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        inpaint_by_crops(textured_img, m, Recorder(), size=64, cancel=tok)

    calls = []

    def cancel_after_first(crop, mask):
        calls.append(1)
        tok2.cancel()
        return crop

    tok2 = CancelToken()
    with pytest.raises(Cancelled):
        inpaint_by_crops(textured_img, m, cancel_after_first, size=64, cancel=tok2, whole_frac=1.01)
    assert len(calls) == 1


# ----------------------------------------------------------------------------- tiled mode
def _ring_mask(h, w, thickness):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 1, thickness)
    return m.astype(bool)


def test_thin_large_mask_is_tiled_and_filled(textured_img):
    img = cv2.resize(textured_img, (960, 720), interpolation=cv2.INTER_CUBIC)
    m = _ring_mask(720, 960, 12)
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    colours = [np.array(c, np.uint8) for c in ((250, 0, 0), (0, 250, 0), (0, 0, 250), (250, 250, 0))]

    def fn(crop, mask):
        calls.append((crop.copy(), mask.copy()))
        out = crop.copy()
        out[mask] = colours[(len(calls) - 1) % 4]
        return out

    out = inpaint_by_crops(img, m, fn, size=64, feather=0)  # max side 128 -> tiles of 64 px
    assert len(calls) > 10
    assert np.array_equal(out[~m], img[~m])
    filled = out[m]
    is_fill = np.zeros(len(filled), bool)
    for c in colours:
        is_fill |= np.abs(filled.astype(int) - c).sum(1) < 120
    assert is_fill.mean() > 0.6  # (the rest is edge resampling)
    # later tiles see earlier fills as known context (progressive, seamless)
    seen_prev = False
    for crop, mask in calls[1:]:
        known = crop[~mask].astype(int)
        for c in colours:
            if (np.abs(known - c).sum(1) < 30).sum() > 5:
                seen_prev = True
    assert seen_prev


def test_tiles_respect_max_downscale(textured_img):
    img = cv2.resize(textured_img, (1200, 900), interpolation=cv2.INTER_CUBIC)
    m = _ring_mask(900, 1200, 10)
    sizes: list[int] = []

    def fn(crop, mask):
        sizes.append(crop.shape[0])
        return crop

    inpaint_by_crops(img, m, fn, size=100, max_downscale=2.0)
    tiled = len(sizes)
    sizes.clear()
    inpaint_by_crops(img, m, fn, size=100, max_downscale=None)
    assert len(sizes) == 1  # one big downscaled crop without tiling
    assert tiled > 4


def test_compact_large_hole_not_tiled(textured_img):
    img = cv2.resize(textured_img, (960, 720), interpolation=cv2.INTER_CUBIC)
    m = np.zeros((720, 960), bool)
    m[200:500, 300:600] = True  # 300 px blob: deep interior -> keep global context
    fn = Recorder()
    out = inpaint_by_crops(img, m, fn, size=64)
    assert len(fn.calls) == 1
    assert np.array_equal(out[~m], img[~m])


def test_mask_covering_most_of_image_is_whole(textured_img):
    img = cv2.resize(textured_img, (960, 720), interpolation=cv2.INTER_CUBIC)
    m = np.zeros((720, 960), bool)
    m[:, :600] = True
    fn = Recorder()
    out = inpaint_by_crops(img, m, fn, size=64)
    assert len(fn.calls) == 1
    assert np.array_equal(out[~m], img[~m])
    assert plan_crops(m) == [(0, 0, 960, 720)]


def test_tile_cells_partition():
    from vrphoto.inpaint.crops import tile_cells

    cells = tile_cells((10, 20, 1010, 520), 300)
    cover = np.zeros((600, 1100), np.int32)
    for x0, y0, x1, y1 in cells:
        assert x1 - x0 <= 300 and y1 - y0 <= 300
        cover[y0:y1, x0:x1] += 1
    assert cover[20:520, 10:1010].min() == 1 and cover.max() == 1
