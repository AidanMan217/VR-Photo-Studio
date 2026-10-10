"""Tests for geometry.layers: layer split for look-behind (algorithm C)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InferenceError, InputError
from vrphoto.core.types import CancelToken, Intrinsics, Pose, ProgressReporter
from vrphoto.geometry.layers import band_mask, build_layers, depth_edges
from vrphoto.geometry.warp import render_view


class FakeInpainter:
    """Returns the true far-plane texture and records the masks it was asked to fill."""

    id = "fake"

    def __init__(self, far_rgb: np.ndarray, garbage_outside: bool = False):
        self.far = far_rgb
        self.masks: list[np.ndarray] = []
        self.garbage_outside = garbage_outside

    def inpaint(self, img, mask):
        assert mask.dtype == bool and mask.shape == img.shape[:2]
        self.masks.append(mask.copy())
        out = self.far.copy()
        if self.garbage_outside:
            out[~mask] = (255, 0, 255)
        return out


class KwInpainter(FakeInpainter):
    def __init__(self, far_rgb):
        super().__init__(far_rgb)
        self.kwargs: dict = {}

    def inpaint(self, img, mask, *, progress=None, cancel=None):
        self.kwargs = {"progress": progress, "cancel": cancel}
        return super().inpaint(img, mask)


def _shift(img, dx):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- modes
def test_none_mode_single_layer(two_plane_scene):
    s = two_plane_scene
    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None, mode="none")
    assert len(li.layers) == 1 and li.K == s.K
    lay = li.layers[0]
    np.testing.assert_array_equal(lay.rgb, s.img)
    np.testing.assert_array_equal(lay.inv_z, s.inv_z)
    assert lay.inv_z is not s.inv_z
    assert lay.alpha.dtype == np.float32 and (lay.alpha == 1).all()


def test_band_mode_structure(two_plane_scene):
    s = two_plane_scene
    fake = FakeInpainter(s.far_rgb)
    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=fake, mode="band")
    assert len(li.layers) == 2
    bg, fg = li.layers
    np.testing.assert_array_equal(fg.rgb, s.img)
    np.testing.assert_array_equal(fg.inv_z, s.inv_z)
    assert (fg.alpha == 1).all() and (bg.alpha == 1).all()
    assert len(fake.masks) == 1
    H = fake.masks[0]
    # H lies behind the foreground only, as a band inside the square's border
    assert H.any() and not (H & ~s.fg_mask).any()
    ring = s.fg_mask & ~cv2.erode(s.fg_mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    assert H[ring].all(), "band must cover at least the max parallax next to the edge"
    # background layer: inpainted colour and far depth in the band, photo elsewhere
    np.testing.assert_array_equal(bg.rgb[H], s.far_rgb[H])
    np.testing.assert_array_equal(bg.rgb[~H], s.img[~H])
    np.testing.assert_allclose(bg.inv_z[ring], 1.0 / s.z_far, atol=1e-6)
    np.testing.assert_array_equal(bg.inv_z[~H], s.inv_z[~H])
    assert (bg.inv_z <= s.inv_z + 1e-7).all()


def test_band_parameters_follow_algorithm_c(two_plane_scene):
    s = two_plane_scene
    bmax = 0.1
    fb = s.K.fx * bmax
    H, d_bg, k = band_mask(s.inv_z, s.K, bmax)
    expected_k = math.ceil(1.3 * fb * (1 / s.z_near - 1 / s.z_far)) + 4
    assert k == expected_k
    e = depth_edges(s.inv_z, fb)
    border = s.fg_mask ^ cv2.erode(s.fg_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert e[border].all()
    assert not e[30:60, 30:60].any()
    # k is capped at 15 % of the width
    _, _, k_big = band_mask(s.inv_z, s.K, 5.0)
    assert k_big == int(0.15 * s.img.shape[1])
    # pooled path (k > 24) still yields a valid band behind the square only
    Hb, d_b, kb = band_mask(s.inv_z, s.K, 0.3)
    assert kb > 24 and Hb is not None
    assert not (Hb & ~s.fg_mask).any()
    ring = s.fg_mask & ~cv2.erode(s.fg_mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    assert Hb[ring].all()
    assert (d_b <= s.inv_z + 1e-7).all()


@pytest.mark.parametrize("tx", [0.05, -0.05])
@pytest.mark.parametrize("bmax", [0.05, 0.1])
def test_look_behind_golden(two_plane_scene, psnr_fn, tx, bmax):
    """Golden: revealed background comes from the inpainted band, no interior holes."""
    s = two_plane_scene
    h, w = s.img.shape[:2]
    f = s.K.fx
    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=bmax, inpainter=FakeInpainter(s.far_rgb))
    rr = render_view(li, Pose.translate(tx, 0.0, 0.0))
    df, dn = -f * tx / s.z_far, -f * tx / s.z_near
    # holes away from the image border strip that has no source pixels at all
    m = int(math.ceil(abs(df))) + 2
    holes = rr.alpha[:, m : w - m] < 0.5
    assert holes.mean() < 0.001
    # revealed region: far-plane pixels that were hidden by the square in the source,
    # at least 1 px away from the square's new silhouette
    u = np.arange(w)[None, :].astype(np.float64)
    rows = np.zeros((h, 1), bool)
    rows[80:160] = True
    src_far = u - df
    src_near = u - dn
    hidden = (src_far >= 119.5) & (src_far <= 199.5) & rows
    covered = (src_near >= 118.5) & (src_near <= 200.5) & rows
    revealed = hidden & ~covered
    assert revealed.sum() >= 3 * 80
    expected = _shift(s.far_rgb, df)
    assert psnr_fn(rr.rgb, expected, revealed) > 30.0


def test_band_reveals_far_texture_not_stretched_foreground(two_plane_scene):
    s = two_plane_scene
    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=FakeInpainter(s.far_rgb))
    single = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None, mode="none")
    rr2 = render_view(li, Pose.translate(0.05))
    rr1 = render_view(single, Pose.translate(0.05))
    gap = (slice(85, 155), slice(192, 196))
    assert (rr1.alpha[gap] < 0.5).all()  # a hole without look-behind
    assert (rr2.alpha[gap] > 0.99).all()  # filled by the background layer


def test_no_depth_edges_gives_single_layer(photo_rgb):
    h, w = photo_rgb.shape[:2]
    K = Intrinsics.from_hfov(w, h, 60.0)
    inv = np.full((h, w), 0.25, np.float32)
    calls = []

    class Spy:
        def inpaint(self, img, mask):
            calls.append(1)
            return img

    li = build_layers(photo_rgb, inv, K, max_baseline_m=0.1, inpainter=Spy())
    assert len(li.layers) == 1 and not calls


def test_inpainter_none_fallback(two_plane_scene):
    s = two_plane_scene
    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None)
    assert len(li.layers) == 2
    bg = li.layers[0]
    H, _, _ = band_mask(s.inv_z, s.K, 0.1)
    assert H is not None
    np.testing.assert_array_equal(bg.rgb[~H], s.img[~H])
    # the band is filled from the surroundings (no longer a copy of the square texture)
    assert not np.array_equal(bg.rgb[H], s.img[H])
    rr = render_view(li, Pose.translate(0.05))
    assert (rr.alpha[85:155, 192:196] > 0.99).all()


def test_inpainter_kwargs_and_output_outside_mask_ignored(two_plane_scene):
    s = two_plane_scene
    kw = KwInpainter(s.far_rgb)
    tok = CancelToken()
    build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=kw, cancel=tok)
    assert kw.kwargs["cancel"] is tok
    assert isinstance(kw.kwargs["progress"], ProgressReporter)
    garbage = FakeInpainter(s.far_rgb, garbage_outside=True)
    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=garbage)
    H = garbage.masks[0]
    np.testing.assert_array_equal(li.layers[0].rgb[~H], s.img[~H])


def test_inpainter_float_output_accepted(two_plane_scene):
    s = two_plane_scene

    class FloatInp:
        def inpaint(self, img, mask):
            return s.far_rgb.astype(np.float32) / 255.0

    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=FloatInp())
    H, _, _ = band_mask(s.inv_z, s.K, 0.1)
    np.testing.assert_array_equal(li.layers[0].rgb[H], s.far_rgb[H])


def test_inpainter_failures_are_readable(two_plane_scene):
    s = two_plane_scene

    class Boom:
        id = "boom"

        def inpaint(self, img, mask):
            raise RuntimeError("CUDA out of memory")

    class WrongSize:
        def inpaint(self, img, mask):
            return img[:10, :10]

    with pytest.raises(InferenceError, match="boom"):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=Boom())
    with pytest.raises(InferenceError):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=WrongSize())
    with pytest.raises(InputError):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=object())


def test_inpainter_cancel_propagates(two_plane_scene):
    s = two_plane_scene

    class Cancelling:
        def inpaint(self, img, mask, *, progress=None, cancel=None):
            raise Cancelled("stop")

    with pytest.raises(Cancelled):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=Cancelling())


# --------------------------------------------------------------------------- clean plate
def test_cleanplate_removes_subject(two_plane_scene, psnr_fn):
    s = two_plane_scene
    fake = FakeInpainter(s.far_rgb)
    alpha = s.fg_mask.astype(np.float32)
    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=fake, alpha=alpha, mode="cleanplate")
    assert len(li.layers) == 2
    bg, fg = li.layers
    H = fake.masks[0]
    dil = cv2.dilate(s.fg_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
    assert H[dil].all()
    np.testing.assert_array_equal(bg.rgb[s.fg_mask], s.far_rgb[s.fg_mask])
    np.testing.assert_allclose(bg.inv_z[s.fg_mask], 1.0 / s.z_far, atol=1e-5)
    np.testing.assert_array_equal(fg.alpha, alpha)
    np.testing.assert_array_equal(fg.rgb, s.img)
    # moving the camera reveals the true background next to the subject
    rr = render_view(li, Pose.translate(0.05))
    df = -s.K.fx * 0.05 / s.z_far
    exp = _shift(s.far_rgb, df)
    gap = np.zeros(s.fg_mask.shape, bool)
    gap[85:155, 192:196] = True
    assert psnr_fn(rr.rgb, exp, gap) > 30.0
    assert (rr.alpha[:, 10:300] > 0.99).all()


def test_cleanplate_requires_alpha(two_plane_scene):
    s = two_plane_scene
    with pytest.raises(InputError, match="subject"):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None, mode="cleanplate")
    with pytest.raises(InputError):
        build_layers(
            s.img,
            s.inv_z,
            s.K,
            max_baseline_m=0.1,
            inpainter=None,
            mode="cleanplate",
            alpha=np.ones((5, 5), np.float32),
        )


def test_cleanplate_empty_alpha_falls_back_to_band(two_plane_scene):
    s = two_plane_scene
    li = build_layers(
        s.img,
        s.inv_z,
        s.K,
        max_baseline_m=0.1,
        inpainter=FakeInpainter(s.far_rgb),
        alpha=np.zeros(s.inv_z.shape, np.float32),
        mode="cleanplate",
    )
    assert len(li.layers) == 2
    assert (li.layers[1].alpha == 1).all()


def test_cleanplate_zero_baseline_only_removes_subject(two_plane_scene):
    s = two_plane_scene
    fake = FakeInpainter(s.far_rgb)
    li = build_layers(
        s.img,
        s.inv_z,
        s.K,
        max_baseline_m=0.0,
        inpainter=fake,
        alpha=s.fg_mask.astype(np.float32),
        mode="cleanplate",
    )
    assert len(li.layers) == 2 and len(fake.masks) == 1


# --------------------------------------------------------------------------- validation
def test_validation_errors(two_plane_scene):
    s = two_plane_scene
    with pytest.raises(InputError):
        build_layers(s.img, s.inv_z[:100], s.K, max_baseline_m=0.1, inpainter=None)
    with pytest.raises(InputError):
        build_layers(s.img, s.inv_z, Intrinsics.from_hfov(100, 100, 60), max_baseline_m=0.1, inpainter=None)
    with pytest.raises(InputError):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None, mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(InputError):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.0, inpainter=None, mode="band")
    with pytest.raises(InputError):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=float("nan"), inpainter=None)
    with pytest.raises(InputError):
        build_layers(np.zeros((240, 320, 2), np.uint8), s.inv_z, s.K, max_baseline_m=0.1, inpainter=None)


def test_input_normalisation(two_plane_scene):
    s = two_plane_scene
    rgba = np.dstack([s.img, np.full(s.inv_z.shape, 255, np.uint8)])
    li = build_layers(rgba, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None, mode="none")
    np.testing.assert_array_equal(li.layers[0].rgb, s.img)
    li = build_layers(
        s.img.astype(np.float32) / 255.0, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None, mode="none"
    )
    assert li.layers[0].rgb.dtype == np.uint8
    assert np.abs(li.layers[0].rgb.astype(int) - s.img).max() <= 1
    gray = s.img[..., 0]
    assert (
        build_layers(gray, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None, mode="none").layers[0].rgb.shape
        == s.img.shape
    )
    bad = s.inv_z.copy()
    bad[0, 0] = np.nan
    bad[0, 1] = -3
    li = build_layers(s.img, bad, s.K, max_baseline_m=0.1, inpainter=None)
    for lay in li.layers:
        assert np.isfinite(lay.inv_z).all() and (lay.inv_z > 0).all()


def test_cancel_and_progress(two_plane_scene):
    s = two_plane_scene
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None, cancel=tok)
    seen: list[float] = []
    build_layers(
        s.img,
        s.inv_z,
        s.K,
        max_baseline_m=0.1,
        inpainter=FakeInpainter(s.far_rgb),
        progress=ProgressReporter(lambda f, m: seen.append(f)),
    )
    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    seen.clear()
    build_layers(
        s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None, progress=lambda f, m: seen.append(f)
    )
    assert seen[-1] == pytest.approx(1.0)


def test_person_matte_cleanplate_soft_alpha(person_alpha):
    h, w = person_alpha.shape
    K = Intrinsics.from_hfov(w, h, 55.0)
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    inv = np.where(person_alpha > 0.5, 0.5, 0.15).astype(np.float32)
    soft = cv2.GaussianBlur(person_alpha, (0, 0), 1.5)
    li = build_layers(img, inv, K, max_baseline_m=0.08, inpainter=None, alpha=soft, mode="cleanplate")
    assert len(li.layers) == 2
    np.testing.assert_allclose(li.layers[1].alpha, np.clip(soft, 0, 1))
    rr = render_view(li, Pose.translate(-0.04))
    assert rr.alpha[:, 10 : w - 10].min() > 0.99


def test_band_region_recoverable_from_layers(two_plane_scene):
    s = two_plane_scene
    fake = FakeInpainter(s.far_rgb)
    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=fake)
    np.testing.assert_array_equal(li.layers[0].inv_z < li.layers[1].inv_z, fake.masks[0])


# --------------------------------------------------------------------------- review additions
def _scene_with_table():
    """Far wall (4 m), a table edge (1.5 m, NOT the subject) and a subject (1.2 m)."""
    h, w = 240, 320
    K = Intrinsics.from_hfov(w, h, 60.0)
    rng = np.random.default_rng(3)
    far = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 1.5)
    table = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 1.5)
    subj_rgb = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 1.5)
    img = far.copy()
    inv = np.full((h, w), 1 / 4.0, np.float32)
    tmask = np.zeros((h, w), bool)
    tmask[150:, :130] = True
    img[tmask] = table[tmask]
    inv[tmask] = 1 / 1.5
    alpha = np.zeros((h, w), np.float32)
    alpha[50:200, 200:260] = 1.0
    smask = alpha > 0.5
    img[smask] = subj_rgb[smask]
    inv[smask] = 1 / 1.2
    return img, inv, alpha, far, K


def test_cleanplate_reproduces_photo_with_other_depth_edges(psnr_fn):
    """Regression: the bands of non-subject edges must not replace the photo.

    Previously the band-inpainted background was the only layer outside the subject, so
    the source view showed inpainted content along the table edge."""
    img, inv, alpha, far, K = _scene_with_table()
    inpainted: list[np.ndarray] = []

    class TrueWall:
        def inpaint(self, im, mask):
            inpainted.append(mask.copy())
            return far.copy()

    li = build_layers(img, inv, K, max_baseline_m=0.1, inpainter=TrueWall(), alpha=alpha, mode="cleanplate")
    assert len(li.layers) == 3
    back, plate, subject = li.layers
    np.testing.assert_array_equal(subject.alpha, alpha)
    np.testing.assert_array_equal(subject.rgb, img)
    # the plate is the photo outside the (dilated) subject; the band only in the back layer
    ring = (
        cv2.dilate((alpha > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
    )
    np.testing.assert_array_equal(plate.rgb[~ring], img[~ring])
    band = back.inv_z < plate.inv_z
    assert band[150:, 100:130].any(), "a band behind the table edge is expected"
    np.testing.assert_array_equal(back.rgb[band], far[band])
    rr = render_view(li, Pose.identity())
    assert psnr_fn(rr.rgb, img, ~ring) > 45.0
    assert rr.alpha.min() > 0.99
    # moving right reveals the true wall behind the table's right edge
    tx = 0.06
    rr = render_view(li, Pose.translate(tx))
    f = K.fx
    dn, df = -f * tx * 1.5**-1, -f * tx / 4.0
    cols = np.arange(int(math.ceil(129 + dn)) + 2, int(math.floor(130 + df)) - 1)
    assert cols.size >= 4
    gap = np.zeros(inv.shape, bool)
    gap[160:235, cols] = True
    assert (rr.alpha[gap] > 0.99).all()
    assert psnr_fn(rr.rgb, _shift(far, df), gap) > 30.0


def test_cleanplate_without_other_edges_stays_two_layers(two_plane_scene):
    s = two_plane_scene
    li = build_layers(
        s.img,
        s.inv_z,
        s.K,
        max_baseline_m=0.1,
        inpainter=None,
        alpha=s.fg_mask.astype(np.float32),
        mode="cleanplate",
    )
    assert len(li.layers) == 2


def test_fallback_fill_takes_colour_from_the_background_side():
    """No-model look-behind: the band must not be smeared with foreground colour."""
    h, w = 240, 320
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = np.zeros((h, w, 3), np.uint8)
    img[...] = (0, 0, 255)  # blue wall at 4 m
    inv = np.full((h, w), 0.25, np.float32)
    img[60:200, 40:280] = (255, 0, 0)  # red box at 1.2 m
    inv[60:200, 40:280] = 1 / 1.2
    li = build_layers(img, inv, K, max_baseline_m=0.1, inpainter=None)
    H = li.layers[0].inv_z < li.layers[1].inv_z
    band = li.layers[0].rgb[H].astype(np.float64)
    assert band[:, 0].mean() < 20 and band[:, 2].mean() > 235
    rr = render_view(li, Pose.translate(0.1))
    f = K.fx
    dn, df = -f * 0.1 / 1.2, -f * 0.1 * 0.25
    revealed = rr.rgb[80:180, int(279 + dn) + 2 : int(280 + df) - 1].reshape(-1, 3).astype(np.float64)
    assert revealed[:, 0].mean() < 20 and revealed[:, 2].mean() > 235


def test_image_dtype_normalisation():
    h, w = 20, 30
    K = Intrinsics.from_hfov(w, h, 60.0)
    inv = np.full((h, w), 0.5, np.float32)
    img16 = np.full((h, w, 3), 30000, np.uint16)
    rgb = build_layers(img16, inv, K, max_baseline_m=0.1, inpainter=None, mode="none").layers[0].rgb
    assert rgb.dtype == np.uint8 and (rgb == round(30000 / 257)).all()
    imgb = np.zeros((h, w, 3), bool)
    imgb[:, :10] = True
    rgb = build_layers(imgb, inv, K, max_baseline_m=0.1, inpainter=None, mode="none").layers[0].rgb
    assert (rgb[:, :10] == 255).all() and (rgb[:, 10:] == 0).all()
    imgf = np.full((h, w, 3), 200.0, np.float64)  # float in 0..255
    imgf[0, 0] = np.nan
    rgb = build_layers(imgf, inv, K, max_baseline_m=0.1, inpainter=None, mode="none").layers[0].rgb
    assert rgb[5, 5, 0] == 200 and rgb[0, 0, 0] == 0


def test_uint8_matte_is_scaled(two_plane_scene):
    s = two_plane_scene
    a8 = (s.fg_mask * 255).astype(np.uint8)
    a8[s.fg_mask & (np.arange(s.img.shape[1])[None, :] < 125)] = 64  # soft edge
    li = build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=None, alpha=a8, mode="cleanplate")
    fa = li.front.alpha
    assert fa.dtype == np.float32 and fa.max() == pytest.approx(1.0)
    assert fa[100, 121] == pytest.approx(64 / 255.0)


def test_inpainter_bad_returns_are_readable(two_plane_scene):
    s = two_plane_scene

    class Nothing:
        id = "none-returner"

        def inpaint(self, img, mask):
            return None

    class Scalar:
        def inpaint(self, img, mask):
            return 3

    with pytest.raises(InferenceError, match="none-returner"):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=Nothing())
    with pytest.raises(InferenceError):
        build_layers(s.img, s.inv_z, s.K, max_baseline_m=0.1, inpainter=Scalar())
    with pytest.raises(InputError, match="Depth map"):
        build_layers(s.img, np.float32(0.5), s.K, max_baseline_m=0.1, inpainter=None)


def test_cleanplate_subject_filling_the_frame_is_safe():
    h, w = 40, 50
    K = Intrinsics.from_hfov(w, h, 60.0)
    img = np.full((h, w, 3), 90, np.uint8)
    inv = np.linspace(0.3, 0.8, w, dtype=np.float32)[None, :].repeat(h, 0)
    li = build_layers(
        img, inv, K, max_baseline_m=0.1, inpainter=None, alpha=np.ones((h, w), np.float32), mode="cleanplate"
    )
    for lay in li.layers:
        assert np.isfinite(lay.inv_z).all() and (lay.inv_z > 0).all()
        assert lay.rgb.shape == img.shape
    rr = render_view(li, Pose.translate(0.02))
    assert np.isfinite(rr.alpha).all()
