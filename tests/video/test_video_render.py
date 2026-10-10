"""Frame rendering along camera paths (flat mono/stereo and VR180)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import (
    CancelToken,
    Intrinsics,
    Layer,
    LayeredImage,
    Pose,
    ProgressReporter,
    StereoParams,
)
from vrphoto.geometry import projection as gproj
from vrphoto.geometry import stereo as gstereo
from vrphoto.geometry import warp as gwarp
from vrphoto.video import render
from vrphoto.video.paths import CameraPath, make_path
from vrphoto.video.render import (
    eye_K,
    eye_pose,
    overscan_K,
    render_path_frames,
    render_vr180_frames,
)


def _li(scene) -> LayeredImage:
    h, w = scene.inv_z.shape
    return LayeredImage([Layer(scene.img, scene.inv_z, np.ones((h, w), np.float32))], scene.K)


class FakeRenderer:
    """Records (pose, K) of every render_filled call; returns a frame encoding the call index."""

    def __init__(self, with_cancel: bool = True):
        self.calls: list[tuple[Pose, Intrinsics, object]] = []
        self.with_cancel = with_cancel

    def __call__(self, li, pose, dst_K=None, **kw):
        K = dst_K if dst_K is not None else li.K
        self.calls.append((pose, K, kw.get("cancel")))
        return np.full((K.height, K.width, 3), len(self.calls) % 256, np.uint8)


@pytest.fixture
def fake_render(monkeypatch):
    fr = FakeRenderer()

    def render_filled(li, pose, dst_K=None, *, iters=6, cancel=None):
        return fr(li, pose, dst_K, cancel=cancel)

    monkeypatch.setattr(gwarp, "render_filled", render_filled)
    return fr


@pytest.fixture
def small_li():
    K = Intrinsics.from_hfov(64, 48, 60.0)
    img = np.zeros((48, 64, 3), np.uint8)
    inv = np.full((48, 64), 0.25, np.float32)
    return LayeredImage([Layer(img, inv, np.ones((48, 64), np.float32))], K)


# --------------------------------------------------------------------------- helpers
def test_overscan_and_eye_helpers():
    K = Intrinsics(100.0, 110.0, 32.0, 24.0, 64, 48)
    Ko = overscan_K(K, 0.04)
    assert (Ko.fx, Ko.fy, Ko.cx, Ko.cy, Ko.width, Ko.height) == pytest.approx((104.0, 114.4, 32, 24, 64, 48))
    assert overscan_K(K, 0.0) == K
    # algorithm B: cx_L = cx - f*B/(2*Zc), cx_R = cx + f*B/(2*Zc)
    assert eye_K(K, -0.03, 2.0).cx == pytest.approx(32.0 - 100.0 * 0.03 / 2.0)
    assert eye_K(K, 0.03, 2.0).cx == pytest.approx(32.0 + 100.0 * 0.03 / 2.0)
    assert eye_K(K, 0.03, math.inf) == K
    center = Pose.look(yaw_deg=30.0, t=(0.1, 0.0, 0.2))
    left = eye_pose(center, -0.03)
    np.testing.assert_allclose(left.R, center.R)
    np.testing.assert_allclose(left.t, np.asarray(center.t) - 0.03 * np.asarray(center.R)[:, 0])


# --------------------------------------------------------------------------- flat (fakes)
def test_mono_uses_path_poses_and_overscan(fake_render, small_li):
    path = make_path("orbit", n_frames=7, amplitude_m=0.05)
    frames = list(render_path_frames(small_li, path))
    assert len(frames) == 7 and len(fake_render.calls) == 7
    Kexp = overscan_K(small_li.K, 0.04)
    for (pose, K, _c), ref in zip(fake_render.calls, path.poses):
        assert K == Kexp
        np.testing.assert_array_equal(pose.R, ref.R)
        np.testing.assert_array_equal(pose.t, ref.t)
    assert frames[0].shape == (48, 64, 3)


def test_out_K_and_zero_overscan(fake_render, small_li):
    outK = Intrinsics.from_hfov(128, 96, 60.0)
    frames = list(render_path_frames(small_li, make_path("dolly", n_frames=3), out_K=outK, overscan=0.0))
    assert frames[0].shape == (96, 128, 3)
    assert all(K == outK for _p, K, _c in fake_render.calls)


def test_stereo_sbs_eyes_follow_path_pose(fake_render, small_li, monkeypatch):
    monkeypatch.setattr(gstereo, "resolve_stereo", lambda li, params, K: (0.06, 2.0))
    path = make_path("orbit", n_frames=4, amplitude_m=0.1)  # rotated poses: eyes use the pose x axis
    frames = list(render_path_frames(small_li, path, layout="sbs", stereo=StereoParams()))
    assert len(frames) == 4 and frames[0].shape == (48, 128, 3)
    Ko = overscan_K(small_li.K, 0.04)
    calls = fake_render.calls
    assert len(calls) == 8
    for i, center in enumerate(path.poses):
        (pl, kl, _), (pr, kr, _) = calls[2 * i], calls[2 * i + 1]
        R = np.asarray(center.R)
        np.testing.assert_allclose(pl.t, np.asarray(center.t) - 0.03 * R[:, 0], atol=1e-12)
        np.testing.assert_allclose(pr.t, np.asarray(center.t) + 0.03 * R[:, 0], atol=1e-12)
        np.testing.assert_allclose(pl.R, R)
        assert kl.cx == pytest.approx(Ko.cx - Ko.fx * 0.06 / (2 * 2.0))
        assert kr.cx == pytest.approx(Ko.cx + Ko.fx * 0.06 / (2 * 2.0))
        assert kl.fx == kr.fx == pytest.approx(Ko.fx)
        # left eye image (call 2i+1) on the left half, right eye (2i+2) on the right
        assert frames[i][0, 0, 0] == (2 * i + 1) % 256 and frames[i][0, -1, 0] == (2 * i + 2) % 256


def test_stereo_tb_vr_mode_and_swap(fake_render, small_li):
    p = StereoParams(mode="vr", ipd_m=0.064, swap_eyes=True)
    frames = list(render_path_frames(small_li, make_path("circle", n_frames=2), layout="tb", stereo=p))
    assert frames[0].shape == (96, 64, 3)
    (pl, kl, _), (pr, kr, _) = fake_render.calls[:2]
    assert kl == kr == overscan_K(small_li.K, 0.04)  # convergence at infinity: no shift
    assert np.asarray(pr.t)[0] - np.asarray(pl.t)[0] == pytest.approx(0.064)
    # swapped: the right-eye render (call 2) is on top
    assert frames[0][0, 0, 0] == 2 and frames[0][-1, 0, 0] == 1


def test_default_stereo_params_and_other_layouts(fake_render, small_li):
    path = make_path("swing", n_frames=2)
    assert next(render_path_frames(small_li, path, layout="half_sbs")).shape == (48, 64, 3)
    assert next(render_path_frames(small_li, path, layout="anaglyph")).shape == (48, 64, 3)


def test_local_stereo_resolution_matches_geometry(two_plane_scene):
    li = _li(two_plane_scene)
    K = overscan_K(li.K, 0.04)
    cases = [
        StereoParams(),
        StereoParams(convergence_m=3.0, strength=2.0),
        StereoParams(max_parallax_pct=0.5),
        StereoParams(max_parallax_pct=0.0),
        StereoParams(mode="vr", ipd_m=0.07),
        StereoParams(convergence_m=1.0, strength=3.0, screen_width_m=6.0),  # divergence limit
        StereoParams(convergence_m=1.0, strength=3.0, screen_width_m=0.0),
    ]
    for p in cases:
        B1, z1 = gstereo.resolve_stereo(li, p, K)
        B2, z2 = render._resolve_stereo_local(li, p, K)
        assert B2 == pytest.approx(B1, rel=1e-6)
        assert z2 == pytest.approx(z1, rel=1e-6)
    for bad in (StereoParams(ipd_m=0), StereoParams(mode="weird"), StereoParams(convergence_m=-1.0)):
        with pytest.raises(InputError):
            render._resolve_stereo_local(li, bad, K)


def test_fallback_without_geometry_resolve(fake_render, small_li, monkeypatch):
    monkeypatch.delattr(gstereo, "resolve_stereo")
    frames = list(render_path_frames(small_li, make_path("dolly", n_frames=2), layout="sbs"))
    assert len(frames) == 2 and len(fake_render.calls) == 4


def test_cancel_and_progress(fake_render, small_li):
    tok = CancelToken()
    seen: list[float] = []
    rep = ProgressReporter(lambda f, m: seen.append(f))
    gen = render_path_frames(small_li, make_path("circle", n_frames=10), cancel=tok, progress=rep)
    next(gen)
    next(gen)
    assert fake_render.calls[0][2] is tok  # forwarded to the warp
    tok.cancel()
    with pytest.raises(Cancelled):
        next(gen)
    assert len(fake_render.calls) == 2
    assert seen == sorted(seen) and seen[0] == 0.0
    seen.clear()
    list(render_path_frames(small_li, make_path("circle", n_frames=4), progress=rep))
    assert seen[-1] == 1.0


def test_renderer_without_cancel_parameter(monkeypatch, small_li):
    calls = []

    def render_filled(li, pose, dst_K=None):  # contract signature only
        calls.append(pose)
        return np.zeros((dst_K.height, dst_K.width, 3), np.uint8)

    monkeypatch.setattr(gwarp, "render_filled", render_filled)
    frames = list(render_path_frames(small_li, make_path("dolly", n_frames=3), cancel=CancelToken()))
    assert len(frames) == 3 and len(calls) == 3


@pytest.mark.parametrize(
    "kwargs",
    [{"layout": "interlaced"}, {"overscan": -0.1}, {"overscan": 0.9}],
)
def test_flat_validation(fake_render, small_li, kwargs):
    with pytest.raises(InputError):
        list(render_path_frames(small_li, make_path("dolly", n_frames=2), **kwargs))


def test_empty_inputs(fake_render, small_li):
    with pytest.raises(InputError):
        list(render_path_frames(small_li, CameraPath([], 30.0)))
    with pytest.raises(InputError):
        list(render_path_frames(LayeredImage([], small_li.K), make_path("dolly", n_frames=2)))
    with pytest.raises(InputError):
        list(render_vr180_frames(small_li, CameraPath([], 30.0), eye_px=64))


# --------------------------------------------------------------------------- VR180 (fakes)
def test_vr180_with_fakes(fake_render, small_li, monkeypatch):
    seen = []

    def vr180_frame(left, right, K, eye_px, *, size_mult=1.0, fill="blur", layout="sbs"):
        seen.append((left, right, K, eye_px, size_mult, fill, layout))
        return np.zeros((eye_px, 2 * eye_px, 3), np.uint8)

    monkeypatch.setattr(gproj, "vr180_frame", vr180_frame)
    path = make_path("orbit", n_frames=3, amplitude_m=0.1)
    frames = list(render_vr180_frames(small_li, path, eye_px=96, size_mult=1.3, fill="black"))
    assert len(frames) == 3 and frames[0].shape == (96, 192, 3)
    for i, center in enumerate(path.poses):
        (pl, kl, _), (pr, kr, _) = fake_render.calls[2 * i : 2 * i + 2]
        assert kl == kr == small_li.K  # true angular size: no overscan, no convergence shift
        R = np.asarray(center.R)
        np.testing.assert_allclose(pl.t, np.asarray(center.t) - 0.0315 * R[:, 0], atol=1e-12)
        np.testing.assert_allclose(pr.t, np.asarray(center.t) + 0.0315 * R[:, 0], atol=1e-12)
        left, right, K, eye, sm, fill, lay = seen[i]
        assert K == small_li.K and eye == 96 and sm == 1.3 and fill == "black" and lay == "sbs"
        assert left[0, 0, 0] == (2 * i + 1) % 256 and right[0, 0, 0] == (2 * i + 2) % 256
    fake_render.calls.clear()
    seen.clear()
    list(
        render_vr180_frames(
            small_li, path, eye_px=32, stereo=StereoParams(ipd_m=0.07, swap_eyes=True), layout="tb"
        )
    )
    left, right, *_rest, lay = seen[0]
    assert lay == "tb" and left[0, 0, 0] == 2 and right[0, 0, 0] == 1
    (pl, _, _), (pr, _, _) = fake_render.calls[:2]
    assert np.linalg.norm(np.asarray(pr.t) - np.asarray(pl.t)) == pytest.approx(0.07)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eye_px": 4},
        {"eye_px": "big"},
        {"eye_px": 64, "layout": "cross"},
        {"eye_px": 64, "size_mult": 0},
        {"eye_px": 64, "stereo": StereoParams(ipd_m=-1)},
    ],
)
def test_vr180_validation(fake_render, small_li, kwargs):
    with pytest.raises(InputError):
        list(render_vr180_frames(small_li, make_path("dolly", n_frames=2), **kwargs))


# --------------------------------------------------------------------------- real geometry
def test_identity_path_reproduces_photo(two_plane_scene, psnr_fn):
    li = _li(two_plane_scene)
    path = make_path("dolly", n_frames=2, amplitude_m=0.0)
    frame = next(render_path_frames(li, path, overscan=0.0))
    assert psnr_fn(frame, two_plane_scene.img) > 35


def test_overscan_is_a_centred_zoom(two_plane_scene, psnr_fn):
    """Golden: with a static camera, 4% overscan == magnifying about the principal point."""
    li = _li(two_plane_scene)
    K = li.K
    frame = next(render_path_frames(li, make_path("dolly", n_frames=1, amplitude_m=0.0), overscan=0.04))
    s = 1.04
    xs = ((np.arange(K.width) - K.cx) / s + K.cx).astype(np.float32)
    ys = ((np.arange(K.height) - K.cy) / s + K.cy).astype(np.float32)
    mx, my = np.meshgrid(xs, ys)
    ref = cv2.remap(two_plane_scene.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    assert psnr_fn(frame, ref) > 26
    assert psnr_fn(frame, two_plane_scene.img) < psnr_fn(frame, ref)


def test_overscan_hides_frame_edges(two_plane_scene):
    """Golden: with 4% overscan no pose of a 5 cm circle reveals holes at the frame border
    (far plane at 4 m shifts by f*A/Z ~ 3.5 px; overscan margin ~4.6 px); without it does."""
    li = _li(two_plane_scene)
    path = make_path("circle", n_frames=8, amplitude_m=0.05)
    ring = np.ones(two_plane_scene.inv_z.shape, bool)
    ring[2:-2, 2:-2] = False

    def border_holes(K):
        return max(int(np.count_nonzero(gwarp.render_view(li, p, K).alpha[ring] < 0.5)) for p in path.poses)

    assert border_holes(overscan_K(li.K, 0.04)) == 0
    assert border_holes(li.K) > 0


def test_real_stereo_zero_parallax_at_convergence(two_plane_scene, psnr_fn):
    """Golden: the far plane placed at the convergence distance is identical in both eyes."""
    li = _li(two_plane_scene)
    params = StereoParams(convergence_m=two_plane_scene.z_far, max_parallax_pct=0.0)
    frame = next(
        render_path_frames(li, make_path("dolly", n_frames=1, amplitude_m=0.0), stereo=params, layout="sbs")
    )
    h, w = two_plane_scene.inv_z.shape
    assert frame.shape == (h, 2 * w, 3)
    left, right = frame[:, :w], frame[:, w:]
    far = ~two_plane_scene.fg_mask
    far = cv2.erode(far.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    far[:10], far[-10:], far[:, :10], far[:, -10:] = False, False, False, False
    assert psnr_fn(left, right, far) > 30
    # the near square has crossed parallax: it appears further right in the left eye
    diff = np.abs(left.astype(int) - right.astype(int)).sum(-1)
    assert diff[two_plane_scene.fg_mask].mean() > 5 * max(diff[far].mean(), 0.5)


def test_real_vr180_frames(two_plane_scene):
    li = _li(two_plane_scene)
    frames = list(render_vr180_frames(li, make_path("circle", n_frames=3, amplitude_m=0.03), eye_px=96))
    assert len(frames) == 3
    f = frames[0]
    assert f.shape == (96, 192, 3) and f.dtype == np.uint8
    left, right = f[:, :96].astype(int), f[:, 96:].astype(int)
    assert np.abs(left - right).mean() < 40  # same view up to parallax
    assert not np.array_equal(frames[0], frames[1])  # the camera moves


def test_generators_validate_eagerly(fake_render, small_li):
    path = make_path("dolly", n_frames=2)
    with pytest.raises(InputError):
        render_path_frames(small_li, path, layout="bogus")
    with pytest.raises(InputError):
        render_vr180_frames(small_li, path, eye_px=2)
    assert not fake_render.calls  # nothing rendered yet
    gen = render_path_frames(small_li, path)
    assert not fake_render.calls  # lazy
    next(gen)
    assert len(fake_render.calls) == 1


# --------------------------------------------------------------------------- conventions (golden)
def _shift(a: np.ndarray, b: np.ndarray, rows: slice, cols: range, search: range) -> int:
    """Integer ``s`` minimising ``|a[:, x] - b[:, x - s]|`` over ``cols`` (content of ``b``
    appears ``s`` px further right in ``a``)."""
    A = a[rows][:, list(cols)].astype(float)
    errs = {s: np.abs(A - b[rows][:, [c - s for c in cols]].astype(float)).mean() for s in search}
    return min(errs, key=errs.get)


def test_stereo_parallax_sign_and_magnitude(two_plane_scene):
    """Golden (algorithm B): u_L - u_R = f*B*(1/Z - 1/Zc) > 0 for the near square (crossed,
    in front of the screen) and 0 at the convergence plane; left eye on the left half."""
    sc = two_plane_scene
    li = _li(sc)
    params = StereoParams(convergence_m=sc.z_far, max_parallax_pct=0.0)
    frame = next(render_path_frames(li, CameraPath([Pose.identity()], 30.0), stereo=params, layout="sbs"))
    w = sc.img.shape[1]
    left, right = frame[:, :w], frame[:, w:]
    f = sc.K.fx * 1.04  # overscan
    expect = f * params.ipd_m * (1.0 / sc.z_near - 1.0 / sc.z_far)  # ~7.6 px
    near = _shift(left, right, slice(95, 145), range(140, 180), range(-15, 16))
    assert near > 0 and abs(near - expect) <= 1.0
    assert _shift(left, right, slice(20, 60), range(40, 280), range(-15, 16)) == 0
    # swap_eyes puts the right eye on the left: the sign flips
    swapped = next(
        render_path_frames(
            li,
            CameraPath([Pose.identity()], 30.0),
            stereo=StereoParams(convergence_m=sc.z_far, max_parallax_pct=0.0, swap_eyes=True),
            layout="sbs",
        )
    )
    assert _shift(swapped[:, :w], swapped[:, w:], slice(95, 145), range(140, 180), range(-15, 16)) == -near


def test_path_translation_sign(two_plane_scene):
    """Golden: a path pose moved RIGHT (x > 0) shifts the content LEFT by f*x/Z (cv frame)."""
    sc = two_plane_scene
    li = _li(sc)
    x = 0.08
    path = make_path("custom", n_frames=1, loop=False, keyframes=[{"t": 0, "x": x}])
    frame = next(render_path_frames(li, path, overscan=0.0))
    far = _shift(frame, sc.img, slice(20, 60), range(40, 280), range(-12, 4))
    near = _shift(frame, sc.img, slice(95, 145), range(140, 180), range(-20, 4))
    assert abs(far - (-sc.K.fx * x / sc.z_far)) <= 1.0  # ~ -5.5 px
    assert abs(near - (-sc.K.fx * x / sc.z_near)) <= 1.0  # ~ -14.8 px


def test_vr180_crossed_parallax_left_eye_left(two_plane_scene):
    """VR mode (convergence at infinity): the near square sits further right in the left
    (left-half) eye than in the right eye; the far plane barely moves."""
    sc = two_plane_scene
    li = _li(sc)
    eye = 256
    f = next(render_vr180_frames(li, CameraPath([Pose.identity()], 30.0), eye_px=eye, fill="black"))
    left, right = f[:, :eye], f[:, eye:]
    c = eye // 2
    near = _shift(left, right, slice(c - 10, c + 10), range(c - 12, c + 12), range(-10, 11))
    # the photo spans ~85x66 px around the centre; the near square ~28x22 px
    far = _shift(left, right, slice(c - 30, c - 20), range(c - 38, c + 38), range(-10, 11))
    assert near >= 2 and abs(far) <= 1 and near > far


# --------------------------------------------------------------------------- VR180 eye resolution
def test_vr180_eye_K_density_and_fov():
    K = Intrinsics.from_hfov(4000, 3000, 65.0)
    Ke = render.vr180_eye_K(K, 3840)
    need = render.VR180_DENSITY_MARGIN * 3840 / math.pi
    assert Ke.fx == pytest.approx(need, rel=0.01) and Ke.width < K.width
    assert Ke.hfov_deg == pytest.approx(K.hfov_deg, abs=0.05)
    assert Ke.vfov_deg == pytest.approx(K.vfov_deg, abs=0.05)
    # pixel-centre exact rescale of the principal point
    sx = Ke.width / K.width
    assert Ke.cx == pytest.approx((K.cx + 0.5) * sx - 0.5)
    # size_mult > 1 enlarges the field of view -> more source pixels needed
    assert render.vr180_eye_K(K, 3840, 1.5).width > Ke.width
    assert render.vr180_eye_K(K, 3840, 0.5).width < Ke.width
    # a photo that is not denser than needed is rendered as is
    small = Intrinsics.from_hfov(800, 600, 65.0)
    assert render.vr180_eye_K(small, 3840) is small
    # portrait photos: the limiting (vertical) density is respected
    P = Intrinsics.from_hfov(3000, 4000, 50.0)
    Pe = render.vr180_eye_K(P, 2048)
    assert min(Pe.fx, Pe.fy) >= 0.99 * render.VR180_DENSITY_MARGIN * 2048 / math.pi


def test_vr180_renders_eyes_at_needed_resolution(fake_render, monkeypatch):
    K = Intrinsics.from_hfov(640, 480, 60.0)
    li = LayeredImage(
        [
            Layer(
                np.zeros((480, 640, 3), np.uint8),
                np.full((480, 640), 0.3, np.float32),
                np.ones((480, 640), np.float32),
            )
        ],
        K,
    )
    seen = []

    def vr180_frame(left, right, K, eye_px, *, size_mult=1.0, fill="blur", layout="sbs"):
        seen.append((left.shape, K))
        return np.zeros((eye_px, 2 * eye_px, 3), np.uint8)

    monkeypatch.setattr(gproj, "vr180_frame", vr180_frame)
    list(render_vr180_frames(li, make_path("dolly", n_frames=2), eye_px=128, size_mult=1.2))
    Ke = render.vr180_eye_K(K, 128, 1.2)
    assert Ke.width < 640
    assert all(k == Ke for _p, k, _c in fake_render.calls)
    assert all(shape == (Ke.height, Ke.width, 3) and k == Ke for shape, k in seen)


def test_vr180_reduced_eye_resolution_matches_full(two_plane_scene, psnr_fn, monkeypatch):
    """Rendering the eyes at the needed density gives the same VR180 frame as rendering them
    at the full photo resolution (the equirect is the bottleneck)."""
    li = _li(two_plane_scene)
    path = CameraPath([Pose.translate(0.02, 0.0, 0.0)], 30.0)
    eye = 192
    fast = next(render_vr180_frames(li, path, eye_px=eye))
    assert render.vr180_eye_K(li.K, eye).width < li.K.width
    monkeypatch.setattr(render, "vr180_eye_K", lambda K, e, s=1.0: K)
    full = next(render_vr180_frames(li, path, eye_px=eye))
    assert psnr_fn(fast, full) > 38


# --------------------------------------------------------------------------- misc
@pytest.mark.parametrize(
    ("layout", "swap", "expect_x"), [("left", False, -1), ("right", False, 1), ("left", True, 1)]
)
def test_single_eye_layouts_render_one_eye(fake_render, small_li, monkeypatch, layout, swap, expect_x):
    monkeypatch.setattr(gstereo, "resolve_stereo", lambda li, params, K: (0.06, 2.0))
    frames = list(
        render_path_frames(
            small_li, make_path("dolly", n_frames=3), layout=layout, stereo=StereoParams(swap_eyes=swap)
        )
    )
    assert len(frames) == 3 and len(fake_render.calls) == 3  # not 6
    Ko = overscan_K(small_li.K, 0.04)
    for (pose, K, _c), center in zip(fake_render.calls, make_path("dolly", n_frames=3).poses):
        assert np.asarray(pose.t)[0] - np.asarray(center.t)[0] == pytest.approx(expect_x * 0.03)
        assert K.cx == pytest.approx(Ko.cx + expect_x * Ko.fx * 0.06 / (2 * 2.0))
    assert frames[0].shape == (48, 64, 3)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_invalid_pose_rejected(fake_render, small_li, bad):
    poses = make_path("dolly", n_frames=3).poses
    poses[1] = Pose(np.eye(3), np.array([0.0, bad, 0.0]))
    with pytest.raises(InputError, match="frame 2"):
        render_path_frames(small_li, CameraPath(poses, 30.0))
    with pytest.raises(InputError):
        render_vr180_frames(small_li, CameraPath([Pose(np.eye(2), np.zeros(3))], 30.0), eye_px=64)
    assert not fake_render.calls
