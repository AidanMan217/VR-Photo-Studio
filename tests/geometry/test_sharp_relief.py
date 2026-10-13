"""geometry/sharp_relief.py: the back view's relief from a SHARP cloud of that view (M6).

The synthetic person (synth_person.py) turned 180 deg plays the back view: its SHARP-like
cloud (two layers, wall / floor, ray-aligned cards at the outline) is what SHARP would make
of an AI back view, and the analytic ray cast gives the true inverse depth."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import synth_person as sp

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, SplatCloud
from vrphoto.geometry.sharp_relief import sharp_relief
from vrphoto.geometry.splat import subset_splats, write_splats

W, H, G, OUT = 192, 256, 160, 128


@pytest.fixture(scope="module")
def back_scene():
    body = sp.make_body(yaw_deg=180.0)  # the camera sees the person's back
    cam = sp.make_cam(W, H, grid=G)
    return body, cam, sp.sharp_like_scene(body, cam, G=G)


@pytest.fixture(scope="module")
def truth(back_scene):
    """True disparity at the output pixel centres, normalised over the body like the relief."""
    body, cam, _ = back_scene
    uu, vv = np.meshgrid((np.arange(OUT) + 0.5) * W / OUT, (np.arange(OUT) + 0.5) * H / OUT)
    t, _, _, kind = body.raycast_scene(np.zeros(3), sp.pixel_rays(cam, uu.ravel(), vv.ravel()))
    on = (kind == sp.KIND_BODY).reshape(OUT, OUT)
    disp = 1.0 / t.reshape(OUT, OUT)
    lo, hi = np.percentile(disp[on], [0.5, 99.5])
    inner = cv2.erode(on.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return np.clip((disp - lo) / (hi - lo), 0.0, 1.0), on, inner


def _relief(scene, cloud=None, **kw):
    body, cam, s = scene
    kw.setdefault("alpha", s.alpha)
    return sharp_relief(s.sc if cloud is None else cloud, f_px=cam.f, width=W, height=H, out_size=OUT, **kw)


def test_relief_matches_the_true_back_depth(back_scene, truth, tmp_path):
    body, cam, s = back_scene
    gt, on, inner = truth
    ply = write_splats(s.sc, tmp_path / "back_sharp.ply")
    diag: dict = {}
    d = sharp_relief(ply, f_px=cam.f, width=W, height=H, out_size=OUT, alpha=s.alpha, diag=diag)
    assert d.shape == (OUT, OUT) and d.dtype == np.float32
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0
    err = np.abs(d - gt)[inner]
    assert np.corrcoef(d[inner], gt[inner])[0, 1] > 0.97
    assert err.mean() < 0.08 and np.percentile(err, 95) < 0.15
    # 1 = nearest: the nearest 5 % of the back is far brighter than the farthest 5 %
    near = gt[inner] >= np.percentile(gt[inner], 95)
    far = gt[inner] <= np.percentile(gt[inner], 5)
    assert d[inner][near].mean() > d[inner][far].mean() + 0.5
    assert diag["matte"] and diag["n_points"] == len(s.sc) and 0 < diag["n_used"] <= diag["n_points"]
    assert diag["coverage"] > 0.9 and 0.0 < diag["radius_px"] <= 3.0 and not diag["flat"]
    z0, z1 = diag["z_range_m"]
    assert 2.2 < z0 < z1 < 2.7  # the person's back, not the wall at 3.5 m
    # the cloud itself instead of its PLY: the same map
    np.testing.assert_array_equal(_relief(back_scene), d)


def test_the_bag_on_the_back_is_nearer_than_the_torso(back_scene):
    """Asymmetric detail survives: seen from behind, the bag (slightly behind the body's right
    hip) sticks out toward the camera."""
    body, cam, s = back_scene
    d = _relief(back_scene)
    first = (s.layer == 0) & (s.kind == sp.KIND_BODY)

    def mean_d(part):
        sel = first & (s.part == body.names.index(part))
        P = s.sc.means[sel].astype(np.float64)
        u = (cam.f * P[:, 0] / P[:, 2] + W / 2) * OUT / W
        v = (cam.f * P[:, 1] / P[:, 2] + H / 2) * OUT / H
        return float(d[v.astype(int), u.astype(int)].mean())

    assert mean_d("bag") > mean_d("torso") + 0.1


def test_without_a_matte_the_background_is_far(back_scene, truth):
    body, cam, s = back_scene
    gt, on, inner = truth
    diag: dict = {}
    d = _relief(back_scene, alpha=None, diag=diag)
    assert not diag["matte"] and diag["region_px"] == OUT * OUT  # SHARP covers the whole picture
    assert np.corrcoef(d[inner], gt[inner])[0, 1] > 0.95
    wall = cv2.resize((s.depth > 3.0).astype(np.uint8), (OUT, OUT), interpolation=cv2.INTER_NEAREST) > 0
    wall &= cv2.erode((~on).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert wall.sum() > 100 and d[wall].mean() < d[inner].mean() - 0.2


def test_see_through_gaps_and_outline_cards_are_filled(back_scene, truth):
    """Remove 15 % of the visible body Gaussians (the layer behind and the wall show through)
    - the relief barely changes; the ray-aligned cards at the outline do not pull it back."""
    body, cam, s = back_scene
    gt, on, inner = truth
    rng = np.random.default_rng(1)
    drop = (s.layer == 0) & (s.kind == sp.KIND_BODY) & (rng.random(len(s.sc)) < 0.15)
    holey = subset_splats(s.sc, np.flatnonzero(~drop))
    diag: dict = {}
    d = _relief(back_scene, holey, diag=diag)
    err = np.abs(d - gt)[inner]
    assert err.mean() < 0.1 and np.percentile(err, 95) < 0.2
    assert diag["gap_px"] > 0
    assert (s.kind == sp.KIND_CARD).sum() > 0  # the fixture really has outline cards


def test_translucent_floaters_are_ignored(back_scene):
    body, cam, s = back_scene
    rng = np.random.default_rng(2)
    n = 400
    z = np.full(n, 1.2)
    u, v = rng.uniform(0.3 * W, 0.7 * W, n), rng.uniform(0.2 * H, 0.8 * H, n)
    P = np.stack([(u - W / 2) * z / cam.f, (v - H / 2) * z / cam.f, z], axis=1).astype(np.float32)
    floaters = SplatCloud(
        means=P,
        scales=np.full((n, 3), 0.01, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        colors=np.full((n, 3), 0.5, np.float32),
        opacity=np.full(n, 0.1, np.float32),
    )
    from vrphoto.geometry.splat import concat_splats

    with_floaters = concat_splats([s.sc, floaters])
    np.testing.assert_array_equal(_relief(back_scene, with_floaters), _relief(back_scene))
    assert not np.array_equal(_relief(back_scene, with_floaters, min_opacity=0.05), _relief(back_scene))


def test_sizes_and_matte_formats(back_scene):
    body, cam, s = back_scene
    d = sharp_relief(s.sc, f_px=cam.f, width=W, height=H, out_size=(96, 64), alpha=s.alpha)
    assert d.shape == (96, 64)
    small = cv2.resize(s.alpha, (OUT, OUT), interpolation=cv2.INTER_AREA)  # matte in the output's frame
    u8 = np.clip(np.rint(s.alpha * 255), 0, 255).astype(np.uint8)
    a = _relief(back_scene, alpha=small)
    b = _relief(back_scene, alpha=u8)
    c = _relief(back_scene, alpha=s.alpha[..., None])
    ref = _relief(back_scene)
    assert np.abs(a - ref).mean() < 0.02 and np.abs(b - ref).mean() < 0.005 and np.array_equal(c, ref)


def test_flat_cloud_gives_a_flat_map():
    n = 4000
    rng = np.random.default_rng(3)
    u, v = rng.uniform(0, 64, n), rng.uniform(0, 64, n)
    z = np.full(n, 2.0)
    f = 80.0
    P = np.stack([(u - 32) * z / f, (v - 32) * z / f, z], axis=1).astype(np.float32)
    sc = SplatCloud(means=P, scales=np.full((n, 3), 0.02, np.float32),
                    quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
                    colors=np.full((n, 3), 0.5, np.float32), opacity=np.full(n, 0.9, np.float32))
    diag: dict = {}
    d = sharp_relief(sc, f_px=f, width=64, height=64, out_size=32, diag=diag)
    assert diag["flat"] and d.shape == (32, 32) and not d.any()


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"f_px": 0}, "focal length"),
        ({"f_px": float("nan")}, "focal length"),
        ({"width": 0}, "picture width"),
        ({"height": "tall"}, "picture height"),
        ({"out_size": 1}, "relief size"),
        ({"out_size": "big"}, "relief size"),
        ({"out_size": (64, 99999)}, "relief size"),
        ({"alpha": np.zeros((H, W), np.float32)}, "matte is empty"),
        ({"alpha": np.zeros(5)}, "H x W map"),
        ({"f_px": 1e6}, "no visible surface|does not cover"),  # another picture's camera
    ],
)
def test_bad_arguments(back_scene, kw, match):
    body, cam, s = back_scene
    args = {"f_px": cam.f, "width": W, "height": H, "out_size": OUT, "alpha": s.alpha, **kw}
    with pytest.raises(InputError, match=match):
        sharp_relief(s.sc, **args)


def test_unusable_clouds_and_files(back_scene, tmp_path):
    body, cam, s = back_scene
    behind = SplatCloud(means=s.sc.means * np.array([1, 1, -1], np.float32), scales=s.sc.scales,
                        quats=s.sc.quats, colors=s.sc.colors, opacity=s.sc.opacity)
    with pytest.raises(InputError, match="no visible surface"):
        sharp_relief(behind, f_px=cam.f, width=W, height=H, out_size=OUT)
    # a matte where SHARP put nothing (the subject is elsewhere in the picture)
    corner = np.zeros((H, W), np.float32)
    corner[:4, :4] = 1.0
    empty_corner = subset_splats(s.sc, np.flatnonzero((s.sc.means[:, 0] > 0) | (s.sc.means[:, 1] > 0)))
    with pytest.raises(InputError, match="does not cover the subject"):
        sharp_relief(empty_corner, f_px=cam.f, width=W, height=H, out_size=OUT, alpha=corner)
    with pytest.raises(InputError, match="not found"):
        sharp_relief(tmp_path / "missing.ply", f_px=cam.f, width=W, height=H, out_size=OUT)


def test_cancel(back_scene):
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        _relief(back_scene, cancel=tok)
