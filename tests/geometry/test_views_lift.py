"""SHARP 360 view lifting (geometry/views_lift.py, design 5.2 test_views_lift.py, nominal views):
the levelled input, the views folder, segmentation, the back lock, side views, the hull, the
relief and the contact sheet, on the analytic person of tests/geometry/synth_person.py."""

from __future__ import annotations

import dataclasses
import json
import math
import time

import cv2
import numpy as np
import pytest
import synth_person as sp

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken
from vrphoto.geometry import register as rg
from vrphoto.geometry import views_lift as vl
from vrphoto.geometry.back_mould import inflation_prior

S = 256  # levelled picture size used by most tests (the app uses 512)
MAGENTA = np.array([255, 0, 255], np.uint8)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def body() -> sp.Body:
    return sp.make_body()


@pytest.fixture(scope="module")
def cam():
    return sp.make_cam(240, 320)


@pytest.fixture(scope="module")
def front(body, cam):
    return sp.synth_front_model(body, cam, grid_long=256)


@pytest.fixture(scope="module")
def canon(body, cam):
    return sp.synth_canon(body, cam, thickness=0.11)


@pytest.fixture(scope="module")
def photo(body, cam):
    rgb, alpha, _, _ = sp.render_photo(body, cam)
    return rgb, alpha


@pytest.fixture(scope="module")
def ortho(photo, front, canon) -> vl.OrthoInput:
    return vl.levelled_ortho_input(*photo, front, canon, size=S, ref=2 * S)


@pytest.fixture(scope="module")
def prior(canon):
    """The canonical frame as the pipeline knows it before side views: the orbit-axis depth
    is only a prior, here 6 cm too near."""
    zh = canon.R[:, 2]
    return dataclasses.replace(canon, z_axis=canon.z_axis - 0.06, c=canon.c - 0.06 * zh)


@pytest.fixture(scope="module")
def ortho_prior(photo, front, prior) -> vl.OrthoInput:
    return vl.levelled_ortho_input(*photo, front, prior, size=S, ref=2 * S)


def nominal(
    body,
    canon,
    phi: float,
    *,
    m: float = 0.006,
    N: int = 360,
    scale: float = 1.0,
    roll: float = 0.0,
    shift=(0.0, 0.0),
    paint: dict[str, np.ndarray] | None = None,
    name: str | None = None,
) -> vl.GenView:
    """A "user photo" of the body from azimuth ``phi``: an orthographic view on grey, then
    scaled / rolled about the picture centre and shifted (the photo's framing is unknown to
    the code); ``paint`` recolours body parts (name -> colour) to track them."""
    rgb, alpha, _ = sp.ortho_view(body, canon, phi, m=m, N=N)
    if paint:
        parts = sp.ortho_view_parts(body, canon, phi, m=m, N=N)
        for part, colour in paint.items():
            rgb[parts == body.part(part)] = colour
    M = cv2.getRotationMatrix2D((N / 2.0, N / 2.0), roll, scale)
    M[:, 2] += shift
    rgb = cv2.warpAffine(rgb, M, (N, N), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    alpha = cv2.warpAffine(alpha, M, (N, N), flags=cv2.INTER_LINEAR, borderValue=0)
    return vl.GenView(name or vl._phi_name(phi), phi, rgb, alpha, kind="nominal")


def visible_in_truth(body, canon, sc_means, phi: float, m: float = 0.005, N: int = 480, tol: float = 0.02):
    """``(visible, true colour class)`` of points in the true orthographic view at ``phi``."""
    rgb, _, depth = sp.ortho_view(body, canon, phi, m=m, N=N, ss=1)
    u, v, d = sp.ortho_project(canon, phi, m, N, sc_means)
    iu = np.clip(np.floor(u).astype(int), 0, N - 1)
    iv = np.clip(np.floor(v).astype(int), 0, N - 1)
    vis = np.isfinite(depth[iv, iu]) & (np.abs(d - depth[iv, iu]) < tol)
    return vis, sp.classify_rgb(rgb[iv, iu])


# --------------------------------------------------------------------------- levelled input
def _centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def _spread(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.std(), ys.std()])


def test_levelled_input_keeps_scale_and_centre(ortho, body, canon):
    assert ortho.rgba.shape == (S, S, 4) and ortho.rgba.dtype == np.uint8
    assert ortho.rgba_ref.shape == (2 * S, 2 * S, 4) and ortho.alpha0.dtype == bool
    assert ortho.m_ref == pytest.approx(ortho.m_in / 2)
    # the subject's long side fills ~90 % of the picture (robust extent of the visible front)
    assert ortho.m_in * 0.9 * S == pytest.approx(1.69, rel=0.03)
    # the orthographic front view at m_in centred on center_S is the same picture
    x0, y0 = ortho.center_S
    c2 = canon.c + canon.R @ np.array([x0, y0, 0.0])
    truth_rgb, truth_a, _ = sp.ortho_view(body, dataclasses.replace(canon, c=c2), 0.0, m=ortho.m_in, N=S)
    truth = truth_a >= 0.5
    assert rg.silhouette_iou(ortho.alpha0, truth) > 0.94
    assert np.abs(_centroid(ortho.alpha0) - _centroid(truth)).max() < 0.5  # px
    assert np.allclose(_spread(ortho.alpha0), _spread(truth), rtol=0.015)  # scale exact to 1.5 %
    both = ortho.alpha0 & truth
    diff = np.abs(ortho.rgba[..., :3].astype(int) - truth_rgb.astype(int))[both].mean()
    assert diff < 10
    # the reference picture is the same frame at twice the resolution
    a_ref = cv2.resize(ortho.rgba_ref[..., 3], (S, S), interpolation=cv2.INTER_AREA) >= 128
    assert rg.silhouette_iou(a_ref, ortho.alpha0) > 0.97
    # colours are bled past the matte (no black background for the generator)
    outside = ~cv2.dilate(ortho.alpha0.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert ortho.rgba[..., :3][outside].max(axis=1).min() > 20


def test_levelled_input_removes_camera_pitch():
    body = sp.make_body(pitch_deg=20.0)
    cam = sp.make_cam(240, 320)
    front = sp.synth_front_model(body, cam, grid_long=256)
    canon = sp.synth_canon(body, cam)
    rgb, alpha, _, _ = sp.render_photo(body, cam)
    oi = vl.levelled_ortho_input(rgb, alpha, front, canon, size=S, ref=S)
    x0, y0 = oi.center_S
    c2 = canon.c + canon.R @ np.array([x0, y0, 0.0])
    truth = sp.ortho_view(body, dataclasses.replace(canon, c=c2), 0.0, m=oi.m_in, N=S)[1] >= 0.5
    assert rg.silhouette_iou(oi.alpha0, truth) > 0.9
    assert np.allclose(_spread(oi.alpha0), _spread(truth), rtol=0.03)
    # the photo itself is foreshortened: its height / width ratio differs from the truth's
    ys, xs = np.nonzero(alpha >= 0.5)
    photo_ratio = np.ptp(ys) / np.ptp(xs)
    lev_ratio = np.ptp(np.nonzero(oi.alpha0)[0]) / np.ptp(np.nonzero(oi.alpha0)[1])
    true_ratio = np.ptp(np.nonzero(truth)[0]) / np.ptp(np.nonzero(truth)[1])
    assert abs(lev_ratio - true_ratio) < abs(photo_ratio - true_ratio)


def test_levelled_input_validation(photo, front, canon):
    rgb, alpha = photo
    with pytest.raises(InputError):
        vl.levelled_ortho_input(rgb, np.zeros_like(alpha), front, canon)
    with pytest.raises(InputError):
        vl.levelled_ortho_input(rgb, alpha[:-1], front, canon)
    with pytest.raises(InputError):
        vl.levelled_ortho_input(rgb, alpha, front, canon, size=8)
    with pytest.raises(InputError):
        vl.levelled_ortho_input(rgb, alpha, front, canon, size=256, ref=128)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        vl.levelled_ortho_input(rgb, alpha, front, canon, size=64, ref=64, cancel=tok)


# --------------------------------------------------------------------------- views folder
def _touch(folder, *names):
    for n in names:
        (folder / n).write_bytes(b"not read")


def test_views_from_folder_naming_rules(tmp_path):
    _touch(
        tmp_path,
        "IMG_0001.jpg",  # the chosen back photo
        "Right.PNG",
        "LEFT.jpeg",
        "back-right.webp",
        "Back Left.png",
        "front_right.jpg",
        "frontleft.JPG",
        "az030.png",
        "AZ_200.tif",
        "rear.png",  # 180 again: the chosen file wins
        "my_back.png",  # prefixes / suffixes do not count
        "left2.png",
        "right.txt",  # not a picture
        "back_rightish.png",
    )
    got = vl.views_from_folder(tmp_path / "IMG_0001.jpg")
    assert [(phi, p.name) for phi, p in got] == [
        (30.0, "az030.png"),
        (45.0, "front_right.jpg"),
        (90.0, "Right.PNG"),
        (135.0, "back-right.webp"),
        (180.0, "IMG_0001.jpg"),
        (200.0, "AZ_200.tif"),
        (225.0, "Back Left.png"),
        (270.0, "LEFT.jpeg"),
        (315.0, "frontleft.JPG"),
    ]


def test_views_from_folder_own_name_folder_and_json(tmp_path):
    _touch(tmp_path, "left.jpg", "back.png", "behind.jpg")
    # the given file follows its own name; "back" is the 180 view (behind is a duplicate)
    got = vl.views_from_folder(tmp_path / "left.jpg")
    assert [(phi, p.name) for phi, p in got] == [(180.0, "back.png"), (270.0, "left.jpg")]
    # a folder: only named pictures
    assert [p.name for _, p in vl.views_from_folder(tmp_path)] == ["back.png", "left.jpg"]
    # a saved views folder: views.json sets the azimuths of the files it lists
    (tmp_path / "views.json").write_text(json.dumps({"views": [{"file": "left.jpg", "azimuth_deg": 260}]}))
    got = dict((p.name, phi) for phi, p in vl.views_from_folder(tmp_path / "back.png"))
    assert got == {"back.png": 180.0, "left.jpg": 260.0}
    (tmp_path / "views.json").write_text("{broken")
    assert len(vl.views_from_folder(tmp_path / "back.png")) == 2  # unreadable json: names only
    with pytest.raises(InputError, match="not found"):
        vl.views_from_folder(tmp_path / "missing.jpg")
    _touch(tmp_path, "notes.txt")
    with pytest.raises(InputError, match="not a picture"):
        vl.views_from_folder(tmp_path / "notes.txt")


def test_load_views_reads_downscales_and_keeps_cutout_alpha(tmp_path):
    big = np.zeros((300, 1400, 3), np.uint8)
    big[:, :, 1] = 200
    cv2.imwrite(str(tmp_path / "back.jpg"), big)
    rgba = np.zeros((100, 80, 4), np.uint8)
    rgba[..., 2] = 255  # BGRA: red
    rgba[20:80, 20:60, 3] = 255  # a cut-out with a transparent background
    cv2.imwrite(str(tmp_path / "right.png"), rgba)
    (tmp_path / "left.png").write_bytes(b"broken")
    views = vl.load_views(vl.views_from_folder(tmp_path / "back.jpg"), max_long=700)
    assert [(v.name, v.phi_deg) for v in views] == [("right", 90.0), ("back", 180.0)]
    right, back = views
    assert back.rgb.shape == (150, 700, 3) and back.rgb.dtype == np.uint8 and back.kind == "nominal"
    assert not back.meta["alpha_from_file"] and np.all(back.alpha == 1.0)
    assert back.meta["file"] == "back.jpg" and max(back.meta["source_size"]) >= 700  # JPEG draft decode
    assert right.meta["alpha_from_file"] and right.alpha.dtype == np.float32
    assert right.alpha[50, 40] == 1.0 and right.alpha[5, 5] == 0.0 and tuple(right.rgb[50, 40]) == (255, 0, 0)
    with pytest.raises(InputError):
        vl.load_views([(180.0, tmp_path / "left.png")])
    with pytest.raises(InputError):
        vl.GenView("x", 180.0, np.zeros((4, 4, 3), np.uint8), np.zeros((5, 4)))
    with pytest.raises(InputError):
        vl.GenView("x", 180.0, np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4)), kind="other")


# --------------------------------------------------------------------------- V1 segmentation
class BoxSegmenter:
    id = "box"

    def __init__(self, result=None, error: Exception | None = None):
        self.result, self.error, self.calls = result, error, 0

    def segment(self, img, *, rect=None, progress=None, cancel=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        a = np.zeros(img.shape[:2], np.float32)
        a[30:150, 60:110] = 1.0
        return a


def test_segment_views_segmenter_key_fallback_and_drops():
    rgb = np.full((200, 200, 3), 128, np.uint8)
    rgb[40:160, 72:104] = (40, 70, 220)  # a blue subject on a flat grey background
    v = vl.GenView("back", 180.0, rgb, np.ones((200, 200), np.float32))
    seg = BoxSegmenter()
    (out,) = vl.segment_views([v], seg)
    assert out.meta["matte"] == "segmenter" and seg.calls == 1
    assert out.alpha[90, 85] == 1.0 and out.alpha[5, 5] == 0.0
    # a failing or empty segmenter falls back to keying out the flat background
    truth = np.zeros((200, 200), bool)
    truth[40:160, 72:104] = True
    for bad in (BoxSegmenter(error=RuntimeError("model failed")), BoxSegmenter(result=np.zeros((200, 200)))):
        (out,) = vl.segment_views([v], bad)
        assert out.meta["matte"] == "background key"
        assert rg.silhouette_iou(out.alpha >= 0.5, truth) > 0.9
    (out,) = vl.segment_views([v], None)
    assert out.meta["matte"] == "background key"
    # a view with nothing in it is dropped with a plain sentence
    empty = vl.GenView("left", 270.0, np.full((200, 200, 3), 90, np.uint8), np.ones((200, 200), np.float32))
    dropped: list[str] = []
    assert vl.segment_views([empty, v], None, dropped=dropped)[0].name == "back"
    assert len(dropped) == 1 and "left" in dropped[0] and "no subject" in dropped[0]
    # a matte that came with the picture is kept
    cut = dataclasses.replace(
        v, alpha=np.pad(np.ones((100, 40), np.float32), ((50, 50), (80, 80))), meta={"alpha_from_file": True}
    )
    (out,) = vl.segment_views([cut], seg)
    assert out.meta["matte"] == "file" and seg.calls == 1
    with pytest.raises(Cancelled):
        vl.segment_views([v], BoxSegmenter(error=Cancelled("stop")))


def test_segment_views_small_subjects():
    """A user's photo needs a subject 64 px across, not 2 % of the picture (a slim person side-on,
    3 m away, covers about that); a generated view fills its frame, so 2 % stays the rule for
    it.  "Too small" is told apart from "no subject"."""
    big = np.full((768, 1024, 3), 128, np.uint8)
    big[300:500, 500:540] = (40, 70, 220)  # 200 x 40 px: 1 % of the picture
    v = vl.GenView("right", 90.0, big, np.ones((768, 1024), np.float32), meta={"file": "right.jpg"})
    dropped: list[str] = []
    (out,) = vl.segment_views([v], None, dropped=dropped)
    assert out.meta["coverage"] < 0.02 and not dropped
    tiny = big.copy()
    tiny[:] = 128
    tiny[300:350, 500:510] = (40, 70, 220)  # 50 px tall: too small to line up
    assert vl.segment_views([dataclasses.replace(v, rgb=tiny)], None, dropped=dropped) == []
    assert len(dropped) == 1
    assert "right.jpg" in dropped[0] and "too small" in dropped[0] and "closer or crop" in dropped[0]
    assert "no subject" not in dropped[0]
    # a generated ("ortho") view keeps the design's 2 % rule
    dropped.clear()
    gen = dataclasses.replace(v, kind="ortho")
    assert vl.segment_views([gen], None, dropped=dropped) == [] and "too small" in dropped[0]


# --------------------------------------------------------------------------- V2 / V3 the back
@pytest.fixture(scope="module")
def back_only(body, canon, front, ortho) -> vl.BackPart:
    """A nominal back photo (rolled 3 deg, 10 % larger, shifted) with the bag painted magenta."""
    v = nominal(body, canon, 180.0, scale=1.1, roll=3.0, shift=(6.0, -9.0), paint={"bag": MAGENTA})
    views = vl.segment_views([v], None)
    return vl.view_based_back(front, canon, ortho, views, color_match=False)


def test_nominal_back_locks(back_only, body, canon, front, ortho):
    part = back_only
    assert part.ok and part.reason == "" and part.kind == "mould" and part.source == "images"
    b = part.diag["back"]
    assert b["iou"] > 0.9 and b["iou_locked"] >= 0.95
    assert abs(abs(b["similarity"]["roll_deg"]) - 3.0) < 1.0
    assert b["pose_frac"] < 0.02 and b["tps_max_frac"] < 0.06
    (back,) = part.views
    assert back.meta["registered"] and back.meta["used"] and back.rgb.shape == (S, S, 3)
    assert back.m == pytest.approx(ortho.m_in) and back.meta["offset_m"] == pytest.approx(-ortho.center_S[0])
    # locked: the back view's outline is the mirrored front outline
    assert rg.silhouette_iou(back.alpha >= 0.5, ortho.alpha0[:, ::-1]) >= 0.95
    assert part.hull is None and part.diag["relief"]["mode"] == "prior"  # no side views, no depth
    assert len(part.sc) == len(part.normals) and np.isfinite(part.sc.means).all()
    assert part.spacing_m > 0 and np.isfinite(part.back_z[front.matte]).all()
    json.dumps(part.diag)  # JSON-ready for debug.json
    # a big, finely sampled back photo (shrunk 2.5x, area-filtered first) locks as well
    big = nominal(body, canon, 180.0, m=0.0025, N=880, roll=-2.0)
    part2 = vl.view_based_back(front, canon, ortho, vl.segment_views([big], None), color_match=False)
    assert part2.ok and part2.diag["back"]["iou_locked"] >= 0.95


def test_back_colours_are_blue_and_the_bag_on_the_right(back_only, body, canon):
    sc = back_only.sc
    G_S = canon.to_S(sc.means)
    n_S = back_only.normals.astype(np.float64) @ canon.R
    cls = sp.classify_rgb(sc.colors, names=(*sp.SIDES, "view_bg"))
    _, truth = visible_in_truth(body, canon, sc.means, 180.0)  # the back is the prior's shape
    facing = (n_S[:, 2] > 0.8) & (truth == "back")
    assert facing.sum() > 1000
    assert np.mean(cls[facing] == "back") > 0.85
    # the magenta bag (the body's RIGHT, x < 0 in S, i.e. image left in the photo) lands on the
    # right side; mirroring the back photo would put it under the raised left arm
    mag = np.abs(sc.colors - MAGENTA / 255.0).max(axis=1) < 0.25
    assert mag.sum() > 30
    bag_x = body.to_B(sc.means[mag])[:, 0]  # body frame: x_B = the body's own left
    assert np.median(bag_x) < -0.2 and np.mean(bag_x < -0.12) > 0.9
    # and it is at the bag's height (around the hip, 0.86 - 1.14 m)
    h = -body.to_B(sc.means[mag])[:, 1]
    assert 0.8 < np.median(h) < 1.2
    assert np.all(G_S[mag, 2] > -0.1)  # behind the front surface, not on it


# --------------------------------------------------------------------------- V5 side views
@pytest.fixture(scope="module")
def with_sides(body, canon, front, prior, ortho_prior) -> vl.BackPart:
    """Back + right (90) + left (270) photos, each framed differently, and a front frame whose
    orbit-axis depth is 6 cm off."""
    views = [
        nominal(body, canon, 180.0, scale=1.1, roll=3.0),
        nominal(body, canon, 90.0, scale=0.9, shift=(20.0, 5.0)),
        nominal(body, canon, 270.0, m=0.003, N=720, shift=(-30.0, 0.0)),  # a big picture: shrunk 2.4x
    ]
    return vl.view_based_back(front, prior, ortho_prior, vl.segment_views(views, None), color_match=False)


def test_side_views_find_the_orbit_axis(with_sides, canon):
    part = with_sides
    assert part.ok, part.reason
    orbit = part.diag["orbit"]
    assert abs(orbit["z_axis"] - canon.z_axis) < 0.01  # the prior was 6 cm off
    assert abs(orbit["agree_m"]) < orbit["agree_tol_m"]
    for rec in orbit["views"]:
        assert rec["used"] and rec["phi_used"] == rec["phi"] and rec["mad_m"] < 0.01
    assert part.diag["warnings"] == []
    assert part.hull is not None and part.diag["hull"]["valid_rays"] > 0.9 * part.diag["hull"]["matte_rays"]
    assert part.diag["relief"]["mode"] == "sides"


def test_registered_views_are_reused(with_sides, front, prior, ortho_prior):
    # views registered by a previous run (or exact cameras, M4) skip registration and V5
    again = vl.view_based_back(front, prior, ortho_prior, with_sides.views, color_match=True)
    assert again.ok
    recs = again.diag["orbit"]["views"]
    assert len(recs) == 2 and all(r.get("given") for r in recs)
    for a, b in zip(again.views[1:], with_sides.views[1:], strict=True):
        assert a.meta["offset_m"] == b.meta["offset_m"] and "color_match" in a.meta
        assert np.array_equal(a.alpha, b.alpha)
    vol = with_sides.diag["hull"]["volume_m3"]
    assert again.diag["hull"]["volume_m3"] == pytest.approx(vol, rel=0.01)
    assert abs(again.diag["orbit"]["z_axis"] - with_sides.diag["orbit"]["z_axis"]) < 0.003


def test_side_colours_land_on_the_correct_sides(with_sides, body, canon):
    sc = with_sides.sc
    n_S = with_sides.normals.astype(np.float64) @ canon.R
    cls = sp.classify_rgb(sc.colors, names=(*sp.SIDES, "view_bg"))
    for phi, sign, side in ((90.0, 1.0, "left"), (270.0, -1.0, "right")):
        vis, truth = visible_in_truth(body, canon, sc.means, phi)
        sel = (sign * n_S[:, 0] > 0.9) & vis & (truth == side)
        assert sel.sum() > 300, side
        assert np.mean(cls[sel] == side) > 0.85, side
    # the back keeps its back colour
    vis, truth = visible_in_truth(body, canon, sc.means, 180.0)
    sel = (n_S[:, 2] > 0.8) & vis & (truth == "back")
    assert np.mean(cls[sel] == "back") > 0.85


def test_mislabelled_side_views_are_turned_round(body, canon, front, prior, ortho_prior):
    # the user named the pictures by the subject's left / right instead of "where you walked"
    views = [
        nominal(body, canon, 180.0),
        nominal(body, canon, 90.0, name="left", shift=(10.0, 0.0)),
        nominal(body, canon, 270.0, name="right", shift=(-10.0, 0.0)),
    ]
    views = [dataclasses.replace(views[0])] + [
        dataclasses.replace(v, phi_deg=360.0 - v.phi_deg, meta={"file": v.name + ".jpg"}) for v in views[1:]
    ]
    part = vl.view_based_back(front, prior, ortho_prior, vl.segment_views(views, None), color_match=False)
    assert part.ok
    recs = {r["name"]: r for r in part.diag["orbit"]["views"]}
    assert recs["left"]["phi_used"] == 90.0 and recs["right"]["phi_used"] == 270.0
    assert sum("other side" in w for w in part.diag["warnings"]) == 2
    assert abs(part.diag["orbit"]["z_axis"] - canon.z_axis) < 0.01


def test_swapped_names_in_perspective_photos_never_share_a_side(body, canon, front, ortho):
    """Phone photos (perspective, 40 deg lens, 2.5 m) of the right and left sides with the
    names swapped: the raised arm, magnified toward its camera, makes one picture fit its
    wrong name.  Turning only the other one put both on the same side (side colours right on
    5 % / 12 % of the Gaussians) with a note that suggested the mix-up was fixed."""

    def side(phi: float, name_phi: float) -> vl.GenView:
        b = sp.make_body(yaw_deg=-phi)
        rgb, alpha, _, _ = sp.render_photo(b, sp.make_cam(240, 320, 40.0), floor=False, wall=False)
        a = alpha[..., None]
        comp = np.clip(np.rint(rgb * a + 128 * (1 - a)), 0, 255).astype(np.uint8)
        name = vl._phi_name(name_phi)
        return vl.GenView(name, name_phi, comp, alpha.astype(np.float32), meta={"file": name + ".jpg"})

    views = [nominal(body, canon, 180.0), side(90.0, 270.0), side(270.0, 90.0)]
    part = vl.view_based_back(
        front, canon, ortho, vl.segment_views(views, None), color_match=False, fill_holes=False
    )
    assert part.ok
    used = [v.phi_deg for v in part.views[1:] if v.meta.get("used")]
    assert len(used) == len(set(used))  # never two pictures on one side
    (pair,) = part.diag["orbit"]["pairs"]
    assert pair["decision"] in ("both", "unsure", "as named")
    warns = " ".join(part.diag["warnings"])
    if pair["decision"] == "both":  # both turned round, each said so
        assert sorted(used) == [90.0, 270.0] and warns.count("other side") == 2
    else:  # neither: and the naming is not claimed to be fixed
        assert "other side" not in warns
        assert pair["decision"] == "as named" or "could not be confirmed" in warns


def _registered_sides(views, front, frame_canon, ortho_in):
    """Side views lined up (V2) in the levelled frame, plus the frame and the front points."""
    fr = vl._frame(ortho_in, frame_canon)
    Pf_S = fr.to_S(vl._front_points(front, front.D_f, front.matte))
    cut0 = (frame_canon.cut_top, frame_canon.cut_bottom)
    return [vl._register_side(v, fr, Pf_S, cut0)[0] for v in views], fr, Pf_S


def test_one_view_per_azimuth_and_left_right_on_one_side(body, canon, front, prior, ortho_prior):
    right = nominal(body, canon, 90.0)
    fat = dataclasses.replace(right, alpha=cv2.dilate(right.alpha, np.ones((9, 9), np.uint8)), name="az090")
    regs, fr, Pf_S = _registered_sides(vl.segment_views([right, fat], None), front, prior, ortho_prior)
    warnings: list[str] = []
    out, _ = vl._solve_offsets(regs, Pf_S, fr, prior, warnings)
    assert [v.meta["used"] for v in out] == [True, False]  # the better-fitting one is kept
    assert "already shows that side" in out[1].meta["reason"] and len(warnings) == 1
    # named for both sides but turned onto one: the names contradict the outlines - neither is used
    true_left = dataclasses.replace(nominal(body, canon, 270.0), name="right", phi_deg=90.0)
    az265 = dataclasses.replace(nominal(body, canon, 265.0), name="az265")
    regs, fr, Pf_S = _registered_sides(vl.segment_views([true_left, az265], None), front, prior, ortho_prior)
    warnings.clear()
    out, info = vl._solve_offsets(regs, Pf_S, fr, prior, warnings)
    if out[0].meta.get("flipped"):
        assert not any(v.meta["used"] for v in out) and info.get("same_side")
        assert any("same side" in w for w in warnings)


def crop_feet(v: vl.GenView, frac: float) -> vl.GenView:
    """The picture cut ``frac`` of the subject's height above its feet (by the picture border)."""
    rows = np.flatnonzero((v.alpha >= 0.5).any(axis=1))
    cut = int(rows[0] + (rows[-1] - rows[0]) * (1.0 - frac))
    return dataclasses.replace(v, rgb=v.rgb[:cut].copy(), alpha=v.alpha[:cut].copy())


def test_cut_off_side_photos_keep_their_scale(body, canon, front, prior, ortho_prior):
    """Side photos with the feet cut off by the picture border: the row extent alone scaled
    them 4 % / 19 % too large (20 / 30 % cut), the orbit axis moved and the back grew thick."""
    true_s = 0.006 / ortho_prior.m_in  # the levelled frame measures the front ~1 % short
    views = [
        nominal(body, canon, 180.0),
        crop_feet(nominal(body, canon, 90.0), 0.3),
        crop_feet(nominal(body, canon, 270.0), 0.2),
    ]
    part = vl.view_based_back(
        front, prior, ortho_prior, vl.segment_views(views, None), color_match=False, fill_holes=False
    )
    assert part.ok and part.diag["warnings"] == []
    for v in part.views[1:]:
        assert v.meta["used"] and v.meta["anchor"] == "top"
        assert abs(v.meta["scale"] / true_s - 1.0) < 0.03, v.meta["scale"] / true_s
    assert abs(part.diag["orbit"]["z_axis"] - canon.z_axis) < 0.01
    true_back = sp.true_back_depth(body, front.cam, 256)
    inner = front.matte & np.isfinite(true_back) & (front.dt > 3)
    assert abs(float(np.median((part.back_z - true_back)[inner]))) < 0.015
    # cut at the top AND the bottom: no height in it is known, so it is not used
    feet = vl.segment_views([views[1]], None)[0]
    top = int(np.flatnonzero((feet.alpha >= 0.5).any(axis=1))[0]) + 5  # through the head as well
    both = dataclasses.replace(feet, rgb=feet.rgb[top:], alpha=feet.alpha[top:])
    rv, info = vl._register_side(both, vl._frame(ortho_prior, prior), np.zeros((0, 3)))
    assert rv is None and "top and the bottom" in info["reason"]


def test_side_photos_line_up_with_a_photo_cut_off_at_the_shins():
    """The photo (camera 1.5 m high) misses the feet; the side photo shows them: its bottom row
    is not the photo's (the row extent alone scaled it 17 % too small)."""
    body = sp.make_body(cam_height=1.5)
    cam = sp.make_cam(240, 320)
    front = sp.synth_front_model(body, cam, grid_long=128)
    canon = sp.synth_canon(body, cam)
    assert canon.cut_bottom and not canon.cut_top
    rgb, alpha, _, _ = sp.render_photo(body, cam)
    oi = vl.levelled_ortho_input(rgb, alpha, front, canon, size=S, ref=S)
    side = vl.segment_views([nominal(body, canon, 90.0)], None)
    (rv,), _, _ = _registered_sides(side, front, canon, oi)
    assert rv.meta["anchor"] == "top" and abs(rv.meta["scale"] / (0.006 / oi.m_in) - 1.0) < 0.03


def test_inconsistent_side_views_are_dropped_or_not_filled(body, canon, front, prior, ortho_prior):
    views = vl.segment_views([nominal(body, canon, phi) for phi in (180.0, 90.0, 270.0, 135.0)], None)
    back, right, left, quarter = views

    def fat(v, px):
        return dataclasses.replace(v, alpha=cv2.dilate(v.alpha, np.ones((px, px), np.uint8)))

    # a right view far too wide: its back edge disagrees with the left view's -> both unused
    part = vl.view_based_back(front, prior, ortho_prior, [back, fat(right, 41), left], color_match=False)
    assert part.ok and part.hull is None and part.diag["relief"]["mode"] == "prior"
    assert part.diag["orbit"]["agree_m"] > part.diag["orbit"]["agree_tol_m"]
    assert any("do not agree" in w for w in part.diag["warnings"])
    assert "do not agree" in part.diag["warning"]
    # a consistent set leaves almost nothing uncovered; the few side holes are lifted onto
    # the hull entry, facing their view, inside the photo's outline and behind its front
    ok = vl.view_based_back(front, prior, ortho_prior, views, color_match=False)
    holes = {r["name"]: r for r in ok.diag["holes"]["views"]}
    assert set(holes) == {"right", "left", "back_right"}
    assert all(r["lifted"] == r["holes"] < 50 for r in holes.values())
    n_h = ok.diag["n_holes"]
    if n_h:
        P = ok.sc.means[-n_h:]
        u, v, z = sp.project_points(front.cam, P)
        cell = front.grid.cell_of(u, v)
        assert np.all(cell >= 0) and front.matte.ravel()[cell].all()
        assert np.all(z >= front.D_f.ravel()[cell] - 0.02)
    # a quarter view far too wide (no left/right check applies): its holes are not filled
    bad = vl.view_based_back(
        front, prior, ortho_prior, [back, right, left, fat(quarter, 31)], color_match=False
    )
    rec = {r["name"]: r for r in bad.diag["holes"]["views"]}["back_right"]
    assert rec["skipped"] and rec["lifted"] == 0
    assert any("does not fit" in w for w in bad.diag["warnings"])


# --------------------------------------------------------------------------- V6 hull
def _hull_length(h) -> float:
    return float(np.nansum(np.where(h.valid, h.z_out - h.z_in, 0.0)))


def test_hull_voting_tolerates_one_corrupted_view(body, canon, front, prior, ortho_prior):
    good = [nominal(body, canon, phi) for phi in (90.0, 270.0, 135.0, 315.0)]
    # the 315 picture is squeezed sideways by 30 % (a bad generation or a wrong crop)
    bad = good[-1]
    M = np.array([[0.7, 0.0, 0.3 * 180.0], [0.0, 1.0, 0.0]])
    squeezed = dataclasses.replace(
        bad,
        rgb=cv2.warpAffine(bad.rgb, M, (360, 360), borderValue=(128, 128, 128)),
        alpha=cv2.warpAffine(bad.alpha, M, (360, 360)),
    )
    # the views lined up once (V2 + V5); the hull is built from them directly
    regs, fr, Pf_S = _registered_sides([*good, squeezed], front, prior, ortho_prior)
    clean, _ = vl._solve_offsets(regs[:4], Pf_S, fr, prior, [])
    dirty, _ = vl._solve_offsets([*regs[:3], regs[4]], Pf_S, fr, prior, [])
    assert all(v.meta["used"] for v in clean + dirty)  # the squeezed view is not caught by any gate ...
    h_clean = vl.hull_intervals(front, prior, ortho_prior, clean)
    ref = _hull_length(h_clean)
    dirty_hull = vl.hull_intervals(front, prior, ortho_prior, dirty)
    assert abs(_hull_length(dirty_hull) - ref) / ref < 0.10  # ... but voting outvotes it
    strict = vl.hull_intervals(front, prior, ortho_prior, dirty, rule="all")
    assert abs(_hull_length(strict) - ref) / ref > 0.10  # carving by any view would not
    # against the true body: the hull contains it, and is not much larger than four nominal
    # views allow (their visual hull keeps the corners between the views: ~1.3 x the body's
    # depth along the photo's rays, so the design's "within 10 % of the truth" is out of
    # reach for nominal views - docs/CONTRACTS.md section 13)
    true_back = sp.true_back_depth(body, front.cam, 256)
    ok = h_clean.valid & np.isfinite(true_back) & (front.dt > 3)
    assert np.mean(true_back[ok] <= h_clean.z_out[ok] + 0.02) > 0.97
    assert np.median((h_clean.z_out - true_back)[ok]) < 0.04  # measured 2.9 cm
    ratio = np.sum((h_clean.z_out - h_clean.z_in)[ok]) / np.sum((true_back - front.D_f)[ok])
    assert 1.0 < ratio < 1.45, ratio  # measured 1.31
    with pytest.raises(InputError):
        vl.hull_intervals(front, prior, ortho_prior, dirty, rule="any")


# --------------------------------------------------------------------------- V7 relief
def _adversarial_depth(calls: list):
    """A depth model gone wrong: a deep hole in the middle of the back plus noise."""

    def fn(rgb):
        calls.append(rgb.shape)
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        d = 0.5 + 0.3 * yy / h + 0.05 * np.sin(xx / 3.0)
        d[(yy - h * 0.45) ** 2 + (xx - w * 0.5) ** 2 < (0.12 * h) ** 2] = 0.0
        return d.astype(np.float32)

    return fn


def test_relief_never_carves_more_than_40_percent(body, canon, front, ortho, prior, ortho_prior):
    calls: list = []
    depth = _adversarial_depth(calls)
    inner = front.matte & (front.dt > 4)
    D_f = front.D_f.astype(np.float64)
    # back photo only: relief detail on the shell prior
    v = nominal(body, canon, 180.0)
    part = vl.view_based_back(front, canon, ortho, [v], depth_fn=depth, color_match=False)
    assert part.ok and calls == [(S, S, 3)] and part.diag["relief"]["mode"] == "detail"
    env = np.maximum(inflation_prior(front, canon, thickness=canon.T_prior / canon.height_s)[0], D_f + 0.005)
    carve = (env - part.back_z)[inner]
    assert np.mean(carve <= 0.4 * (env - D_f)[inner] + 0.005) > 0.99
    assert carve.max() > 0.01  # the relief did act
    # with side views: relief inside the hull
    views = [nominal(body, canon, phi) for phi in (180.0, 90.0, 270.0)]
    part = vl.view_based_back(
        front, prior, ortho_prior, views, depth_fn=depth, color_match=False, fill_holes=False
    )
    assert part.ok and part.diag["relief"]["mode"] == "sides" and part.diag["relief"]["shape"] == "depth"
    h = part.hull
    ok = inner & h.valid
    carve = (h.z_out - part.back_z)[ok]
    assert np.mean(carve <= 0.4 * (h.z_out - D_f)[ok] + 0.005) > 0.99
    # "flat" never asks the depth model; a heuristic result counts as no model
    n = len(calls)
    flat = vl.view_based_back(front, canon, ortho, [v], depth_fn=depth, relief="flat", color_match=False)
    assert len(calls) == n and flat.diag["relief"]["mode"] == "prior"

    class Heuristic:
        model_id = "heuristic"
        disparity = np.zeros((S, S), np.float32)

    part = vl.view_based_back(front, canon, ortho, [v], depth_fn=lambda rgb: Heuristic(), relief="depth")
    assert part.ok and part.diag["relief"]["source"] == "flat"
    assert any("depth model" in w for w in part.diag["warnings"])
    with pytest.raises(InputError):
        vl.view_based_back(front, canon, ortho, [v], relief="sharp")


# --------------------------------------------------------------------------- rejections
@pytest.mark.parametrize("arm", [40.0, 60.0])
def test_a_different_pose_is_rejected(body, canon, front, ortho, arm):
    # the photo's raised arm is at 20 deg; 40 deg moves the hand by 19 cm
    other = sp.make_body_pose(left_arm_deg=arm)
    v = nominal(other, canon, 180.0, scale=1.05)
    part = vl.view_based_back(front, canon, ortho, [v])
    assert not part.ok and len(part.sc) == 0
    assert part.diag["back"]["iou"] >= 0.75  # the outline gate passes ...
    assert part.diag["back"]["pose_frac"] > 0.06  # ... but the outlines stay too far apart
    assert "different pose" in part.reason and "%" in part.reason
    assert part.views and part.views[0].meta["used"] is False  # still shown on the contact sheet
    json.dumps(part.diag)


def perspective_back(body_fn, *, cam_h: float, dist: float, hfov: float = 60.0, pitch: float | None = None):
    """A phone photo from behind: a perspective camera ``cam_h`` m high, ``dist`` m away, aimed
    at the body's middle (tilted down) unless ``pitch`` is given; 240 x 320 portrait."""
    if pitch is None:
        pitch = math.degrees(math.atan((cam_h - 0.85) / dist))
    b = body_fn(distance=dist, pitch_deg=pitch, yaw_deg=180.0, cam_height=cam_h)
    rgb, alpha, _, _ = sp.render_photo(b, sp.make_cam(240, 320, hfov), floor=False, wall=False)
    a = alpha[..., None]
    comp = np.clip(np.rint(rgb * a + 128 * (1 - a)), 0, 255).astype(np.uint8)
    return vl.GenView("back", 180.0, comp, alpha.astype(np.float32))


@pytest.mark.parametrize(
    "cam_h,dist,hfov,pitch",
    [(1.6, 2.0, 60.0, None), (1.7, 2.0, 40.0, 25.0)],  # tilted 21 / 25 deg down
)
def test_a_perspective_back_photo_from_eye_level_locks(body, canon, front, ortho, cam_h, dist, hfov, pitch):
    """A phone at eye level, tilted down at the person: the head is nearer than the feet, a
    keystone no similarity absorbs (before the perspective stage: 'a different pose', 9 %)."""
    v = perspective_back(sp.make_body, cam_h=cam_h, dist=dist, hfov=hfov, pitch=pitch)
    part = vl.view_based_back(
        front, canon, ortho, vl.segment_views([v], None), color_match=False, fill_holes=False
    )
    assert part.ok, part.reason
    b = part.diag["back"]
    assert b["iou_similarity"] < 0.85 < b["iou"]  # the perspective stage takes the keystone out
    assert b["pose_frac"] < 0.05 and b["iou_locked"] >= 0.95
    assert abs(b["perspective"][2][1]) > 1e-4  # a vertical keystone was fitted
    # ... while the same camera on a different pose is still rejected
    other = perspective_back(
        lambda **kw: sp.make_body_pose(left_arm_deg=40.0, **kw),
        cam_h=cam_h,
        dist=dist,
        hfov=hfov,
        pitch=pitch,
    )
    bad = vl.view_based_back(front, canon, ortho, vl.segment_views([other], None), color_match=False)
    assert not bad.ok and "different pose" in bad.reason


def test_outline_pairs_survive_two_features_level_at_the_top():
    """Each outline starts at its own topmost point; with a raised hand level with the head
    (4 mm apart) the two outlines start on different features - the pairs must not follow."""

    def figure(hand_top: int) -> np.ndarray:
        m = np.zeros((200, 200), np.uint8)
        cv2.ellipse(m, (100, 120), (30, 60), 0, 0, 360, 1, -1)  # body
        cv2.circle(m, (100, 42), 18, 1, -1)  # head, top at y = 24
        cv2.line(m, (120, 80), (165, hand_top + 6), 1, 12)  # raised arm, hand top near y = 24
        return m.astype(bool)

    dst, src = figure(20), figure(26)  # the head is the top in one, the hand in the other
    assert vl.contour_points(dst, 8)[0][0] > 130 and vl.contour_points(src, 8)[0][0] < 110
    sp_, dp = vl._contour_pairs(src, dst, 64)
    d = np.linalg.norm(sp_ - dp, axis=1)
    assert np.median(d) < 2.0 and d.max() < 8.0, (np.median(d), d.max())


def test_no_back_view_or_wrong_outline(body, canon, front, ortho):
    side = nominal(body, canon, 90.0)
    part = vl.view_based_back(front, canon, ortho, [side])
    assert not part.ok and "no picture of the back" in part.reason
    # a back photo of something else entirely (a box) does not match the outline
    rgb = np.full((300, 300, 3), 128, np.uint8)
    alpha = np.zeros((300, 300), np.float32)
    alpha[40:260, 60:240] = 1.0
    rgb[alpha > 0] = (40, 70, 220)
    part = vl.view_based_back(front, canon, ortho, [vl.GenView("back", 180.0, rgb, alpha)])
    assert not part.ok and "outline" in part.reason
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        vl.view_based_back(front, canon, ortho, [nominal(body, canon, 180.0)], cancel=tok)


def test_progress_and_colour_match(body, canon, front, ortho):
    seen: list[tuple[float, str]] = []
    v = nominal(body, canon, 180.0)
    v = dataclasses.replace(v, rgb=np.clip(v.rgb.astype(int) * 0.6, 0, 255).astype(np.uint8))  # underexposed
    part = vl.view_based_back(front, canon, ortho, [v], progress=lambda f, m: seen.append((f, m)))
    assert part.ok and part.diag["color_match"] and part.views[0].meta["color_match"] == "reinhard"
    fr = [f for f, _ in seen]
    assert fr == sorted(fr) and fr[-1] == 1.0 and any("back photo" in m for _, m in seen)
    raw = vl.view_based_back(front, canon, ortho, [v], color_match=False)
    assert "color_match" not in raw.views[0].meta

    def lightness(view):
        lab = cv2.cvtColor(view.rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
        return float(lab[..., 0][view.alpha >= 0.5].mean())

    ref = cv2.cvtColor(ortho.rgba[..., :3].astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)[..., 0][
        ortho.alpha0
    ]
    # half way (strength 0.5) from the dark photo toward the photo's rim lightness
    assert lightness(raw.views[0]) < lightness(part.views[0]) < float(ref.mean())
    # ... but a back of another colour (blue, the front is red) keeps its colour: only a
    # white-balance-sized a/b shift is applied
    shift = part.views[0].meta["color_shift"]
    assert abs(shift["a_shift"]) <= 6.0 and abs(shift["b_shift"]) <= 6.0 and shift["L_shift"] > 3.0
    n_S = part.normals.astype(np.float64) @ canon.R
    _, truth = visible_in_truth(body, canon, part.sc.means, 180.0)
    sel = (n_S[:, 2] > 0.8) & (truth == "back")
    c = part.sc.colors[sel]
    assert np.mean(c[:, 2] - np.maximum(c[:, 0], c[:, 1]) > 0.15) > 0.85  # still clearly blue


# --------------------------------------------------------------------------- contact sheet
def test_contact_sheet_layout_and_labels(ortho, with_sides, back_only):
    sheet = vl.contact_sheet(ortho, with_sides.views, tile=128)
    n = 1 + len(with_sides.views)
    assert sheet.shape == (128 * ((n + 3) // 4), 128 * min(4, n), 3) and sheet.dtype == np.uint8
    plain = vl.contact_sheet(ortho, with_sides.views, tile=128, labels=False)
    assert np.any(sheet[:30] != plain[:30]) and np.array_equal(sheet[40:], plain[40:])
    # the expected outline (green) is drawn on the input and the back view
    green = (sheet[..., 1] > 200) & (sheet[..., 0] < 100) & (sheet[..., 2] < 100)
    assert green[:, :128].any() and green[:, 128:256].any()
    six = vl.contact_sheet(ortho, back_only.views * 5, tile=64)
    assert six.shape == (128, 256, 3)
    with pytest.raises(InputError):
        vl.contact_sheet(ortho, [], tile=8)


# --------------------------------------------------------------------------- performance
@pytest.mark.slow
def test_performance_768_mould_grid():
    body = sp.make_body()
    cam = sp.make_cam(576, 768)
    front = sp.synth_front_model(body, cam, grid_long=768)
    canon = sp.synth_canon(body, cam)
    rgb, alpha, _, _ = sp.render_photo(body, cam)
    views = [nominal(body, canon, phi, m=0.003, N=720) for phi in (180.0, 90.0, 270.0)]
    t0 = time.perf_counter()
    oi = vl.levelled_ortho_input(rgb, alpha, front, canon)
    views = vl.segment_views(views, None)
    part = vl.view_based_back(front, canon, oi, views, depth_fn=_adversarial_depth([]))
    dt = time.perf_counter() - t0
    assert part.ok and front.grid.Hm == 768
    # a guard against a gross slow-down, not a benchmark: ~8 s alone, 13 s on a loaded machine,
    # and CI runners are slower still
    assert dt < 45.0, f"{dt:.1f} s"
    assert math.isfinite(part.diag["qc"]["cost"])


def test_a_lone_quarter_view_is_kept_as_named_but_a_lone_side_view_can_turn():
    """A view without its mirror partner: a left / right view whose mirror fits clearly better is
    turned round, a quarter view never is (from 45 deg off the axis the two candidates' outlines
    are too alike to overrule the name)."""
    import numpy as np

    from vrphoto.geometry import views_lift as vl

    def view(name, phi):
        return vl.GenView(name, phi, np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), "nominal")

    def hyp(phi, named, swapped):
        return [{"phi": phi, "misfit_m": named}, {"phi": 360.0 - phi, "misfit_m": swapped}]

    views = [view("back_right", 135.0), view("right", 90.0)]
    hyps = [hyp(135.0, 0.064, 0.022), hyp(90.0, 0.064, 0.022)]
    flip, pairs = vl._flip_decisions(views, hyps)
    assert flip == [False, True] and pairs == []
