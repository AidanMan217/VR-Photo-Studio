"""SHARP 360 orthographic AI views (geometry/views_lift.py, design 3.5.2 V2-V5 "mvadapter",
3.5.3, 5.2): reading a generator's camera file, the lateral / invariant-row / back-lock checks,
the colour map, the shared orbit axis and its sign safety net, QC between tries and the saved
views folder - on MV-Adapter-like sets of the analytic person (tests/geometry/synth_person.py,
``mvadapter_like_set``)."""

from __future__ import annotations

import copy
import dataclasses
import json
import time

import cv2
import numpy as np
import pytest
import synth_person as sp

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken
from vrphoto.geometry import views_lift as vl

S = 256  # levelled picture / generated view size of most tests (the app uses 512)
MAGENTA = np.array([255, 0, 255], np.uint8)
AZ = [0.0, 45.0, 90.0, 180.0, 270.0, 315.0]


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
    """The true frame: the generated views turn about the body's own axis."""
    return sp.synth_canon(body, cam, thickness=0.11)


@pytest.fixture(scope="module")
def prior(canon):
    """The frame as the pipeline knows it before the side views: the orbit-axis depth is a
    prior, here 6 cm too near."""
    zh = canon.R[:, 2]
    return dataclasses.replace(canon, z_axis=canon.z_axis - 0.06, c=canon.c - 0.06 * zh)


@pytest.fixture(scope="module")
def photo(body, cam):
    rgb, alpha, _, _ = sp.render_photo(body, cam)
    return rgb, alpha


@pytest.fixture(scope="module")
def ortho(photo, front, prior) -> vl.OrthoInput:
    return vl.levelled_ortho_input(*photo, front, prior, size=S, ref=2 * S)


def load_set(mvset: sp.MVSet, folder) -> list[vl.GenView]:
    """What the pipeline does with a worker's output: read the camera file, cut the views out."""
    return vl.segment_views(vl.views_from_cameras(mvset.write(folder)), None)


@pytest.fixture(scope="module")
def make_set(body, canon, ortho, tmp_path_factory):
    def make(**kw) -> list[vl.GenView]:
        b = kw.pop("body", body)
        return load_set(sp.mvadapter_like_set(b, canon, ortho, size=S, **kw), tmp_path_factory.mktemp("set"))

    return make


@pytest.fixture(scope="module")
def lift(front, prior, ortho):
    def run(views, **kw) -> vl.BackPart:
        kw.setdefault("source", "mvadapter")
        return vl.view_based_back(front, prior, ortho, views, **kw)

    return run


@pytest.fixture(scope="module")
def clean(make_set, lift):
    views = make_set(paint={"bag": MAGENTA})
    return views, lift(views)


def visible_in_truth(body, canon, sc_means, phi: float, m: float = 0.005, N: int = 480, tol: float = 0.02):
    """``(visible, true colour class, true part)`` of points in the true orthographic view at
    ``phi``."""
    rgb, _, depth = sp.ortho_view(body, canon, phi, m=m, N=N, ss=1)
    parts = sp.ortho_view_parts(body, canon, phi, m=m, N=N)
    u, v, d = sp.ortho_project(canon, phi, m, N, sc_means)
    iu = np.clip(np.floor(u).astype(int), 0, N - 1)
    iv = np.clip(np.floor(v).astype(int), 0, N - 1)
    vis = np.isfinite(depth[iv, iu]) & (np.abs(d - depth[iv, iu]) < tol)
    return vis, sp.classify_rgb(rgb[iv, iu]), parts[iv, iu]


def side_colours(part, body, canon) -> dict[str, tuple[int, float]]:
    """Per side (back, body-left, body-right): how many generated Gaussians face it and are
    visible in the true view, and the share of them with that side's colour (the bag, painted
    in some tests, is left out)."""
    sc = part.sc
    n_S = part.normals.astype(np.float64) @ canon.R
    cls = sp.classify_rgb(sc.colors, names=(*sp.SIDES, "view_bg"))
    out = {}
    for phi, side, facing in (
        (180.0, "back", n_S[:, 2] > 0.8),
        (90.0, "left", n_S[:, 0] > 0.9),
        (270.0, "right", -n_S[:, 0] > 0.9),
    ):
        vis, truth, parts = visible_in_truth(body, canon, sc.means, phi)
        sel = facing & vis & (truth == side) & (parts != body.part("bag"))
        out[side] = (int(sel.sum()), float(np.mean(cls[sel] == side)) if sel.any() else 0.0)
    return out


def assert_sides_coloured(part, body, canon, share: float = 0.85) -> None:
    for side, (n, frac) in side_colours(part, body, canon).items():
        assert n > 300, (side, n)
        assert frac > share, (side, frac)


def bag_on_the_right(part, body) -> None:
    """The magenta-painted bag (the body's RIGHT hip, x_B < 0) lands on that side."""
    sc = part.sc
    mag = np.abs(sc.colors - MAGENTA / 255.0).max(axis=1) < 0.25
    assert mag.sum() > 30
    bag_x = body.to_B(sc.means[mag])[:, 0]
    assert np.median(bag_x) < -0.2 and np.mean(bag_x < -0.12) > 0.9
    h = -body.to_B(sc.means[mag])[:, 1]
    assert 0.8 < np.median(h) < 1.2


def z_error(part, canon) -> float:
    return abs(part.diag["orbit"]["z_axis"] - canon.z_axis)


# --------------------------------------------------------------------------- the camera file
def test_views_from_cameras_reads_a_generator_set(body, canon, ortho, tmp_path):
    st = sp.mvadapter_like_set(body, canon, ortho, size=64, ss=1)
    path = st.write(tmp_path)
    views = vl.views_from_cameras(path)
    assert [v.phi_deg for v in views] == AZ
    assert [v.name for v in views] == ["front", "front_right", "right", "back", "left", "front_left"]
    for v in views:
        assert v.kind == "ortho" and v.m is None and v.rgb.shape == (64, 64, 3) and v.rgb.dtype == np.uint8
        assert v.meta["framing"] == "input" and v.meta["camera"]["phi_source"] == "phi_deg"
        assert v.meta["ortho_half_extent"] == 0.55 and v.meta["input_fill"] == 0.9
        assert np.all(v.alpha == 1.0) and not v.meta["alpha_from_file"]  # cut out by segment_views
    assert views[3].meta["file"] == "view_3.png"
    cams = json.loads(path.read_text(encoding="utf-8"))

    def phis(data, **kw):
        return [round(v.phi_deg, 6) for v in vl.views_from_cameras(data, **kw)]

    # without phi_deg: from c2w relative to the input's camera (the world axes are not assumed)
    no_phi = {**cams, "views": [{k: x for k, x in e.items() if k != "phi_deg"} for e in cams["views"]]}
    got = vl.views_from_cameras(no_phi, base_dir=tmp_path)
    assert [round(v.phi_deg, 6) for v in got] == AZ and got[1].meta["camera"]["phi_source"] == "c2w"
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    gl = {
        **no_phi,
        "convention": "c2w_opengl",
        "views": [{**e, "c2w": (np.array(e["c2w"]) @ flip_yz).tolist()} for e in no_phi["views"]],
    }
    assert phis(gl, base_dir=tmp_path) == AZ
    # the generator's own angle as the last resort (its sign is checked by V5)
    az_only = {**cams, "views": [{"file": e["file"], "azimuth_deg": e["azimuth_deg"]} for e in cams["views"]]}
    assert phis(az_only, base_dir=tmp_path) == AZ
    # malformed files are plain InputErrors
    bad = copy.deepcopy(cams)
    bad["views"][1]["elevation_deg"] = 20.0
    with pytest.raises(InputError, match="not level"):
        vl.views_from_cameras(bad, base_dir=tmp_path)
    with pytest.raises(InputError, match="orthographic"):
        vl.views_from_cameras({**cams, "projection": "perspective"}, base_dir=tmp_path)
    with pytest.raises(InputError, match="no views"):
        vl.views_from_cameras({**cams, "views": []}, base_dir=tmp_path)
    with pytest.raises(InputError, match="no folder"):
        vl.views_from_cameras(cams)  # relative paths in a dict need base_dir
    (tmp_path / "view_4.png").unlink()
    with pytest.raises(InputError, match="could not be read"):
        vl.views_from_cameras(path)
    (tmp_path / "broken.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(InputError, match="could not be read"):
        vl.views_from_cameras(tmp_path / "broken.json")


# --------------------------------------------------------------------------- a clean set
def test_clean_set_lifts_a_blue_back_with_green_and_yellow_sides(clean, body, canon, ortho):
    _, part = clean
    assert part.ok, part.reason
    assert part.kind == "mould" and part.source == "mvadapter" and part.diag["cameras"] == "orthographic"
    d = part.diag
    assert d["warnings"] == [] and "warning" not in d
    assert d["back"]["camera"] == "exact" and d["back"]["iou"] > 0.9 and d["back"]["iou_locked"] >= 0.95
    assert d["lateral"]["iou"] > 0.9 and not d["lateral"]["applied"]  # nothing to correct
    assert d["colour_fit"]["method"] == "identity"  # the generator kept the photo's colours
    assert [v.name for v in part.views] == ["front", "back", "front_right", "right", "left", "front_left"]
    for v in part.views:
        assert v.meta["registered"] and v.meta["used"] and v.rgb.shape == (S, S, 3) and v.m == ortho.m_in
    front_v, back_v = part.views[:2]
    assert front_v.meta["offset_m"] == pytest.approx(ortho.center_S[0])
    assert back_v.meta["offset_m"] == pytest.approx(-ortho.center_S[0])
    # the back is BLUE, the body's left GREEN and its right YELLOW (measured 0.94 - 0.96)
    assert_sides_coloured(part, body, canon, share=0.9)
    bag_on_the_right(part, body)
    assert len(part.sc) == len(part.normals) and np.isfinite(part.sc.means).all()
    json.dumps(d)


def test_clean_set_finds_the_orbit_axis_within_5_mm(clean, canon, front, body):
    _, part = clean
    orbit = part.diag["orbit"]
    assert z_error(part, canon) < 0.005  # the prior was 6 cm off; measured 2.2 mm
    assert abs(orbit["agree_m"]) < 0.01 and not orbit["sign"]["flipped"]
    assert orbit["sign"]["reversed_misfit_m"] > 3 * orbit["sign"]["named_misfit_m"]
    for v in part.views[2:]:
        assert v.meta["offset_m"] == pytest.approx(
            ortho_offset(v.phi_deg, part, canon), abs=1e-9
        )  # one shared axis: o = x0 cos(phi) + z_a sin(phi)
    assert part.diag["hull"]["views"] == 4
    assert part.diag["hull"]["valid_rays"] > 0.95 * part.diag["hull"]["matte_rays"]
    assert part.diag["relief"]["mode"] == "sides"
    true_back = sp.true_back_depth(body, front.cam, 256)
    inner = front.matte & np.isfinite(true_back) & (front.dt > 3)
    assert abs(float(np.median((part.back_z - true_back)[inner]))) < 0.015  # measured 4 mm


def ortho_offset(phi: float, part, canon) -> float:
    orbit = part.diag["orbit"]
    x0 = part.views[0].meta["offset_m"]
    p = np.radians(phi)
    return float(x0 * np.cos(p) + orbit["z_axis_S"] * np.sin(p))


# --------------------------------------------------------------------------- V2 corrections
def test_a_set_with_scale_and_offset_is_corrected(make_set, lift, body, canon):
    part = lift(make_set(scale=0.97, offset_px=(3.0, 3.0), paint={"bag": MAGENTA}))
    assert part.ok, part.reason
    lat = part.diag["lateral"]
    assert lat["iou_raw"] < 0.8 and lat["applied"] and lat["iou"] > 0.95
    # one similarity for the whole set: ~1/0.97 (the true views are ~1 % taller than the
    # levelled front, whose head top the photo does not see) and the shift undone
    assert 1.0 / 0.97 * 0.98 < lat["similarity"]["s"] < 1.0 / 0.97 * 1.005
    assert 0.0 <= lat["outline_offset_px"] < 0.6  # the keyed mattes lie outside the outline
    assert part.diag["back"]["iou"] > 0.9
    assert z_error(part, canon) < 0.005  # measured 0.2 mm
    assert all(r["used"] for r in part.diag["views"] if "used" in r)
    assert_sides_coloured(part, body, canon)
    bag_on_the_right(part, body)


@pytest.mark.parametrize("phi,k", [(270.0, 1.0 / 1.09), (270.0, 1.09), (45.0, 1.09)])
def test_one_view_with_a_9_percent_scale_error_is_dropped(make_set, lift, body, canon, phi, k):
    """9 % smaller is measured by its rows; 9 % larger fills its picture top to bottom (no row
    to anchor on) and is dropped because even its visible part is too tall."""
    part = lift(make_set(view_scale={phi: k}))
    assert part.ok, part.reason
    name = vl._phi_name(phi)
    (rec,) = [r for r in part.diag["views"] if r.get("name") == name]
    assert rec["used"] is False
    (dropped,) = [v for v in part.views if v.name == name]
    assert dropped.meta["used"] is False and "size" in dropped.meta["reason"]
    if k > 1.0 and S == 256:  # the larger view touches its picture's top and bottom: a bound only
        assert rec["anchor"] == "none" and "at least" in rec["reason"]
    assert any(name in w and "size does not match" in w for w in part.diag["warnings"])
    assert part.diag["qc"]["terms"]["dropped"] == pytest.approx(0.5)
    others = [v for v in part.views if v.name not in (name, "front", "back")]
    assert len(others) == 3 and all(v.meta["used"] for v in others)
    assert z_error(part, canon) < 0.005  # from the other side view alone when a +-90 one is gone
    assert part.diag["hull"]["views"] == 3
    assert_sides_coloured(part, body, canon)


def test_a_lone_side_view_still_carves_the_hull(clean, front, prior, ortho, body):
    """In orthographic projection the 90 and 270 views carry the same silhouette, so with
    one of them dropped the other counts twice in the hull's vote: the hull stays as tight as
    with the whole set (with one vote it loosened from 3.9 to 8.7 cm beyond the true back)."""
    _, part = clean
    true_back = sp.true_back_depth(body, front.cam, 256)
    inner = front.matte & np.isfinite(true_back) & (front.dt > 3)

    def beyond(views, rule="vote"):
        hull = vl.hull_intervals(front, prior, ortho, views, rule=rule)
        ok = inner & hull.valid
        assert ok.sum() > 0.9 * inner.sum()
        return np.asarray(hull.z_out - true_back, np.float64)[ok]

    full = beyond(part.views)
    for gone in ("left", "right"):
        lone = beyond([v for v in part.views if v.name != gone])
        assert np.median(lone) == pytest.approx(np.median(full), abs=0.002), gone
        assert np.percentile(lone, 90) < np.percentile(full, 90) + 0.01, gone
        assert (lone > -0.01).mean() > 0.99  # never carved into the body
    cv = vl._constraining([v for v in part.views if v.name != "left"], vl._frame(ortho, prior))
    assert sorted((round(c[0]), c[3]) for c in cv) == [(45, 1), (90, 2), (315, 1)]
    assert all(c[3] == 1 for c in vl._constraining(part.views, vl._frame(ortho, prior)))


# --------------------------------------------------------------------------- V5 the azimuth sign
def test_flipped_sign_cameras_are_detected_and_turned_round(make_set, lift, body, canon):
    part = lift(make_set(flip_sign=True, paint={"bag": MAGENTA}))
    assert part.ok, part.reason
    sign = part.diag["orbit"]["sign"]
    assert sign["flipped"] and sign["named_misfit_m"] > 3 * sign["reversed_misfit_m"]
    assert any("turned round" in w for w in part.diag["warnings"])
    assert "turned round" in part.diag["warning"]
    # the picture labelled 270 was rendered from 90 (and so on): it is used as 90 now
    by_file = {v.meta["file"]: v for v in part.views}
    assert by_file["view_2.png"].phi_deg == 90.0 and by_file["view_2.png"].meta["phi_named"] == 270.0
    assert by_file["view_1.png"].phi_deg == 45.0 and by_file["view_4.png"].phi_deg == 270.0
    assert by_file["view_3.png"].phi_deg == 180.0 and by_file["view_0.png"].phi_deg == 0.0
    assert z_error(part, canon) < 0.005
    assert_sides_coloured(part, body, canon)
    bag_on_the_right(part, body)
    # both tests said so: the rows, and SHARP's front rendered at the quarter views' azimuths
    assert sign["rows_say_reversed"] and sign["render"]["contradicts"]
    assert sign["render"]["reversed"] > sign["render"]["named"] + 0.3
    assert part.diag["qc"]["terms"]["sign"] == pytest.approx(0.5)  # an unflipped try wins


def test_side_views_drawn_turned_are_not_mirrored(body, canon, ortho, lift, tmp_path):
    """The generator drew the side and quarter views 18 deg past their azimuth (the cameras
    are right): their rows fit better mirrored, but SHARP's front rendered at the quarter
    views' azimuths does not confirm it, so the sides are kept (turning them round put the
    bag on the wrong side and the axis 9 cm off)."""
    az = tuple(p if p in (0.0, 180.0) else p + 18.0 for p in AZ)
    st = sp.mvadapter_like_set(body, canon, ortho, size=S, azimuths=az, paint={"bag": MAGENTA})
    for ent, phi in zip(st.cameras["views"], AZ, strict=True):  # labelled with the true azimuths
        ent.update(phi_deg=phi, azimuth_deg=phi, c2w=sp._c2w_opencv(phi, 1.8))
    part = lift(load_set(st, tmp_path), fill_holes=False)
    assert part.ok, part.reason
    sign = part.diag["orbit"]["sign"]
    assert sign["rows_say_reversed"] and not sign["flipped"] and sign["unconfirmed"]
    assert sign["render"]["ran"] and sign["render"]["named"] > sign["render"]["reversed"] + 0.2
    assert any("does not confirm" in w for w in part.diag["warnings"])
    assert not any("turned round" in w for w in part.diag["warnings"])
    assert all(not v.meta.get("flipped") for v in part.views)
    assert z_error(part, canon) < 0.06  # turned views: 4 cm (mirrored they gave 9 cm)
    assert part.diag["qc"]["terms"]["sign"] == pytest.approx(0.25)
    mag = np.abs(part.sc.colors - MAGENTA / 255.0).max(axis=1) < 0.25
    assert mag.sum() > 30 and np.median(body.to_B(part.sc.means[mag])[:, 0]) < 0  # the bag stays right


def test_a_dropped_side_view_is_turned_round_with_its_set(make_set, lift, ortho, tmp_path):
    """The picture rendered from 90 deg (labelled 270 by the reversed cameras) is 9 % too
    small and dropped; when the set is turned round it is turned round too, so the contact
    sheet, the warning and the saved views folder call it the right view (to fix and re-use)."""
    part = lift(make_set(flip_sign=True, view_scale={90.0: 1.0 / 1.09}), fill_holes=False)
    assert part.ok and part.diag["orbit"]["sign"]["flipped"]
    by_file = {v.meta["file"]: v for v in part.views}
    dropped = by_file["view_2.png"]
    assert dropped.meta["used"] is False and dropped.phi_deg == 90.0 and dropped.name == "right"
    assert dropped.meta["flipped"] and dropped.meta["phi_named"] == 270.0
    assert by_file["view_4.png"].phi_deg == 270.0 and by_file["view_4.png"].meta["used"]
    (rec,) = [r for r in part.diag["views"] if r.get("file") == "view_2.png"]
    assert rec["used"] is False and rec["phi_used"] == 90.0
    assert any(
        w.startswith("The right view (view_2.png) was not used: its size") for w in part.diag["warnings"]
    )
    assert not any("left view (view_2.png)" in w for w in part.diag["warnings"])
    folder = vl.save_views_folder(part.views, tmp_path / "views", ortho=ortho)
    doc = json.loads((folder / vl.VIEWS_JSON).read_text(encoding="utf-8"))
    saved = {r["file"]: r for r in doc["views"]}
    assert saved["right.png"]["source_file"] == "view_2.png" and saved["right.png"]["used"] is False
    assert saved["left.png"]["source_file"] == "view_4.png" and "used" not in saved["left.png"]
    # of two views at one azimuth, the one the back was built with is saved
    twin = dataclasses.replace(dropped, phi_deg=270.0, name="left")
    folder = vl.save_views_folder([twin, by_file["view_4.png"]], tmp_path / "twins", ortho=ortho)
    doc = json.loads((folder / vl.VIEWS_JSON).read_text(encoding="utf-8"))
    assert [(r["file"], r["source_file"]) for r in doc["views"]] == [("left.png", "view_4.png")]


# --------------------------------------------------------------------------- rejections
def test_an_unusable_set_is_rejected_with_a_plain_reason(make_set, lift, body, canon, ortho, tmp_path):
    st = sp.mvadapter_like_set(body, canon, ortho, size=S, ss=1)
    for name, img in st.images.items():  # something else entirely: a blue blob in every view
        blob = np.full_like(img, 128)
        cv2.ellipse(blob, (S // 2, S // 2), (S // 4, int(S * 0.43)), 0, 0, 360, (40, 70, 220), -1)
        st.images[name] = blob
    part = lift(load_set(st, tmp_path))
    assert not part.ok and len(part.sc) == 0
    assert part.diag["lateral"]["iou"] < 0.75
    assert "do not match the photo's outline" in part.reason and "75 %" in part.reason
    assert len(part.views) == 6 and all(v.meta["used"] is False for v in part.views)  # for the contact sheet
    json.dumps(part.diag)
    # the same person in another pose (the raised arm at 40 instead of 20 deg): the outline
    # gates pass, the back's pose check does not
    other = lift(make_set(body=sp.make_body_pose(left_arm_deg=40.0)))
    assert not other.ok and "different pose" in other.reason
    assert other.diag["lateral"]["iou"] >= 0.75 and other.diag["back"]["pose_frac"] > 0.06


def test_nominal_pictures_among_exact_views_are_not_used_and_cancel(clean, lift, body, canon):
    views, _ = clean
    rgb, alpha, _ = sp.ortho_view(body, canon, 90.0, m=0.008, N=300)
    photo = vl.GenView("right", 90.0, rgb, alpha, kind="nominal", meta={"file": "my_right.jpg"})
    part = lift([*views, photo], fill_holes=False)
    assert part.ok
    assert any("my_right.jpg" in w and "not made with the cameras" in w for w in part.diag["warnings"])
    assert sum(v.meta.get("file") == "my_right.jpg" and v.meta["used"] is False for v in part.views) == 1
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        lift(views, cancel=tok)


# --------------------------------------------------------------------------- V4 colour
def test_one_colour_map_corrects_the_whole_set(make_set, lift, clean):
    def cast(views):
        out = []
        for v in views:
            f = v.rgb.astype(np.float32)
            f[..., 0] *= 0.88  # a cooler white balance in every view
            f[..., 2] *= 1.06
            out.append(dataclasses.replace(v, rgb=np.clip(np.rint(f), 0, 255).astype(np.uint8)))
        return out

    _, ref = clean
    views = cast(make_set(paint={"bag": MAGENTA}))
    fitted, raw = lift(views), lift(views, color_match=False)
    cf = fitted.diag["colour_fit"]
    assert cf["method"] in ("gain", "affine") and cf["error_after"] < 0.25 * cf["error_before"]
    assert cf["gain"][0] > 1.1 and cf["gain"][2] < 0.95 and abs(cf["gain"][1] - 1.0) < 0.02

    def mean_back(part):
        b = next(v for v in part.views if v.name == "back")
        return b.rgb[b.alpha >= 0.9].astype(np.float64).mean(axis=0)

    # fitted on the (red) front, the map also restores the (blue) back: a per-channel gain,
    # not an offset that would add red to it
    assert np.abs(mean_back(fitted) - mean_back(ref)).max() < 4.0
    assert np.abs(mean_back(raw) - mean_back(ref)).max() > 8.0
    assert all(v.meta["color_match"] == cf["method"] for v in fitted.views if v.meta["used"])
    assert "color_match" not in raw.views[1].meta


# --------------------------------------------------------------------------- V11 QC between tries
def test_qc_picks_the_better_of_two_sets(clean, make_set, lift):
    _, good = clean
    worse = lift(make_set(scale=0.97, offset_px=(3.0, 3.0), view_scale={315.0: 1.0 / 1.09}))
    assert worse.ok and good.ok
    assert vl.qc_cost(good) < vl.qc_cost(worse)
    assert vl.qc_cost(good) == pytest.approx(good.diag["qc"]["cost"])
    assert vl.qc_cost(good.diag) == vl.qc_cost(good)  # the diag alone (debug.json) works too
    assert vl.choose_view_set([worse, good]) == 1
    assert vl.choose_view_set([good, worse]) == 0
    assert vl.choose_view_set([good, good]) == 0  # a tie keeps the earlier try
    failed = vl.BackPart(sc=good.sc, ok=False, reason="no", diag={"ok": False})
    assert vl.qc_cost(failed) == float("inf") and vl.qc_cost(None) == float("inf")
    assert vl.choose_view_set([failed, None, worse.diag]) == 2
    assert vl.choose_view_set([failed, None]) is None and vl.choose_view_set([]) is None


# --------------------------------------------------------------------------- the saved views folder
def test_saved_views_folder_restores_the_orthographic_cameras(clean, lift, ortho, canon, tmp_path):
    views, part = clean
    folder = vl.save_views_folder(views, tmp_path / "views", ortho=ortho)
    names = sorted(p.name for p in folder.iterdir())
    assert names == sorted(
        ["back.png", "right.png", "left.png", "front_right.png", "front_left.png", "front.png", vl.VIEWS_JSON]
    )
    doc = json.loads((folder / vl.VIEWS_JSON).read_text(encoding="utf-8"))
    assert doc["format"] == vl.VIEWS_FORMAT and doc["projection"] == "orthographic"
    assert doc["frame"]["m"] == pytest.approx(ortho.m_in) and doc["frame"]["size"] == S
    # 'images' source: pick back.png; views.json gives the azimuths and the exact cameras
    entries = vl.views_from_folder(folder / "back.png")
    assert [(phi, p.name) for phi, p in entries] == [
        (0.0, "front.png"),
        (45.0, "front_right.png"),
        (90.0, "right.png"),
        (180.0, "back.png"),
        (270.0, "left.png"),
        (315.0, "front_left.png"),
    ]
    loaded = vl.load_views(entries)
    for v in loaded:
        assert v.kind == "ortho" and v.m == pytest.approx(ortho.m_in)
        assert v.meta["frame_center_S"] == pytest.approx(list(ortho.center_S))
        assert v.meta["framing"] == "saved" and v.meta["camera"]["kind"] == "ortho"
    again = lift(vl.segment_views(loaded, None))
    assert again.ok, again.reason
    assert abs(again.diag["orbit"]["z_axis"] - part.diag["orbit"]["z_axis"]) < 0.001
    assert vl.qc_cost(again) == pytest.approx(vl.qc_cost(part), abs=0.05)
    # the registered (colour-matched) views of a run save and reload as well, matched only once
    reg = vl.save_views_folder(part.views, tmp_path / "registered", ortho=ortho)
    reloaded = vl.segment_views(vl.load_views(vl.views_from_folder(reg / "back.png")), None)
    assert all(v.kind == "ortho" and v.meta["color_match"] == "identity" for v in reloaded)
    assert lift(reloaded).diag["colour_fit"]["method"] == "matched before"
    # a picture replaced by one of another size (a photo) loses its camera; without views.json
    # the folder follows the naming rule (front.png is not a view name)
    cv2.imwrite(str(folder / "right.png"), np.full((300, 200, 3), 128, np.uint8))
    kinds = {v.name: v.kind for v in vl.load_views(vl.views_from_folder(folder / "back.png"))}
    assert kinds["right"] == "nominal" and kinds["left"] == "ortho"
    (folder / vl.VIEWS_JSON).unlink()
    plain = vl.views_from_folder(folder / "back.png")
    assert [p.name for _, p in plain] == [
        "front_right.png",
        "right.png",
        "back.png",
        "left.png",
        "front_left.png",
    ]
    assert all(v.kind == "nominal" for v in vl.load_views(plain))
    with pytest.raises(InputError):
        vl.save_views_folder([], tmp_path / "empty", ortho=ortho)


def test_a_saved_set_lines_up_in_another_levelled_frame(clean, photo, front, prior, canon, ortho, tmp_path):
    """A re-run whose levelled picture differs (another size, a slightly different centre):
    the saved cameras put every view into the new frame exactly."""
    views, _ = clean
    folder = vl.save_views_folder(views, tmp_path / "views", ortho=ortho)
    other = vl.levelled_ortho_input(*photo, front, prior, size=200, ref=400)
    loaded = vl.segment_views(vl.load_views(vl.views_from_folder(folder / "back.png")), None)
    again = vl.view_based_back(front, prior, other, loaded, source="images")
    assert again.ok, again.reason
    assert again.views[1].rgb.shape == (200, 200, 3)
    assert abs(again.diag["orbit"]["z_axis"] - canon.z_axis) < 0.006
    assert not again.diag["lateral"]["applied"]  # nothing left to correct
    # the map itself: scale about the picture centre, the saved centre onto the new one
    fr = vl._frame(other, prior)
    v = dataclasses.replace(
        loaded[3], m=2.0 * fr.m, meta={**loaded[3].meta, "frame_center_S": [fr.cx + 0.01, fr.cy + 0.02]}
    )
    M = vl._ortho_affine(v, fr)
    n = v.alpha.shape[1]
    assert M[0, 0] == pytest.approx(2.0) and M[1, 1] == pytest.approx(2.0)
    centre = M @ np.array([n / 2.0 - 0.5, n / 2.0 - 0.5, 1.0])
    # phi 180: the saved axis 1 cm to the right of the new one shows 1 cm to the LEFT
    assert centre[0] == pytest.approx(fr.S / 2.0 - 0.5 - 0.01 / fr.m)
    assert centre[1] == pytest.approx(fr.S / 2.0 - 0.5 + 0.02 / fr.m)


# --------------------------------------------------------------------------- performance
@pytest.mark.slow
def test_performance_512_set_on_a_768_mould_grid(tmp_path):
    body = sp.make_body()
    cam = sp.make_cam(576, 768)
    front = sp.synth_front_model(body, cam, grid_long=768)
    canon = sp.synth_canon(body, cam)
    rgb, alpha, _, _ = sp.render_photo(body, cam)
    st = sp.mvadapter_like_set(body, canon, vl.levelled_ortho_input(rgb, alpha, front, canon), size=512)
    path = st.write(tmp_path)
    t0 = time.perf_counter()
    oi = vl.levelled_ortho_input(rgb, alpha, front, canon)
    views = vl.segment_views(vl.views_from_cameras(path), None)
    part = vl.view_based_back(front, canon, oi, views, source="mvadapter")
    dt = time.perf_counter() - t0
    assert part.ok and front.grid.Hm == 768 and oi.size == 512
    assert z_error(part, canon) < 0.003  # measured 0.4 mm at 512 px
    # a guard against a gross slow-down: ~4.5 s alone
    assert dt < 20.0, f"{dt:.1f} s"
