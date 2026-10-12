"""SHARP 360 merge, clean-up and life size (geometry/splat_merge.py), on the analytic
synthetic person (tests/geometry/synth_person.py).

The "generated" back is a TRELLIS-like cloud: flat Gaussians sampled uniformly on the true
outer surface of the body (front and back), so the merge has to remove what the photo
already shows."""

from __future__ import annotations

import dataclasses
import json
import time

import cv2
import numpy as np
import pytest
import synth_person as sp

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, SplatCloud
from vrphoto.geometry import back_mould as bm
from vrphoto.geometry import splat_merge as sm
from vrphoto.geometry.splat import concat_splats, crop_splats, rotmat_to_quat, subset_splats

FIELDS = ("means", "scales", "quats", "colors", "opacity")


# --------------------------------------------------------------------------- synthetic data
def surface_samples(body: sp.Body, spacing: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(P, part)``: points uniform on the outer surface of the body (about one per
    ``spacing**2``), with the ellipsoid each lies on."""
    rng = np.random.default_rng(seed)
    pts, parts = [], []
    for k, r in enumerate(body.radii):
        p = 1.6075  # Knud Thomsen's ellipsoid area approximation
        a, b, c = r
        area = 4 * np.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)
        n = int(area / spacing**2) + 1
        m = int(n * r.max() / r.min() * 1.5) + 20
        s = rng.normal(size=(m, 3))
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        # sphere -> ellipsoid stretches the area by |s / r| (up to a constant): rejection sampling
        accept = rng.random(m) < np.linalg.norm(s / r, axis=1) * r.min()
        s = s[accept][:n]
        pts.append(body.centers[k] + (s * r) @ body.rot[k].T)
        parts.append(np.full(len(s), k))
    P, part = np.concatenate(pts), np.concatenate(parts)
    outer = np.ones(len(P), bool)
    for k in range(len(body.radii)):
        loc = np.einsum("ij,nj->ni", body._A[k], P - body.centers[k])
        outer &= ~(((loc**2).sum(1) < 1 - 1e-9) & (part != k))
    return P[outer], part[outer]


def disc_cloud(body: sp.Body, P: np.ndarray, part: np.ndarray, size: float = 0.006) -> SplatCloud:
    """Flat Gaussians on the body surface (true normals and side colours)."""
    n = body.normal_at(P, part)
    return SplatCloud(
        means=P.astype(np.float32),
        scales=np.tile([size, size, 0.1 * size], (len(P), 1)).astype(np.float32),
        quats=rotmat_to_quat(sp._tangent_frames(n)).astype(np.float32),
        colors=body.colour_of(n).astype(np.float32) / 255.0,
        opacity=np.full(len(P), 0.95, np.float32),
    )


def blobs(P: np.ndarray, *, size: float = 0.005, opacity: float = 0.9, color=(1.0, 0.0, 1.0)) -> SplatCloud:
    n = len(P)
    return SplatCloud(
        means=np.asarray(P, np.float32).reshape(n, 3),
        scales=np.full((n, 3), size, np.float32),
        quats=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        colors=np.tile(color, (n, 1)).astype(np.float32),
        opacity=np.full(n, opacity, np.float32),
    )


def index_of(ref: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Row of ``ref`` with the bit-identical position of each row of ``rows`` (-1: none)."""
    lut = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(ref, np.float32))}
    return np.array([lut.get(r.tobytes(), -1) for r in np.ascontiguousarray(rows, np.float32)], np.int64)


def min_dist(A: np.ndarray, B: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Brute-force distance from every row of ``A`` to the nearest row of ``B``."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    c = B.mean(0)  # centred: |a|^2 + |b|^2 - 2 a.b keeps its precision
    A, B = A - c, B - c
    bb = (B * B).sum(1)
    out = np.empty(len(A))
    for a in range(0, len(A), chunk):
        x = A[a : a + chunk]
        d2 = (x * x).sum(1)[:, None] + bb[None, :] - 2.0 * (x @ B.T)
        out[a : a + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def cells_of(front, P: np.ndarray) -> np.ndarray:
    """Flat mould cell of each point's projection (-1 outside the photo)."""
    u, v, _ = sp.project_points(front.cam, P)
    ok = np.isfinite(u) & (u >= 0) & (u < front.cam.width) & (v >= 0) & (v < front.cam.height)
    col = np.minimum(np.floor(np.where(ok, u, 0) / front.grid.cell_px), front.grid.Wm - 1).astype(int)
    row = np.minimum(np.floor(np.where(ok, v, 0) / front.grid.cell_px), front.grid.Hm - 1).astype(int)
    return np.where(ok, row * front.grid.Wm + col, -1)


def bits(sc: SplatCloud, name: str) -> np.ndarray:
    return np.ascontiguousarray(getattr(sc, name), np.float32).view(np.uint32)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def body() -> sp.Body:
    return sp.make_body()


@pytest.fixture(scope="module")
def cam():
    return sp.make_cam(240, 320)  # ~1 cm mould cells at 2.5 m: dense enough for 1.5 cm voxels


@pytest.fixture(scope="module")
def front(body, cam):
    return sp.synth_front_model(body, cam, 256)


@pytest.fixture(scope="module")
def canon(body, cam):
    return sp.synth_canon(body, cam)


@pytest.fixture(scope="module")
def samples(body):
    return surface_samples(body, 0.01)


@pytest.fixture(scope="module")
def gen(body, samples) -> SplatCloud:
    return disc_cloud(body, *samples)


@pytest.fixture(scope="module")
def part(gen) -> bm.BackPart:
    return bm.BackPart(sc=gen, kind="cloud", source="test")


@pytest.fixture(scope="module")
def merged(front, part):
    return sm.merge_parts(front, part, seed=7)


def _protected(front, seam: float = sm.SEAM_DEG) -> np.ndarray:
    return (front.theta_deg < seam - sm.SEAM_BAND_DEG) | front.rim


def _assert_front_rows_identical(front, out, is_front) -> np.ndarray:
    """Every front row of ``out`` is a bit-identical row of ``front.sc`` (in order); returns
    their indices."""
    idx = index_of(front.sc.means, out.means[is_front])
    assert (idx >= 0).all() and (np.diff(idx) > 0).all()
    for name in FIELDS:
        assert np.array_equal(bits(out, name)[is_front], bits(front.sc, name)[idx]), name
    return idx


# --------------------------------------------------------------------------- seam weights
def test_seam_weights_profile():
    th = np.array([0.0, 30.0, 49.9, 55.0, 60.0, 65.0, 70.1, 89.0, np.nan])
    w = sm.seam_weights(np.cos(np.radians(th)), 60.0)
    assert w[:3] == pytest.approx(1.0) and w[6:8] == pytest.approx(0.0)
    assert w[8] == 0.0
    t = (0.5 - np.cos(np.radians(70))) / (np.cos(np.radians(50)) - np.cos(np.radians(70)))
    assert w[4] == pytest.approx(t * t * (3 - 2 * t))  # smoothstep in cos(theta), not in theta
    assert np.all(np.diff(w[:8]) <= 1e-12)
    # a narrower seam moves the band
    assert sm.seam_weights(np.cos(np.radians(46.0)), 35.0) == pytest.approx(0.0)
    assert sm.seam_weights(np.cos(np.radians(24.0)), 35.0) == pytest.approx(1.0)
    for bad in (5.0, 95.0, float("nan"), "x"):
        with pytest.raises(InputError):
            sm.seam_weights(np.ones(3), bad)


# --------------------------------------------------------------------------- merge
def test_merge_drops_generated_on_the_photographed_front(body, front, gen, merged):
    out, is_front, diag = merged
    P = gen.means.astype(np.float64)
    # ground truth: the sample is what the photo shows (first hit of its own camera ray),
    # facing the camera within 45 deg, away from internal depth edges
    t, n, _, hit = body.raycast(np.zeros(3), P / P[:, 2:3])
    visible = hit & (np.abs(t - P[:, 2]) < 1e-4)
    rhat = P / np.linalg.norm(P, axis=1, keepdims=True)
    theta = np.degrees(np.arccos(np.clip(-(np.nan_to_num(n) * rhat).sum(1), -1, 1)))
    cell = cells_of(front, P)
    ok = cell >= 0
    D = np.where(np.isfinite(front.D_f), front.D_f, 1e3).astype(np.float32)
    k3 = np.ones((3, 3), np.uint8)
    edge = (cv2.dilate(D, k3) - cv2.erode(D, k3)).ravel() > 0.02  # depth jump next to the cell
    smooth = np.zeros(len(P), bool)
    smooth[ok] = ~edge[cell[ok]]
    on_front = visible & (theta < 45.0) & smooth
    assert on_front.sum() > 1000
    kept = index_of(gen.means, out.means[~is_front])
    assert (kept >= 0).all()
    assert not np.isin(np.flatnonzero(on_front), kept).any()
    # the hidden back survives
    back = ~visible & (P[:, 2] > np.nan_to_num(front.D_f.ravel()[np.maximum(cell, 0)]) + 0.05)
    assert np.isin(np.flatnonzero(back), kept).mean() > 0.99
    assert diag["gen_in"] == len(gen) and diag["gen_kept"] == int((~is_front).sum())


def test_merge_keeps_protected_front_rows_bit_identical(front, merged):
    out, is_front, diag = merged
    assert is_front.dtype == bool and len(is_front) == len(out)
    assert is_front[: diag["front_kept"]].all() and not is_front[diag["front_kept"] :].any()
    idx = _assert_front_rows_identical(front, out, is_front)
    prot = _protected(front)
    assert np.isin(np.flatnonzero(prot), idx).all()
    # the seam actually did something: some unprotected, covered front rows went
    assert diag["front_kept"] < len(front.sc)
    json.dumps(diag)  # goes to debug.json
    assert not np.isin(idx, np.flatnonzero(prot)).all()


def test_merge_never_drops_outline_rows(front, part):
    # a narrow seam leaves most of the front unprotected; the outline band stays
    out, is_front, diag = sm.merge_parts(front, part, seam_deg=35.0, seed=1)
    idx = _assert_front_rows_identical(front, out, is_front)
    assert np.isin(np.flatnonzero(front.rim), idx).all()
    assert diag["front_seam"] > 0.2 * len(front.sc)
    assert len(idx) < len(front.sc) - 200


def _seam_rows(front, sc: SplatCloud, seam: float = sm.SEAM_DEG) -> np.ndarray:
    """Generated rows on the photographed surface inside the seam band (0 < w_s < 1)."""
    P = sc.means.astype(np.float64)
    cell = cells_of(front, P)
    ok = cell >= 0
    d = np.full(len(P), np.nan)
    d[ok] = front.D_f.ravel()[cell[ok]]
    with np.errstate(invalid="ignore"):
        surface = np.abs(P[:, 2] - d) <= np.maximum(0.02, 0.01 * d)
    cos_t = np.full(len(P), np.nan)
    cos_t[ok] = front.cos_theta.ravel()[cell[ok]]
    w = sm.seam_weights(cos_t, seam)
    return surface & (w > 0) & (w < 1)


def test_merge_is_deterministic_and_the_seed_only_changes_the_seam(front, gen, part, merged):
    out, is_front, diag = merged
    again = sm.merge_parts(front, part, seed=7)
    for name in FIELDS:
        assert np.array_equal(bits(out, name), bits(again[0], name))
    assert np.array_equal(is_front, again[1]) and diag == again[2]

    other, other_front, _ = sm.merge_parts(front, part, seed=8)
    g1 = set(index_of(gen.means, out.means[~is_front]).tolist())
    g2 = set(index_of(gen.means, other.means[~other_front]).tolist())
    f1 = set(index_of(front.sc.means, out.means[is_front]).tolist())
    f2 = set(index_of(front.sc.means, other.means[other_front]).tolist())
    assert g1 != g2 and f1 != f2
    seam_gen = set(np.flatnonzero(_seam_rows(front, gen)).tolist())
    th = front.theta_deg
    seam_front = set(np.flatnonzero((th >= 50.0) & (th < 70.0) & ~front.rim).tolist())
    assert (g1 ^ g2) <= seam_gen
    assert (f1 ^ f2) <= seam_front


def test_merge_drops_generated_in_front_or_outside_the_outline(body, front, gen, samples):
    P = gen.means.astype(np.float64)
    rng = np.random.default_rng(3)
    cell = cells_of(front, P)
    band = (front.matte | front.ring).ravel()
    D_f = front.D_f.ravel()
    # (a) front intruders: photographed points pulled 6-20 cm toward the camera
    Fp = front.sc.means.astype(np.float64)[::5]
    pull = rng.uniform(0.06, 0.2, len(Fp))
    intr = Fp * ((Fp[:, 2] - pull) / Fp[:, 2])[:, None]
    # (b) beside the body: shifted sideways, kept where the centre leaves the 1 % ring
    side = P + np.array([0.6, 0.0, 0.0])
    c = cells_of(front, side)
    side = side[(c < 0) | ~band[np.maximum(c, 0)]][:300]
    # (c) hidden back Gaussians near the outline, stretched sideways by 6 cm: an end leaves the ring
    hidden = (cell >= 0) & (P[:, 2] > np.nan_to_num(D_f[np.maximum(cell, 0)], nan=np.inf) + 0.05)
    dt = front.dt.ravel()[np.maximum(cell, 0)]
    near_edge = np.flatnonzero(hidden & front.matte.ravel()[np.maximum(cell, 0)] & (dt <= 3))
    long_ = subset_splats(gen, near_edge)
    long_.scales[:] = (0.06, 0.002, 0.0005)
    long_.quats[:] = (1.0, 0.0, 0.0, 0.0)  # long axis = camera x
    ends = [long_.means + d * np.array([0.12, 0, 0], np.float32) for d in (1, -1)]
    pokes = np.zeros(len(long_), bool)
    for e in ends:
        ce = cells_of(front, e)
        pokes |= (ce < 0) | ~band[np.maximum(ce, 0)]
    long_ = subset_splats(long_, pokes)
    # (d) control: small hidden Gaussians well inside are kept
    inner = np.flatnonzero(hidden & (dt >= 5))[:300]
    ctrl = subset_splats(gen, inner)
    bad = concat_splats([blobs(intr), blobs(side), long_])
    test = concat_splats([bad, ctrl])
    assert len(intr) > 500 and len(side) > 100 and len(long_) > 30 and len(ctrl) == 300
    out, is_front, diag = sm.merge_parts(front, bm.BackPart(sc=test, source="test"), seed=0)
    kept = index_of(test.means, out.means[~is_front])
    assert not (kept < len(bad)).any()
    assert np.isin(np.arange(len(bad), len(test)), kept).all()
    assert diag["gen_in_front"] >= len(intr) and diag["gen_outside"] >= len(side) + len(long_)


def test_merge_budget(front, gen, part):
    nF = len(front.sc)
    budget = nF + 6000
    out, is_front, diag = sm.merge_parts(front, part, budget=budget, seed=0)
    assert len(out) <= budget
    assert diag["gen_decimated_to"] <= budget - nF - budget // 100
    assert not diag["front_decimated"]
    _assert_front_rows_identical(front, out, is_front)
    assert (~is_front).sum() > 3000  # the back was thinned, not dropped

    # only a front (after the seam) larger than 80 % of the budget is merged
    small = nF
    out2, is_front2, diag2 = sm.merge_parts(front, part, budget=small, seed=0)
    assert diag2["front_decimated"] and len(out2) <= small and is_front2.sum() <= 0.8 * small

    # without a back the front comes back as-is
    out3, is_front3, diag3 = sm.merge_parts(front, None)
    assert out3 is front.sc and is_front3.all() and not diag3["front_decimated"]
    failed = dataclasses.replace(part, ok=False, reason="nope")
    assert sm.merge_parts(front, failed)[0] is front.sc


def test_merge_with_the_shell_back(front, canon):
    shell = bm.shell_back(front, canon)
    out, is_front, diag = sm.merge_parts(front, shell, seed=0)
    _assert_front_rows_identical(front, out, is_front)
    G = out.means[~is_front].astype(np.float64)
    cell = cells_of(front, G)
    assert (cell >= 0).all() and (front.matte | front.ring).ravel()[cell].all()
    d = front.D_f.ravel()[cell]
    assert (G[:, 2] >= d - np.maximum(0.02, 0.01 * d) - 1e-6).all()
    assert diag["gen_kept"] > 0.7 * len(shell.sc)


def test_merge_keeps_the_shell_rim_wall(front, canon):
    """Depth-map normals make the outline band look camera-facing (cos ~0.85 instead of
    grazing).  The shell's rim wall and outermost cells there close the side and must not be
    dithered away as "on the photographed surface"; a cloud part still is."""
    band = front.matte & (front.dt <= max(2.0, 0.004 * max(front.grid.Hm, front.grid.Wm)))
    f2 = dataclasses.replace(front, cos_theta=np.where(band, np.float32(0.85), front.cos_theta))
    shell = bm.shell_back(f2, canon)
    nb, nw = shell.diag["n_back"], shell.diag["n_wall"]
    assert nw > 1000

    def kept_rows(part):
        out, is_front, diag = sm.merge_parts(f2, part, seed=0)
        _assert_front_rows_identical(f2, out, is_front)
        kept = np.zeros(len(shell.sc), bool)
        idx = index_of(shell.sc.means, out.means[~is_front])
        kept[idx[idx >= 0]] = True
        return kept, diag

    kept, diag = kept_rows(shell)
    assert diag["gen_surface_rim"] > 0 and diag["gen_surface_dropped"] <= 0.01 * diag["gen_surface"]
    assert kept[nb:].mean() > 0.6  # the rest lies in front of the photo's outline (in_front)
    assert kept[:nb].mean() > 0.85
    as_cloud, cdiag = kept_rows(dataclasses.replace(shell, kind="cloud"))
    assert cdiag["gen_surface_rim"] == 0 and as_cloud[nb:].mean() < 0.5 * kept[nb:].mean()


def test_merge_rejects_bad_input_and_cancels(front, part):
    with pytest.raises(InputError):
        sm.merge_parts(front, part, seam_deg=5.0)
    for bad in (0, -3, 2.5, True):
        with pytest.raises(InputError):
            sm.merge_parts(front, part, budget=bad)
    with pytest.raises(InputError):
        sm.merge_parts(front, part, seed="abc")
    broken = dataclasses.replace(front, rim=front.rim[:-1])
    with pytest.raises(InputError):
        sm.merge_parts(broken, part)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        sm.merge_parts(front, part, cancel=tok)


def test_progress_and_cancel_in_the_clean_up(front, canon, merged):
    sc, is_front, _ = merged
    seen: list[float] = []
    sm.merge_parts(front, None, progress=lambda f, m: seen.append(f))
    sm.remove_floaters(sc, is_front, front, canon, progress=lambda f, m: seen.append(f))
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        sm.remove_floaters(sc, is_front, front, canon, cancel=tok)
    with pytest.raises(Cancelled):
        sm.voxel_components(sc.means, 0.015, cancel=tok)


# --------------------------------------------------------------------------- under-layer
def test_underlayer_fills_holes_in_a_cloud_back(body, front, canon, gen, samples):
    P = gen.means.astype(np.float64)
    hole_c = body.centers[body.part("torso")] + 0.15 * body.R_cb[:, 2]  # behind the torso
    in_hole = np.linalg.norm(P - hole_c, axis=1) < 0.15
    cloud = bm.BackPart(sc=subset_splats(gen, ~in_hole), kind="cloud", source="test")
    sc, is_front, _ = sm.merge_parts(front, cloud, seed=0)
    out, is_front2 = sm.add_underlayer(sc, is_front, front, canon, cloud)
    n0 = len(sc)
    added = out.means[n0:].astype(np.float64)
    assert len(added) > 50 and not is_front2[n0:].any() and np.array_equal(is_front2[:n0], is_front)
    for name in FIELDS:
        assert np.array_equal(bits(out, name)[:n0], bits(sc, name))
    assert min_dist(added, sc.means).min() > sm._UNDER_GAP_M - 1e-6
    # some of it shows through the hole
    cell = cells_of(front, added)
    hole_cells = np.unique(cells_of(front, P[in_hole]))
    assert (
        np.isin(cell, hole_cells) & (added[:, 2] > np.nan_to_num(front.D_f.ravel()[cell]) + 0.05)
    ).sum() > 20
    # never behind the cloud's own back (5 mm inset), unless the front bound wins
    zmax = np.full(front.grid.Hm * front.grid.Wm, -np.inf)
    cp = cells_of(front, cloud.sc.means.astype(np.float64))
    np.maximum.at(zmax, cp[cp >= 0], cloud.sc.means[cp >= 0, 2])
    lim = np.maximum(zmax[cell] - bm.MIN_GAP_M, front.D_f.ravel()[cell] + bm.MIN_GAP_M)
    known = np.isfinite(zmax[cell])
    assert (added[known, 2] <= lim[known] + 1e-4).all()

    # mould backs are closed already
    shell = bm.shell_back(front, canon)
    same = sm.add_underlayer(sc, is_front, front, canon, shell)
    assert same[0] is sc and np.array_equal(same[1], is_front)


# --------------------------------------------------------------------------- colours
def test_feather_colors_moves_nearby_generated_colours_and_never_the_front():
    rng = np.random.default_rng(0)
    g = np.arange(0.0, 0.3, 0.005)
    X, Y = np.meshgrid(g, g)
    plane = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 2.0)], 1)
    red = blobs(plane, color=(0.9, 0.1, 0.1), opacity=0.9)
    faint = blobs(plane[::7] + [0.0, 0.0, -0.004], color=(0.0, 1.0, 0.0), opacity=0.2)
    gaps = (0.004, 0.012, 0.022, 0.05, 0.1)
    gen_parts = [
        blobs(plane[::3] + [0, 0, d] + rng.normal(0, 1e-4, (len(plane[::3]), 3)), color=(0.1, 0.1, 0.9))
        for d in gaps
    ]
    sc = concat_splats([red, faint, *gen_parts])
    nF = len(red) + len(faint)
    is_front = np.zeros(len(sc), bool)
    is_front[:nF] = True
    out = sm.feather_colors(sc, is_front)
    assert np.array_equal(bits(out, "colors")[:nF], bits(sc, "colors")[:nF])
    for name in ("means", "scales", "quats", "opacity"):
        assert np.array_equal(bits(out, name), bits(sc, name))
    moved, a = [], nF
    for gp in gen_parts:
        c = out.colors[a : a + len(gp)].astype(np.float64)
        c0 = gp.colors.astype(np.float64)
        a += len(gp)
        moved.append(np.linalg.norm(c - c0, axis=1).mean() / np.linalg.norm(np.array([0.8, -0.0, -0.8])))
        assert (c[:, 1] < 0.11).all()  # the translucent green front is not a reference
    assert moved[0] > 0.6 and moved[1] > 0.2
    assert moved[0] > moved[1] > moved[2] >= 0.0
    assert moved[3] == 0.0 and moved[4] == 0.0
    # nothing to blend: unchanged object
    assert sm.feather_colors(sc, np.ones(len(sc), bool)) is sc


# --------------------------------------------------------------------------- components
def test_voxel_components_hand_made_cases():
    v = 0.01
    chain = np.stack([np.arange(40) * 0.9 * v, np.zeros(40), np.zeros(40)], 1)
    diag = np.arange(30)[:, None] * np.array([0.9, 0.9, 0.9]) * v + [1.0, 0, 0]
    blob_a = np.random.default_rng(0).uniform(0, 3 * v, (50, 3)) + [0, 1.0, 0]
    blob_b = blob_a + [0, 5.5 * v, 0]  # a gap of more than one empty voxel
    pair = np.array([[2.0, 2.0, 2.0], [2.0 + 2.5 * v, 2.0, 2.0]])  # one empty voxel between
    # a U: the two arms only meet at the bottom, so labels need several rounds
    arm = np.arange(30) * 0.9 * v
    u_shape = np.concatenate(
        [
            np.stack([np.zeros(30), arm, np.zeros(30)], 1),
            np.stack([np.full(30, 20 * v), arm, np.zeros(30)], 1),
            np.stack([np.arange(23) * 0.9 * v, np.zeros(23), np.zeros(23)], 1),
        ]
    ) + [3.0, 3.0, 3.0]
    P = np.concatenate([chain, diag, blob_a, blob_b, pair, u_shape, [[np.nan, 0, 0]]])
    perm = np.random.default_rng(1).permutation(len(P))
    lab = sm.voxel_components(P[perm], v)[np.argsort(perm)]
    assert lab.dtype == np.int64 and lab[-1] == -1
    sizes = np.cumsum([0, len(chain), len(diag), len(blob_a), len(blob_b), 1, 1, len(u_shape)])
    groups = [lab[sizes[i] : sizes[i + 1]] for i in range(len(sizes) - 1)]
    for grp in groups:
        assert len(np.unique(grp)) == 1
    assert len({int(g[0]) for g in groups}) == len(groups)
    # labels 0..K-1 by first occurrence
    lab2 = sm.voxel_components(P, v)
    first = [int(lab2[np.flatnonzero(lab2 == k)[0]]) for k in range(lab2.max() + 1)]
    assert first == list(range(lab2.max() + 1))
    assert [lab2[i] for i in sorted(np.unique(lab2[lab2 >= 0], return_index=True)[1])] == first
    # a far outlier (rank-compressed keys) does not change the rest
    far = np.concatenate([P[:-1], [[1e12, -1e12, 1e12]]])
    lab3 = sm.voxel_components(far, v)
    assert np.array_equal(lab3[:-1], lab2[:-1]) and lab3[-1] == lab2.max() + 1
    assert len(sm.voxel_components(np.zeros((0, 3)), v)) == 0
    with pytest.raises(InputError):
        sm.voxel_components(np.zeros((4, 2)), v)
    with pytest.raises(InputError):
        sm.voxel_components(P, 0.0)


def test_voxel_components_is_fast_on_a_million_points():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(1_000_000, 3))
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    P = s * 0.8
    P[:3000] = rng.normal(0, 0.01, (3000, 3)) + [3.0, 0, 0]  # a separate blob
    t0 = time.perf_counter()
    lab = sm.voxel_components(P, 0.015)
    dt = time.perf_counter() - t0
    assert lab.max() == 1 and (lab[:3000] == lab[0]).all() and (lab[3000:] != lab[0]).all()
    assert dt < 5.0


# --------------------------------------------------------------------------- floaters
def _floaters(body, front, canon, keep_P: np.ndarray, rng) -> SplatCloud:
    """200 floaters of every kind the filter knows."""
    h = canon.height_s
    # far from the body but projecting into the outline (behind it)
    u, v = sp.grid_centers(front.grid)
    m = np.flatnonzero(front.matte.ravel())
    pick = rng.choice(m, 4000)
    ray = sp.pixel_rays(front.cam, u.ravel()[pick], v.ravel()[pick])
    back = sp.true_back_depth(body, front.cam, 256).ravel()[pick]
    cand = ray * (back + rng.uniform(0.2, 0.5, len(pick)))[:, None]
    cand = cand[min_dist(cand, keep_P) > 0.1]
    single = blobs(cand[:60])
    clusters = blobs(np.concatenate([cand[60 + i] + rng.normal(0, 0.003, (5, 3)) for i in range(8)]))
    # on the surface but wrong
    surf = keep_P[rng.choice(len(keep_P), 70, replace=False)] + rng.normal(0, 0.002, (70, 3))
    faint = blobs(surf[:30], opacity=0.02)
    huge = blobs(surf[30:50], size=0.04 * h)
    needle = blobs(surf[50:70])
    needle.scales[:] = (0.02 * h, 0.0005, 0.0005)
    # beside the body, outside the outline
    side = keep_P[rng.choice(len(keep_P), 2000)] + [0.7, 0.0, 0.0]
    c = cells_of(front, side)
    side = side[(c < 0) | ~(front.matte | front.ring).ravel()[np.maximum(c, 0)]]
    huge2 = blobs(keep_P[rng.choice(len(keep_P), 10)] + rng.normal(0, 0.002, (10, 3)), size=0.04 * h)
    fl = concat_splats([single, clusters, faint, huge, needle, huge2, blobs(side[:20])])
    assert len(fl) == 200
    return fl


def test_remove_floaters_removes_injected_and_keeps_the_body(body, front, canon, merged):
    sc, is_front, _ = merged
    fl = _floaters(body, front, canon, sc.means.astype(np.float64), np.random.default_rng(5))
    test = concat_splats([sc, fl])
    isf = np.concatenate([is_front, np.zeros(len(fl), bool)])
    out, out_front, diag = sm.remove_floaters(test, isf, front, canon)
    idx = index_of(test.means, out.means)
    assert (idx >= 0).all() and (np.diff(idx) > 0).all()
    assert np.array_equal(out_front, isf[idx])
    assert not (idx >= len(sc)).any()
    assert (idx < len(sc)).sum() > 0.995 * len(sc)
    prot = np.flatnonzero(is_front)[_protected(front)[index_of(front.sc.means, sc.means[is_front])]]
    assert np.isin(prot, idx).all()
    for rule in ("low_opacity", "too_big", "needle", "outside_matte"):
        assert diag[rule] >= 20, rule
    assert diag["isolated"] >= 60 and diag["component_rows_dropped"] >= 40  # singles / clusters
    assert diag["out"] == len(out)
    json.dumps(diag)  # goes to debug.json


def test_remove_floaters_hull(body, front, canon, merged):
    sc, is_front, _ = merged
    # a dense patch 6 cm behind the true back, inside the outline
    u, v = sp.grid_centers(front.grid)
    back = sp.true_back_depth(body, front.cam, 256)
    rr, cc = np.nonzero(front.dt >= 8)
    r0, c0 = rr[len(rr) // 2], cc[len(cc) // 2]
    sl = (slice(r0 - 2, r0 + 3), slice(c0 - 2, c0 + 3))
    patch = sp.pixel_rays(front.cam, u[sl], v[sl]).reshape(-1, 3) * (back[sl].reshape(-1, 1) + 0.06)
    test = concat_splats(
        [sc, blobs(np.repeat(patch, 2, 0) + [0, 0, 0.002] * np.tile([[0], [1]], (len(patch), 1)))]
    )
    isf = np.concatenate([is_front, np.zeros(len(test) - len(sc), bool)])
    hull = bm.HullIntervals(
        z_in=np.where(front.matte, front.D_f, np.nan), z_out=back, valid=front.matte.copy()
    )
    kept_no_hull = sm.remove_floaters(test, isf, front, canon)[0]
    # near the body, not isolated: only the hull catches the patch
    assert np.isin(np.arange(len(sc), len(test)), index_of(test.means, kept_no_hull.means)).all()
    assert len(kept_no_hull) >= len(test) - 0.005 * len(sc)
    out, _, diag = sm.remove_floaters(test, isf, front, canon, hull=hull)
    assert diag["outside_hull"] >= 50 and not (index_of(test.means, out.means) >= len(sc)).any()
    assert len(out) >= len(sc) - 0.005 * len(sc)


def test_remove_floaters_keeps_a_coarse_back_of_a_distant_subject():
    """'fast' quality builds the shell every 2 mould cells; 4 m away they are ~2.4 cm apart,
    coarser than the 1.5 cm floater voxels.  Sized by the back's spacing, the filter must not
    take the back apart into "isolated" Gaussians and small components (it removed ~60 % of
    it with fixed voxels, so the photo's reverse side showed from behind)."""
    body = sp.make_body(distance=4.0)
    cam = sp.make_cam(240, 320)
    front = sp.synth_front_model(body, cam, 320)
    canon = sp.synth_canon(body, cam)
    shell = bm.shell_back(front, canon, stride=2)
    z_back = np.nanmedian(np.where(front.matte, shell.back_z, np.nan))
    assert shell.spacing_m == pytest.approx(2 * front.grid.cell_px * z_back / cam.f, rel=1e-6)
    assert shell.spacing_m > 0.02 and shell.diag["spacing_m"] == shell.spacing_m
    merged, isf, _ = sm.merge_parts(front, shell, seed=0)
    merged = sm.feather_colors(merged, isf)
    out, out_front, diag = sm.remove_floaters(merged, isf, front, canon, spacing_m=shell.spacing_m)
    assert diag["voxel_m"] == pytest.approx(1.5 * shell.spacing_m)
    n_gen = int((~isf).sum())
    assert n_gen > 500 and diag["gen_dropped"] <= 0.01 * n_gen
    # seen along the photo's rays from behind, the back still covers what it covered
    before = np.unique(cells_of(front, merged.means[~isf].astype(np.float64)))
    after = np.unique(cells_of(front, out.means[~out_front].astype(np.float64)))
    assert np.isin(before, after).mean() > 0.99
    # a fine back (or none known) keeps the 1.5 cm voxels
    assert sm.remove_floaters(merged, isf, front, canon, spacing_m=0.004)[2]["voxel_m"] == 0.015
    with pytest.raises(InputError):
        sm.remove_floaters(merged, isf, front, canon, spacing_m=-1.0)


def test_remove_floaters_never_removes_protected_front_rows(front, canon):
    # front rows far in front of the body form their own tiny component
    far = front.sc.means[::200].astype(np.float64) * 0.8
    n_far = len(far)
    extra = blobs(far, opacity=0.95)
    aug_sc = concat_splats([front.sc, extra, extra])
    aug_sc.means[len(front.sc) + n_far :] += np.float32(0.001)  # a distinct, unprotected copy
    k = len(aug_sc) - len(front.sc)
    aug = dataclasses.replace(
        front,
        sc=aug_sc,
        cell=np.concatenate([front.cell, np.full(k, -1)]),
        theta_deg=np.concatenate([front.theta_deg, np.full(k, 80.0, np.float32)]),
        rim=np.concatenate([front.rim, np.ones(n_far, bool), np.zeros(n_far, bool)]),
    )
    out, out_front, diag = sm.remove_floaters(aug_sc, np.ones(len(aug_sc), bool), aug, canon)
    idx = index_of(aug_sc.means, out.means)
    assert np.isin(np.arange(len(front.sc), len(front.sc) + n_far), idx).all()
    assert not np.isin(np.arange(len(front.sc) + n_far, len(aug_sc)), idx).any()
    assert np.isin(np.flatnonzero(_protected(front)), idx).all() and out_front.all()
    # a front row that is no longer bit-identical is protected by where it projects
    moved = dataclasses.replace(front.sc, means=np.nextafter(front.sc.means, np.float32(np.inf)))
    maps = sm._maps(front)
    by_proj = sm._protected(moved, np.ones(len(moved), bool), front, maps, sm.SEAM_DEG)
    assert np.array_equal(by_proj, _protected(front))


def test_remove_floaters_rejects_bad_mask(front, canon, merged):
    sc, is_front, _ = merged
    with pytest.raises(InputError):
        sm.remove_floaters(sc, is_front[:-1], front, canon)
    with pytest.raises(InputError):
        sm.remove_floaters(
            sc,
            is_front,
            front,
            canon,
            hull=bm.HullIntervals(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2), bool)),
        )


# --------------------------------------------------------------------------- life size
@pytest.fixture(scope="module", params=[2, 0, 5], ids=lambda s: f"seed{s}")
def upright(body, request) -> tuple[SplatCloud, float]:
    """The whole body as an upright glTF-frame cloud (y up, feet at 0) and its true height,
    for a few sampling seeds (the P99.8 height reads 0.35-0.5 % low depending on the draw)."""
    P, part = surface_samples(body, 0.007, seed=request.param)
    B = body.to_B(P)
    g = np.stack([B[:, 0], -B[:, 1], -B[:, 2]], 1)  # x left->x, y down->up, z back->front (det +1)
    g[:, 1] -= g[:, 1].min()
    sc = disc_cloud(body, P, part)
    sc.means[:] = g.astype(np.float32)
    return sc, float(g[:, 1].max())


@pytest.mark.parametrize("n_floaters", [0, 50])
def test_to_life_size_height(upright, n_floaters):
    sc, true_h = upright
    assert true_h == pytest.approx(sp.BODY_HEIGHT, abs=2e-3) and len(sc) > 30_000
    if n_floaters:
        rng = np.random.default_rng(4)
        fl = blobs(rng.uniform([-1.5, 0.0, -1.5], [1.5, 2.6, 1.5], (n_floaters, 3)))
        fl.means[: n_floaters // 2, 1] = rng.uniform(1.8, 2.6, n_floaters // 2)  # above the head
        sc = concat_splats([sc, fl])
    out, h, box, used = sm.to_life_size(sc, mode="height", height_m=1.80)
    assert used == "height" and h == pytest.approx(1.80)
    scale = float(np.linalg.norm(out.means.astype(np.float64)) / np.linalg.norm(sc.means.astype(np.float64)))
    # the real top of the head: P99.8 never reads above it, and less than 0.8 % below
    assert 0.0 <= true_h * scale / 1.80 - 1.0 < 0.008
    assert np.allclose(out.means, sc.means * scale, rtol=1e-5, atol=1e-6)
    assert np.allclose(out.scales, sc.scales * scale, rtol=1e-5)
    assert np.array_equal(out.quats, sc.quats) and np.array_equal(out.opacity, sc.opacity)
    assert box.shape == (2, 3) and box[0, 1] == pytest.approx(-0.02) and box[1, 1] == pytest.approx(1.83)
    kept = crop_splats(out, radius_pct=None, aabb=box)
    assert len(kept) >= 0.99 * (len(out) - n_floaters)


def test_to_life_size_photo_and_auto(upright):
    sc, true_h = upright
    out, h, box, used = sm.to_life_size(sc, mode="photo", height_m=0.0)
    assert used == "photo" and np.array_equal(out.means, sc.means)
    assert -0.008 < (h - true_h) / true_h <= 0.0
    cut = sm.to_life_size(sc, mode="auto", height_m=1.80, cut=True)
    assert cut[3] == "photo" and cut[1] == pytest.approx(h) and np.array_equal(cut[2], box)
    whole = sm.to_life_size(sc, mode="auto", height_m=1.80, cut=False)
    assert whole[3] == "height" and whole[1] == pytest.approx(1.80)
    x = sc.means[:, 0].astype(np.float64)
    assert box[0, 0] == pytest.approx(np.percentile(x, 0.2) - 0.03)
    assert box[1, 0] == pytest.approx(np.percentile(x, 99.8) + 0.03)


def test_to_life_size_rejects_bad_input(upright):
    sc, _ = upright
    with pytest.raises(InputError):
        sm.to_life_size(sc, mode="metric", height_m=1.8)
    for bad in (0.0, -1.0, float("nan"), None):
        with pytest.raises(InputError):
            sm.to_life_size(sc, mode="height", height_m=bad)
    with pytest.raises(InputError):
        sm.to_life_size(subset_splats(sc, []), mode="photo", height_m=1.8)
    flat = dataclasses.replace(sc, means=sc.means * np.array([1, 0, 1], np.float32))
    with pytest.raises(InputError):
        sm.to_life_size(flat, mode="height", height_m=1.8)
