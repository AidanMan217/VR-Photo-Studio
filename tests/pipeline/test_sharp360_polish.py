"""SHARP 360's optional Brush polish (docs/design/SHARP360.md 3.11, 1.2 / 1.3 and 4.5; M5).

The fake Brush (tests/addons/fakes/fake_brush.py) checks the dataset layout like Brush reads
it and "trains" by echoing ``init.ply`` with every mean moved by up to 2 mm
(``FAKE_BRUSH_ECHO_INIT``); other switches make it fail (always, only with ``--sh-degree 0``,
out of memory above a resolution, on options it does not know) or return a broken export.
So the dataset, the retry ladder, the output checks and the exact re-insertion of the
photographed front all run for real on the analytic synthetic person
(tests/geometry/synth_person.py) - no GPU, no torch.  The back is the built-in shell
carrying exact orthographic renders of the same person as its (registered) views.
"""

from __future__ import annotations

import functools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vrphoto.addons import capture
from vrphoto.core.errors import AddonNotInstalled, Cancelled
from vrphoto.core.types import CancelToken, ProgressReporter
from vrphoto.geometry.back_mould import shell_back
from vrphoto.geometry.splat import read_splats, transform_splats, write_splats
from vrphoto.geometry.splat_merge import merge_parts
from vrphoto.geometry.views_lift import GenView, levelled_ortho_input
from vrphoto.pipeline import sharp360_polish as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geometry"))
import synth_person as sp  # noqa: E402

FAKES = Path(__file__).resolve().parents[1] / "addons" / "fakes"
#: A small portrait photo of the person 2.5 m away; a 96-cell mould grid keeps it fast.
W, H, HFOV, GRID = 120, 160, 40.0, 96
VIEW_M, VIEW_N = 0.02, 100  # the generated views: 2 cm per pixel, 100 x 100
PHIS = (180.0, 90.0, 270.0, 45.0, 315.0)
TESTED = {"ok": True, "sh0": True, "masks": True, "sh_option": True}
STEPS = 400


# --------------------------------------------------------------------------- synthetic data
def _views(*, centre: bool = True) -> list[GenView]:
    """Orthographic renders of the person registered in the levelled frame of the TRUE
    canon (its origin maps to the picture centre: offset 0, y0 = 0)."""
    w = _world()
    out = []
    for phi in PHIS:
        rgb, a, _ = sp.ortho_view(w.body, w.canon, phi, m=VIEW_M, N=VIEW_N)
        meta = {"registered": True, "used": True, "offset_m": 0.0, "frame_size": VIEW_N}
        if centre:
            meta["frame_center_S"] = [0.0, 0.0]
        out.append(GenView(f"view{int(phi)}", phi, rgb, a, kind="ortho", m=VIEW_M, meta=meta))
    return out


@functools.lru_cache(maxsize=1)
def _world() -> SimpleNamespace:
    body = sp.make_body()
    cam = sp.make_cam(W, H, HFOV, grid=64)
    front = sp.synth_front_model(body, cam, grid_long=GRID)
    canon = sp.synth_canon(body, cam)
    rgb, alpha, _, _ = sp.render_photo(body, cam, ss=2)
    return SimpleNamespace(body=body, cam=cam, front=front, canon=canon, rgb=rgb, alpha=alpha)


@functools.lru_cache(maxsize=1)
def _scene() -> SimpleNamespace:
    w = _world()
    shell = shell_back(w.front, w.canon, thickness=0.11)
    part = replace(shell, views=_views(), source="mvadapter")
    merged, isf, _ = merge_parts(w.front, part, seam_deg=60.0, seed=0)
    assert len(merged) >= P.MIN_SPLATS and isf.any() and (~isf).any()
    return SimpleNamespace(**vars(w), shell=shell, part=part, merged=merged, isf=isf)


class FakeRun:
    """The bits of ``run._Run`` the polish uses."""

    def __init__(self, tmp: Path, brush_test: dict | None = None) -> None:
        self.tmp = Path(tmp)
        self.cancel = CancelToken()
        self.settings = SimpleNamespace(brush_test=dict(TESTED if brush_test is None else brush_test))
        self.outputs: list[Path] = []

    def work_dir(self, item) -> Path:
        d = self.tmp / "work"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def out_path(self, item, kind: str, ext: str) -> Path:
        d = self.tmp / "out"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"photo{kind}.{ext}"
        self.outputs.append(p)
        return p


@pytest.fixture
def fake_brush(monkeypatch, tmp_path):
    log = tmp_path / "tools.jsonl"
    monkeypatch.setenv("VRPHOTO_BRUSH", str(FAKES / "fake_brush.py"))
    monkeypatch.setenv("FAKE_TOOL_LOG", str(log))
    for k in (
        "FAKE_BRUSH_SLEEP",
        "FAKE_BRUSH_FAIL",
        "FAKE_BRUSH_PLY",
        "FAKE_BRUSH_ECHO_INIT",
        "FAKE_BRUSH_FAIL_SH0",
        "FAKE_BRUSH_OOM_ABOVE",
        "FAKE_BRUSH_REJECT_FLAGS",
        "FAKE_BRUSH_ECHO_SCALE",
        "FAKE_BRUSH_ECHO_NAN",
    ):
        monkeypatch.delenv(k, raising=False)

    def calls() -> list[list[str]]:
        lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return [c for c in map(json.loads, lines) if c[0] == "brush" and "--total-steps" in c]

    return SimpleNamespace(calls=calls)


def _arg(call: list[str], name: str) -> str | None:
    return call[call.index(name) + 1] if name in call else None


def _polish(run: FakeRun, *, part=None, photo: bool = True, quality: str = "best", steps: int = STEPS, **kw):
    s = _scene()
    return P.polish_with_brush(
        run, None, s.merged, s.isf, s.front, s.canon, s.part if part is None else part, steps=steps, seed=3,
        quality=quality, photo_rgb=s.rgb if photo else None, alpha=s.alpha if photo else None, **kw,
    )  # fmt: skip


def _read_model(model: Path) -> dict[str, tuple]:
    """COLMAP text model -> ``{name: (K, R, t, (w, h))}`` (world to camera)."""
    cams = {}
    for ln in (model / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if ln and not ln.startswith("#"):
            p = ln.split()
            assert p[1] == "PINHOLE"
            fx, fy, cx, cy = map(float, p[4:8])
            cams[int(p[0])] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), (int(p[2]), int(p[3])))
    lines = (model / "images.txt").read_text(encoding="utf-8").splitlines()
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    out = {}
    for img_line, pts_line in zip(lines[0::2], lines[1::2], strict=True):
        p = img_line.split()
        assert pts_line == "" and p[9] not in out
        K, size = cams[int(p[8])]
        out[p[9]] = (K, capture._quat_to_R(np.array(p[1:5], float)), np.array(p[5:8], float), size)
    return out


def _project(cam: tuple, P_C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K, R, t, _ = cam
    X = np.asarray(P_C, np.float64) @ R.T + t
    return K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]


def _imread(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def _rows(sc) -> np.ndarray:
    """Every row of a cloud as raw bytes (bit-exact comparison)."""
    parts = [np.ascontiguousarray(a, np.float32).reshape(len(sc), -1).view(np.uint32)
             for a in (sc.means, sc.scales, sc.quats, sc.colors, sc.opacity)]  # fmt: skip
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------- the dataset
def test_dataset_layout_round_trip(fake_brush, tmp_path, monkeypatch):
    """The written COLMAP model is consistent: unique names, every picture and mask present,
    init.ply the only .ply, and a known 3D point lands on the right pixel in every camera
    (photo copies through SHARP's camera and the crop, anchors orbiting the subject centre,
    the generated views as telephoto stand-ins of their orthographic cameras)."""
    monkeypatch.setenv("FAKE_BRUSH_ECHO_INIT", "1")
    s = _scene()
    run = FakeRun(tmp_path)
    _, _, diag = _polish(run)
    assert diag["ok"], diag
    ds = Path(diag["dataset"]["folder"])
    model = _read_model(ds / "sparse" / "0")
    names = list(model)
    files = {p.name for p in (ds / "images").iterdir()}
    assert len(set(names)) == len(names) == len(files) and set(names) == files
    n_anchor = len(P.ANCHORS) + len(P.ANCHORS_BEST)
    assert sorted(n for n in names if n.startswith("p")) == ["p0_photo.png", "p1_photo.png", "p2_photo.png"]
    assert sum(n.startswith("anchor_") for n in names) == n_anchor  # best quality: all nine
    views = [f"view_{i:02d}_az{int(phi):03d}.png" for i, phi in enumerate(PHIS)]
    assert sorted(n for n in names if n.startswith("view_")) == sorted(views)
    masks = {p.name for p in (ds / "masks").iterdir()}
    assert masks == {n for n in names if n.startswith("anchor_")} | {"view_03_az045.png", "view_04_az315.png"}
    for n in names:
        img = _imread(ds / "images" / n)
        assert (model[n][3][0], model[n][3][1]) == (img.shape[1], img.shape[0])
        if n in masks:  # masked pictures are RGB on black, their mask the same size
            assert img.shape[2] == 3 and _imread(ds / "masks" / n).shape == img.shape[:2]
        else:  # the photo and the back / side views train alpha
            assert img.shape[2] == 4
    assert [p.relative_to(ds).as_posix() for p in ds.rglob("*.ply")] == ["init.ply"]
    export = Path(_arg(fake_brush.calls()[-1], "--export-path"))
    assert not export.resolve().is_relative_to(ds.resolve())
    init = read_splats(ds / "init.ply")
    assert init.sh_rest is None and np.allclose(init.means, s.merged.means, atol=1e-6)
    pts = [ln for ln in (ds / "sparse/0/points3D.txt").read_text().splitlines() if not ln.startswith("#")]
    assert len(pts) == min(P.INIT_POINTS, len(s.merged))

    # photo copies: the front's Gaussians (on their cells' centre rays) land inside the matte
    Pf = np.asarray(s.front.sc.means, np.float64)
    for n in ("p0_photo.png", "p2_photo.png"):
        u, v = _project(model[n], Pf)
        a = _imread(ds / "images" / n)[..., 3]
        assert (a[np.floor(v).astype(int), np.floor(u).astype(int)] >= 128).mean() > 0.97
    # anchors orbit the subject centre: it stays on its pixel, the cameras move as asked
    c = np.asarray(s.canon.c, np.float64)
    uc, vc = _project(model["p0_photo.png"], c[None])
    poses = diag["frames"]["anchors"]
    x_hat, up = s.canon.R[:, 0], s.canon.up_C
    for n in (n for n in names if n.startswith("anchor_")):
        K, R, t, _ = model[n]
        ua, va = _project(model[n], c[None])
        assert abs(ua[0] - uc[0]) < 1e-6 and abs(va[0] - vc[0]) < 1e-6
        yaw, pitch = poses[int(n[7:9])]["yaw"], poses[int(n[7:9])]["pitch"]
        moved = -R.T @ t - np.zeros(3)  # the camera centre (the photo camera is at the origin)
        assert np.sign(round(float(moved @ x_hat), 6)) == np.sign(yaw)
        assert np.sign(round(float(moved @ up), 6)) == np.sign(pitch)
    # the generated views: exactly the orthographic projection, up to the 40 m perspective
    for i, phi in enumerate(PHIS):
        n = f"view_{i:02d}_az{int(phi):03d}.png"
        u, v = _project(model[n], Pf)
        uo, vo, depth = sp.ortho_project(s.canon, phi, VIEW_M, VIEW_N, Pf)
        k = P.TELEPHOTO_DIST_M / (P.TELEPHOTO_DIST_M + depth)
        np.testing.assert_allclose(u, VIEW_N / 2 + (uo - VIEW_N / 2) * k, atol=1e-6)
        np.testing.assert_allclose(v, VIEW_N / 2 + (vo - VIEW_N / 2) * k, atol=1e-6)
        # the residual perspective: |offset from the centre| x |depth| / 40 m, about 1 cm at the
        # raised hand in the side views (0.5 px here), a few mm elsewhere
        assert np.abs(u - uo).max() < 0.75 and np.abs(v - vo).max() < 0.75


def test_masks_protect_the_photographed_front():
    """A front quarter view ignores the reliable photographed front (+5 px) but trains the
    rest, black outside the subject; an anchor trains the front's interior in its colours
    and a black target well outside the silhouette.  (Finer views than the other tests: the
    5 px dilation of 2 cm pixels would cover the whole side.)"""
    s = _scene()
    m, N = 0.008, 250
    views = []
    for phi in (180.0, 45.0, 315.0):
        rgb, a, _ = sp.ortho_view(s.body, s.canon, phi, m=m, N=N)
        meta = {"registered": True, "used": True, "offset_m": 0.0, "frame_center_S": [0.0, 0.0]}
        views.append(GenView(f"v{int(phi)}", phi, rgb, a, kind="ortho", m=m, meta=meta))
    frames, info = P.build_frames(
        s.merged, s.isf, s.front, s.canon, replace(s.part, views=views), photo_rgb=s.rgb, alpha=s.alpha
    )
    by_name = {f.name: f for f in frames}
    assert [f.name for f in frames if f.kind == "view"] == ["view_00_az180.png", "view_01_az045.png",
                                                           "view_02_az315.png"]  # fmt: skip
    assert by_name["view_00_az180.png"].mask is None and by_name["view_00_az180.png"].image.shape[2] == 4
    # protected interior front Gaussians, well inside the outline
    interior = (s.front.theta_deg < 40.0) & (s.front.dt.ravel()[s.front.cell] >= 4)
    Pf = np.asarray(s.front.sc.means, np.float64)[interior]
    for n, phi in (("view_01_az045.png", 45.0), ("view_02_az315.png", 315.0)):
        f = by_name[n]
        assert f.image.shape == (N, N, 3) and f.mask.shape == (N, N)
        u, v = _project((f.K, f.R, f.t, None), Pf)
        _, _, depth = sp.ortho_project(s.canon, phi, m, N, Pf)
        _, a, d_hit = sp.ortho_view(s.body, s.canon, phi, m=m, N=N)
        iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
        seen = depth <= d_hit[iv, iu] + 0.02  # not hidden behind another body part in that view
        assert seen.mean() > 0.5
        assert (f.mask[iv[seen], iu[seen]] == 0).mean() > 0.97
        side = sp.classify_rgb(f.image[(a > 0.99) & (f.mask == 255)], ("front", "left", "right", "back"))
        assert len(side) > 200 and (side == ("left" if phi == 45.0 else "right")).mean() > 0.8  # the side
        assert (f.image[(a == 0) & (f.mask == 255)] == 0).all()  # black outside the subject
    f = by_name["anchor_00.png"]  # yaw +12
    trained = (f.mask == 255) & (f.image.sum(axis=2) > 0)
    names = sp.classify_rgb(f.image[trained], ("front", "left", "right", "back"))
    assert trained.sum() > 100 and (names == "front").mean() > 0.9  # SHARP's red front
    assert (f.mask == 255).sum() > trained.sum()  # plus the black target around it
    assert f.image[0, 0].sum() == 0 and f.mask[0, 0] == 255  # the corner is far outside the person
    assert info["anchors"][0]["used"] and info["masked"] == len(P.ANCHORS) + len(P.ANCHORS_BEST) + 2


def test_front_lookup_marks_the_reliable_photographed_surface():
    s = _scene()
    Pf = np.asarray(s.front.sc.means, np.float64)
    on, rel = P._front_lookup(s.front, Pf, 60.0)
    assert on.mean() > 0.99 and not (rel & ~on).any()
    behind = Pf * (1.0 + 0.1 / Pf[:, 2:3])  # 10 cm further along the ray
    assert not P._front_lookup(s.front, behind, 60.0)[0].any()
    rim = s.front.dt.ravel()[s.front.cell] < P._ANCHOR_RIM_CELLS
    assert not rel[rim].any() and not rel[s.front.theta_deg >= 60.0].any()
    assert rel[(s.front.theta_deg < 50.0) & ~rim].mean() > 0.99


def test_telephoto_camera_matches_the_orthographic_view():
    s = _scene()
    rng = np.random.default_rng(1)
    P_C = s.canon.c + rng.uniform(-0.5, 0.5, (500, 3))
    for phi in (0.0, 45.0, 90.0, 135.0, 180.0, 270.0):
        K, R, t = P.telephoto_camera(s.canon, phi, 0.004, (512, 512), 0.0, 0.0)
        assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) > 0
        u, v = _project((K, R, t, None), P_C)
        uo, vo, depth = sp.ortho_project(s.canon, phi, 0.004, 512, P_C)
        k = P.TELEPHOTO_DIST_M / (P.TELEPHOTO_DIST_M + depth)
        np.testing.assert_allclose(u, 256 + (uo - 256) * k, atol=1e-6)
        np.testing.assert_allclose(v, 256 + (vo - 256) * k, atol=1e-6)
        assert np.abs(u - uo).max() < 2.0  # 0.5 m off the axis both ways: ~1.6 px at 4 mm/px
    # an offset and a centre height move the picture like the levelled frame's formula
    K, R, t = P.telephoto_camera(s.canon, 90.0, 0.01, (200, 100), 0.3, -0.2)
    P_S = np.array([[0.1, -0.2, 0.3]])  # P . w = 0.3 (= offset), y = y0: the picture centre
    u, v = _project((K, R, t, None), s.canon.to_C(P_S))
    assert abs(u[0] - 100) < 1e-6 and abs(v[0] - 50) < 1e-6


def test_views_without_their_frame_centre(tmp_path):
    """Views that do not carry ``frame_center_S`` (the back-photo source) are placed in the
    levelled frame the photo gives - exactly ``levelled_ortho_input``'s centre - or the
    caller's ``frame_center``; without a photo it is estimated within a few millimetres."""
    s = _scene()
    part = replace(s.part, views=_views(centre=False))
    oi = levelled_ortho_input(s.rgb, s.alpha, s.front, s.canon, size=512)
    frames, info = P.build_frames(
        s.merged, s.isf, s.front, s.canon, part, quality="balanced", photo_rgb=s.rgb, alpha=s.alpha
    )
    assert info["frame_center"]["source"] == "from the levelled photo"
    np.testing.assert_allclose(info["frame_center"]["center_S"], oi.center_S, atol=1e-12)
    back = next(f for f in frames if f.name == "view_00_az180.png")
    K, R, t = P.telephoto_camera(s.canon, 180.0, VIEW_M, (VIEW_N, VIEW_N), 0.0, float(oi.center_S[1]))
    np.testing.assert_allclose(back.K, K)
    np.testing.assert_allclose(back.t, t, atol=1e-12)
    assert sum(f.kind == "anchor" for f in frames) == len(P.ANCHORS)  # not "best": six anchors
    frames, info = P.build_frames(s.merged, s.isf, s.front, s.canon, part, frame_center=(0.0, 0.25))
    back = next(f for f in frames if f.name == "view_00_az180.png")
    np.testing.assert_allclose(
        back.t, P.telephoto_camera(s.canon, 180.0, VIEW_M, (VIEW_N, VIEW_N), 0.0, 0.25)[2]
    )
    assert "frame_center" not in info
    frames, info = P.build_frames(s.merged, s.isf, s.front, s.canon, part)  # no photo: estimated
    assert info["frame_center"]["source"] == "estimated from the front"
    assert np.abs(np.subtract(info["frame_center"]["center_S"], oi.center_S)).max() < 0.008


# --------------------------------------------------------------------------- the polished result
def test_polished_front_rows_are_bit_identical(fake_brush, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BRUSH_ECHO_INIT", "1")
    s = _scene()
    rec: list[tuple[float, str]] = []
    sc, isf, diag = _polish(FakeRun(tmp_path), progress=ProgressReporter(lambda f, m: rec.append((f, m))))
    assert diag["ok"] and diag["reason"] == "" and "warning" not in diag
    json.dumps(diag)  # goes into debug.json
    # the photographed front: exactly the kept SHARP rows of the stitched cloud, in order
    assert np.array_equal(_rows(sc)[isf], _rows(s.merged)[s.isf])
    assert isf[: int(isf.sum())].all()  # front rows first
    # the back is Brush's (the echo moved every mean by up to 2 mm)
    back = np.asarray(sc.means)[~isf]
    assert len(back) >= P.MIN_KEPT_FRAC * (~s.isf).sum()
    assert not np.isin(_rows(sc)[~isf][:, 0], _rows(s.merged)[~s.isf][:, 0]).all()
    call = fake_brush.calls()[-1]
    assert len(fake_brush.calls()) == 1 and diag["attempts"][0]["ok"]
    assert _arg(call, "--total-steps") == str(STEPS) and _arg(call, "--seed") == "3"
    assert _arg(call, "--sh-degree") == "0" and _arg(call, "--growth-stop-iter") == str(STEPS // 2)
    assert _arg(call, "--match-alpha-weight") == "0.5" and _arg(call, "--max-resolution") == "1024"
    assert _arg(call, "--max-splats") == str(P.CAP_RANGE[0])  # 1.3 x a small cloud, clamped up
    assert _arg(call, "--export-name") == "polished.ply"
    msgs = [m for _, m in rec]
    assert f"Polishing with Brush (step {STEPS}/{STEPS})" in msgs and msgs[-1] == "Brush polish done"
    fr = [f for f, _ in rec]
    assert fr == sorted(fr) and fr[-1] == pytest.approx(1.0)


def _nearest_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Brute-force distance from each row of B to its nearest row of A."""
    out = np.empty(len(B))
    for a in range(0, len(B), 1000):
        out[a : a + 1000] = np.sqrt(((B[a : a + 1000, None, :] - A[None]) ** 2).sum(-1)).min(1)
    return out


def test_trained_copies_of_the_front_are_not_kept_twice(fake_brush, tmp_path, monkeypatch):
    """init.ply holds the front rows too, so Brush returns trained copies of them.  The exact
    rows are put back; their copies must not survive as 'back' rows in the outline band (where
    a mould back's surface rows are kept), or the outline is drawn twice."""
    monkeypatch.setenv("FAKE_BRUSH_ECHO_INIT", "1")  # training moves every mean by <= 2 mm
    s = _scene()
    sc, isf, diag = _polish(FakeRun(tmp_path))
    assert diag["ok"] and diag["front_copies"] >= 0.9 * int(s.isf.sum())
    F = np.asarray(sc.means)[isf]
    near = _nearest_dist(F, np.asarray(sc.means)[~isf]) < 0.0035
    before = _nearest_dist(np.asarray(s.merged.means)[s.isf], np.asarray(s.merged.means)[~s.isf]) < 0.0035
    assert near.sum() <= max(20, 2 * before.sum()), (near.sum(), before.sum())
    # no back row carries a front row's exact colour next to it
    front_cols = {tuple(c) for c in np.asarray(sc.colors)[isf].round(6)}
    assert sum(tuple(c) in front_cols for c in np.asarray(sc.colors)[~isf][near].round(6)) <= 2


def test_front_copies_labels_by_the_nearest_initial_row():
    rng = np.random.default_rng(0)
    F = rng.uniform(0.0, 0.2, (3000, 3))
    G = rng.uniform(0.0, 0.2, (2000, 3)) + [0.0, 0.0, 0.25]  # apart from F
    Q = np.concatenate(
        [
            F + rng.normal(0, 0.001, F.shape),
            G + rng.normal(0, 0.001, G.shape),
            rng.uniform(-0.05, 0.5, (3000, 3)),
            [[np.nan, 0.0, 0.0], [5.0, 5.0, 5.0]],
        ]
    )
    got = P.front_copies(Q, F, G)
    dF, dG = _nearest_dist(F, Q[:-2]), _nearest_dist(G, Q[:-2])
    within = np.minimum(dF, dG) <= max(P.COPY_SEARCH_M)
    assert np.array_equal(got[:-2][within], (dF < dG)[within])  # exact within the search radius
    assert not got[:-2][dF > max(P.COPY_SEARCH_M)].any()  # no front row that near: never a copy
    assert not got[-2:].any()  # non-finite or far away
    assert got[:3000].mean() > 0.95 and not got[3000:5000].any()
    assert not P.front_copies(Q, F[:0], G).any() and len(P.front_copies(Q[:0], F, G)) == 0
    # the search radius is exact: a nearest row 9 mm away is found through the 2 cm stage
    one = P.front_copies(np.array([[0.0, 0.0, 0.009]]), np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0195]]))
    assert one.tolist() == [True]


def test_sh0_failure_retries_with_sh1(fake_brush, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BRUSH_ECHO_INIT", "1")
    monkeypatch.setenv("FAKE_BRUSH_FAIL_SH0", "1")
    _, _, diag = _polish(FakeRun(tmp_path))
    assert diag["ok"], diag
    assert [_arg(c, "--sh-degree") for c in fake_brush.calls()] == ["0", "1"]
    assert [(a["sh_degree"], a["ok"]) for a in diag["attempts"]] == [(0, False), (1, True)]


def test_out_of_memory_retries_smaller(fake_brush, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BRUSH_ECHO_INIT", "1")
    monkeypatch.setenv("FAKE_BRUSH_OOM_ABOVE", "800")
    _, _, diag = _polish(FakeRun(tmp_path))
    assert diag["ok"], diag
    calls = fake_brush.calls()
    assert [_arg(c, "--max-resolution") for c in calls] == ["1024", "768"]
    assert [int(_arg(c, "--max-splats")) for c in calls] == [300_000, 180_000]
    assert [_arg(c, "--sh-degree") for c in calls] == ["0", "0"]  # memory, not the SH degree


def test_options_brush_does_not_know_are_left_out(fake_brush, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BRUSH_ECHO_INIT", "1")
    monkeypatch.setenv("FAKE_BRUSH_REJECT_FLAGS", "--growth-stop-iter,--match-alpha-weight")
    _, _, diag = _polish(FakeRun(tmp_path))
    assert diag["ok"], diag
    calls = fake_brush.calls()
    assert len(calls) == 3
    assert "--growth-stop-iter" not in calls[-1] and "--match-alpha-weight" not in calls[-1]
    assert _arg(calls[-1], "--sh-degree") == "0" and _arg(calls[-1], "--max-resolution") == "1024"


@pytest.mark.parametrize("tested", [True, False])
def test_total_failure_keeps_the_unpolished_result(fake_brush, tmp_path, monkeypatch, tested):
    monkeypatch.setenv("FAKE_BRUSH_FAIL", "1")
    s = _scene()
    run = FakeRun(tmp_path, TESTED if tested else {})
    sc, isf, diag = _polish(run)
    assert sc is s.merged and np.array_equal(isf, s.isf)
    assert not diag["ok"] and "Brush training failed" in diag["reason"]
    assert (
        diag["warning"].startswith("Brush polish did not work (")
        and "unpolished result was kept" in diag["warning"]
    )
    runs = [(a["sh_degree"], a["max_resolution"], a["masks"]) for a in diag["attempts"]]
    ladder = [(0, 1024, True), (0, 768, True), (1, 768, True)]  # out of memory first, then SH 1
    assert runs == (ladder if tested else [*ladder, (1, 768, False)])  # untested: the masks go last
    assert len(fake_brush.calls()) == len(runs)
    kept = Path(diag["log"])
    assert kept in run.outputs and "out of memory" in kept.read_text(encoding="utf-8")
    assert kept.name in diag["warning"]
    ds = Path(diag["dataset"]["folder"])
    assert (ds / "masks").is_dir() == tested
    json.dumps(diag)


@pytest.mark.parametrize(
    ("env", "why"),
    [
        ({}, "only 10 Gaussians"),  # the fake's tiny default export
        ({"FAKE_BRUSH_ECHO_INIT": "1", "FAKE_BRUSH_ECHO_SCALE": "3"}, "3.0 times the size"),
        ({"FAKE_BRUSH_ECHO_INIT": "1", "FAKE_BRUSH_ECHO_NAN": "1"}, "invalid numbers (NaN)"),
    ],
)
def test_rejected_exports_keep_the_unpolished_result(fake_brush, tmp_path, monkeypatch, env, why):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = _scene()
    run = FakeRun(tmp_path)
    sc, isf, diag = _polish(run)
    assert sc is s.merged and np.array_equal(isf, s.isf)
    assert not diag["ok"] and why in diag["reason"], diag["reason"]
    assert len(fake_brush.calls()) == 1  # a bad result is not retried
    assert Path(diag["log"]).is_file()


def test_a_polished_back_that_moved_away_is_rejected(fake_brush, tmp_path, monkeypatch):
    s = _scene()
    moved = transform_splats(s.merged, np.eye(3), np.array([3.0, 0.0, 0.0]))  # same size, 3 m aside
    ply = write_splats(moved, tmp_path / "moved.ply")
    monkeypatch.setenv("FAKE_BRUSH_PLY", str(ply))
    sc, _, diag = _polish(FakeRun(tmp_path))
    assert sc is s.merged and not diag["ok"]
    assert diag["check"]["box_growth"] == pytest.approx(1.0, abs=0.01)  # passes the box check ...
    assert "did not line up with the photo" in diag["reason"]  # ... but nothing fits behind the front


def test_check_polished(tmp_path):
    s = _scene()
    ok = write_splats(s.merged, tmp_path / "ok.ply")
    sc, info = P.check_polished(ok, s.merged)
    assert sc is not None and info == {"n": len(s.merged), "non_finite": 0, "box_growth": pytest.approx(1.0)}
    bad = replace(s.merged, opacity=np.where(np.arange(len(s.merged)) == 5, np.nan, s.merged.opacity))
    sc, info = P.check_polished(write_splats(bad, tmp_path / "bad.ply"), s.merged)
    assert sc is None and info["non_finite"] == 1 and "NaN" in info["reason"]
    (tmp_path / "junk.ply").write_bytes(b"not a ply")
    sc, info = P.check_polished(tmp_path / "junk.ply", s.merged)
    assert sc is None and "could not be read" in info["reason"]


@pytest.mark.parametrize(
    ("bt", "sh", "masks"),
    [
        ({"ok": True, "sh0": False, "masks": True, "sh_option": True}, "1", True),  # SH 0 failed in the test
        ({"ok": True, "sh0": None, "masks": False, "sh_option": False}, None, False),  # defaults only
    ],
)
def test_the_self_test_result_shapes_the_command(fake_brush, tmp_path, monkeypatch, bt, sh, masks):
    monkeypatch.setenv("FAKE_BRUSH_ECHO_INIT", "1")
    _, _, diag = _polish(FakeRun(tmp_path, bt))
    assert diag["ok"], diag
    (call,) = fake_brush.calls()
    assert _arg(call, "--sh-degree") == sh
    ds = Path(diag["dataset"]["folder"])
    assert (ds / "masks").is_dir() == masks
    names = sorted(p.name for p in (ds / "images").iterdir())
    if not masks:  # no anchors, no masked quarter views: the photo and the RGBA views only
        assert names == ["p0_photo.png", "p1_photo.png", "p2_photo.png", "view_00_az180.png",
                         "view_01_az090.png", "view_02_az270.png"]  # fmt: skip
        assert "masks" in diag["frames"]["views"][3]["reason"]


def test_not_polished_without_views_steps_or_brush(fake_brush, tmp_path, monkeypatch):
    s = _scene()
    run = FakeRun(tmp_path)
    for kw, why in (
        ({"part": s.shell}, "no views"),  # the built-in back: nothing to train against
        ({"steps": 0}, "invalid number of training steps"),
    ):
        sc, isf, diag = _polish(run, **kw)
        assert sc is s.merged and not diag["ok"] and why in diag["reason"]
    sc, _, diag = P.polish_with_brush(
        run, None, s.merged, s.isf[:-1], s.front, s.canon, s.part, steps=STEPS, seed=0, quality="best"
    )
    assert sc is s.merged and "do not match" in diag["reason"]
    assert fake_brush.calls() == []

    def missing():
        raise AddonNotInstalled("brush was not found.")

    monkeypatch.setattr(capture, "find_brush", missing)
    sc, _, diag = _polish(run)
    assert sc is s.merged and diag["reason"] == "the Capture add-on (Brush) is not installed"
    assert len(diag["attempts"]) == 1


def test_cancel_propagates(fake_brush, tmp_path, monkeypatch):
    run = FakeRun(tmp_path)
    run.cancel.cancel()
    with pytest.raises(Cancelled):
        _polish(run)
    assert fake_brush.calls() == []
    # cancelled while Brush trains: Brush is stopped and the cancel is not a "failure"
    monkeypatch.setenv("FAKE_BRUSH_ECHO_INIT", "1")
    monkeypatch.setenv("FAKE_BRUSH_SLEEP", "30")
    run = FakeRun(tmp_path / "b")

    def on_progress(frac: float, msg: str) -> None:
        if msg.startswith("Polishing with Brush (step"):
            run.cancel.cancel()

    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        _polish(run, progress=ProgressReporter(on_progress))
    assert time.monotonic() - t0 < 20 and len(fake_brush.calls()) == 1


# --------------------------------------------------------------------------- when to polish
@pytest.mark.parametrize(
    ("mode", "quality", "views", "installed", "bt", "allowed", "note"),
    [
        ("off", "best", True, True, TESTED, False, "off"),
        ("auto", "balanced", True, True, TESTED, False, "'Best'"),
        ("auto", "fast", True, True, TESTED, False, "'Best'"),
        ("auto", "best", True, True, TESTED, True, ""),
        ("auto", "best", True, True, {}, False, "not been tested"),
        ("auto", "best", True, True, {"ok": False, "error": "x"}, False, "failed its self-test"),
        ("auto", "best", False, True, TESTED, False, "made from views"),
        ("auto", "best", True, False, TESTED, False, "not installed"),
        ("on", "fast", True, True, TESTED, True, ""),
        ("on", "balanced", True, True, {}, True, "has not been tested"),
        ("on", "best", True, True, {"ok": False}, True, "failed its self-test"),
        ("on", "best", False, True, TESTED, False, "made from views"),
        ("on", "best", True, False, TESTED, False, "not installed"),
        ("sometimes", "best", True, True, TESTED, False, "unknown polish setting"),
    ],
)
def test_polish_allowed(mode, quality, views, installed, bt, allowed, note):
    s = _scene()
    part = s.part if views else s.shell
    ok, why = P.polish_allowed(
        SimpleNamespace(polish=mode),
        part,
        SimpleNamespace(brush_test=bt),
        quality=quality,
        brush_installed=installed,
    )
    assert ok is allowed and (note in why if note else why == "")
    assert P.polish_allowed(mode, part, bt, quality=quality, brush_installed=installed) == (ok, why)
    if ok and why:  # "on" without a passed self-test: a whole sentence for the user
        assert why.startswith("Brush polish was run although") and why.endswith(".")


def test_polish_allowed_details(fake_brush, monkeypatch):
    s = _scene()
    rejected = [replace(v, meta={**v.meta, "used": False}) for v in s.part.views]
    assert not P.polish_allowed("on", replace(s.part, views=rejected), TESTED, quality="best")[0]
    assert not P.polish_allowed("on", replace(s.part, ok=False), TESTED, quality="best")[0]
    assert not P.polish_allowed("on", None, TESTED, quality="best")[0]
    assert P.polish_allowed(SimpleNamespace(), s.part, TESTED, quality="best") == (True, "")  # default auto
    assert P.polish_allowed("on", s.part, TESTED, quality="best") == (True, "")  # Brush found (the fake)

    def missing():
        raise AddonNotInstalled("brush was not found.")

    monkeypatch.setattr(capture, "find_brush", missing)
    assert P.polish_allowed("on", s.part, TESTED, quality="best") == (
        False,
        "the Capture add-on (Brush) is not installed",
    )


# --------------------------------------------------------------------------- a real view-based back
def test_back_photo_views_line_up(fake_brush, tmp_path, monkeypatch):
    """The views of a real ``images`` back (``views_lift.view_based_back`` on nominal
    photos: registered into the levelled frame, no ``frame_center_S``) are placed where the
    back they built lies: its Gaussians fall inside each view's matte, the locked back's best
    at the camera the polish wrote, not a pixel above or below."""
    from vrphoto.geometry import views_lift as vl

    monkeypatch.setenv("FAKE_BRUSH_ECHO_INIT", "1")
    body = sp.make_body()
    cam = sp.make_cam(240, 320, HFOV, grid=128)
    front = sp.synth_front_model(body, cam, grid_long=128)
    canon = sp.synth_canon(body, cam)
    rgb, alpha, _, _ = sp.render_photo(body, cam, ss=2)
    ortho = vl.levelled_ortho_input(rgb, alpha, front, canon, size=256, ref=512)
    photos = [vl.GenView(f"p{int(phi)}", phi, *sp.ortho_view(body, canon, phi, m=0.006, N=360)[:2])
              for phi in (180.0, 90.0, 270.0)]  # fmt: skip
    part = vl.view_based_back(front, canon, ortho, photos, relief="flat", fill_holes=False, source="images")
    assert part.ok and all("frame_center_S" not in v.meta for v in part.views)
    merged, isf, _ = merge_parts(front, part, seam_deg=60.0, seed=0)
    s = SimpleNamespace(front=front, canon=canon)
    _, _, diag = P.polish_with_brush(
        FakeRun(tmp_path), None, merged, isf, s.front, s.canon, part, steps=STEPS, seed=1, quality="balanced",
        photo_rgb=rgb, alpha=alpha,
    )  # fmt: skip
    assert diag["ok"], diag
    np.testing.assert_allclose(diag["frames"]["frame_center"]["center_S"], ortho.center_S, atol=1e-12)
    model = _read_model(Path(diag["dataset"]["folder"]) / "sparse" / "0")
    back_pts = np.asarray(part.sc.means, np.float64)
    for i, v in enumerate(part.views):
        K, R, t, (w, h) = model[f"view_{i:02d}_az{int(v.phi_deg):03d}.png"]

        def inside(dv: float, K=K, R=R, t=t, v=v, w=w, h=h) -> float:
            Ks = K.copy()
            Ks[1, 2] += dv  # the picture shifted by dv pixels
            u, vv = _project((Ks, R, t, None), back_pts)
            iu = np.clip(np.floor(u).astype(int), 0, w - 1)
            iv = np.clip(np.floor(vv).astype(int), 0, h - 1)
            return float((v.alpha[iv, iu] > 0.5).mean())

        best = inside(0.0)
        assert best > 0.85, v.name
        if v.phi_deg == 180.0:  # the back is locked onto the photo's outline: exact
            assert best > inside(-1.0) and best > inside(1.0)
