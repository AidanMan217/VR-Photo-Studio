"""Tests for geometry.splat: 3DGS PLY / .splat I/O, decimation (algorithm J), cropping,
orientation and mesh sampling."""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, Mesh, SplatCloud
from vrphoto.geometry import splat as S
from vrphoto.geometry.splat import (
    C0,
    crop_splats,
    decimate_splats,
    normalize_orientation,
    read_splats,
    splats_from_mesh,
    write_splats,
)


def random_cloud(n: int, sh: int = 0, seed: int = 0, spread: float = 1.0) -> SplatCloud:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return SplatCloud(
        means=(rng.normal(size=(n, 3)) * spread).astype(np.float32),
        scales=np.exp(rng.normal(-4.0, 0.5, size=(n, 3))).astype(np.float32),
        quats=q.astype(np.float32),
        colors=rng.random((n, 3)).astype(np.float32),
        opacity=rng.uniform(0.05, 1.0, n).astype(np.float32),
        sh_rest=rng.normal(scale=0.2, size=(n, sh, 3)).astype(np.float32) if sh else None,
    )


def _quat_close(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(np.abs(a - b).max(1), np.abs(a + b).max(1))


def _ply_header(path) -> list[str]:
    data = path.read_bytes()
    end = data.index(b"end_header\n")
    return data[:end].decode("ascii").splitlines()


# --------------------------------------------------------------------------- I/O
@pytest.mark.parametrize("sh", [0, 3, 15])
def test_ply_round_trip_is_float32_exact(tmp_path, sh):
    sc = random_cloud(500, sh=sh)
    p = write_splats(sc, tmp_path / "a.ply")
    r = read_splats(p)
    assert len(r) == 500
    np.testing.assert_array_equal(r.means, sc.means)
    np.testing.assert_allclose(r.scales, sc.scales, rtol=1e-6)
    assert _quat_close(r.quats, sc.quats).max() < 1e-6
    np.testing.assert_allclose(r.colors, sc.colors, atol=1e-6)
    np.testing.assert_allclose(r.opacity, sc.opacity, atol=1e-6)
    if sh:
        np.testing.assert_allclose(r.sh_rest, sc.sh_rest, atol=1e-7)
    else:
        assert r.sh_rest is None


def test_ply_uses_standard_3dgs_property_names(tmp_path):
    sc = random_cloud(10, sh=15)
    hdr = _ply_header(write_splats(sc, tmp_path / "s.ply"))
    assert hdr[0] == "ply" and hdr[1] == "format binary_little_endian 1.0"
    assert "element vertex 10" in hdr
    props = [ln.split()[-1] for ln in hdr if ln.startswith("property")]
    assert all(ln.startswith("property float ") for ln in hdr if ln.startswith("property"))
    expected = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    expected += [f"f_rest_{i}" for i in range(45)]
    expected += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    assert props == expected


def test_ply_encoding_formulas(tmp_path):
    sc = SplatCloud(
        means=np.zeros((2, 3), np.float32),
        scales=np.array([[0.01, 0.02, 0.03], [1, 1, 1]], np.float32),
        quats=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32),
        colors=np.array([[0.5, 0.5, 0.5], [0.5 + C0, 0.5, 0.5]], np.float32),
        opacity=np.array([0.5, 0.880797], np.float32),
        sh_rest=np.stack([np.arange(9).reshape(3, 3), np.zeros((3, 3))]).astype(np.float32),
    )
    p = write_splats(sc, tmp_path / "f.ply")
    data = p.read_bytes()
    body = data[data.index(b"end_header\n") + 11 :]
    names = [ln.split()[-1] for ln in _ply_header(p) if ln.startswith("property")]
    rec = np.frombuffer(body, dtype=np.dtype([(n, "<f4") for n in names]))
    assert rec["f_dc_0"][0] == pytest.approx(0.0, abs=1e-6) and rec["f_dc_0"][1] == pytest.approx(
        1.0, abs=1e-5
    )
    assert rec["opacity"][0] == pytest.approx(0.0, abs=1e-6) and rec["opacity"][1] == pytest.approx(
        2.0, abs=1e-4
    )
    np.testing.assert_allclose([rec["scale_0"][0], rec["scale_2"][0]], np.log([0.01, 0.03]), rtol=1e-6)
    assert rec["rot_0"][0] == 1.0 and rec["rot_1"][1] == 1.0
    # f_rest is channel-major: f_rest[c*K + k] = sh_rest[k, c]
    for k in range(3):
        for c in range(3):
            assert rec[f"f_rest_{c * 3 + k}"][0] == sc.sh_rest[0, k, c]


def test_dot_splat_round_trip_within_quantisation(tmp_path):
    sc = random_cloud(2000, sh=3)
    p = write_splats(sc, tmp_path / "a.splat")
    assert p.stat().st_size == 32 * 2000
    r = read_splats(p)
    np.testing.assert_array_equal(r.means, sc.means)
    np.testing.assert_array_equal(r.scales, sc.scales)
    assert np.abs(r.colors - sc.colors).max() <= 0.5 / 255 + 1e-6
    assert np.abs(r.opacity - sc.opacity).max() <= 0.5 / 255 + 1e-6
    assert _quat_close(r.quats, sc.quats).max() < 0.02
    assert r.sh_rest is None


def test_dot_splat_record_layout(tmp_path):
    sc = SplatCloud(
        means=np.array([[1.0, 2.0, 3.0]], np.float32),
        scales=np.array([[0.1, 0.2, 0.3]], np.float32),
        quats=np.array([[0.0, 0.0, 0.0, -1.0]], np.float32),  # w-first
        colors=np.array([[1.0, 0.5, 0.0]], np.float32),
        opacity=np.array([0.25], np.float32),
    )
    raw = write_splats(sc, tmp_path / "one.splat").read_bytes()
    pos = struct.unpack("<3f", raw[0:12])
    scl = struct.unpack("<3f", raw[12:24])
    assert pos == (1.0, 2.0, 3.0) and scl == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(raw[24:28]) == (255, 128, 0, 64)
    assert tuple(raw[28:32]) == (128, 128, 128, 0)  # q*128+128
    sc.quats[:] = (-1.0, 0.0, 0.0, 0.0)  # w < 0 is canonicalised to w > 0 (same rotation)
    raw = write_splats(sc, tmp_path / "two.splat").read_bytes()
    assert tuple(raw[28:32]) == (255, 128, 128, 128)  # 1*128+128 = 256 clipped to 255


def _write_ply(
    path, fmt: str, props: list[tuple[str, str]], rows: list[list[float]], extra: str = ""
) -> None:
    hdr = ["ply", f"format {fmt} 1.0", "comment test", extra, f"element vertex {len(rows)}"]
    hdr += [f"property {t} {n}" for n, t in props] + ["end_header"]
    head = ("\n".join(h for h in hdr if h) + "\n").encode("ascii")
    if fmt == "ascii":
        body = "".join(" ".join(str(v) for v in r) + "\n" for r in rows).encode("ascii")
    else:
        e = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(n, e + {"float": "f4", "uchar": "u1", "double": "f8"}[t]) for n, t in props])
        body = np.array([tuple(r) for r in rows], dtype=dt).tobytes()
    path.write_bytes(head + body)


def test_reads_point_cloud_big_endian_and_ascii_plys(tmp_path):
    props = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("red", "uchar"),
        ("green", "uchar"),
        ("blue", "uchar"),
    ]
    rows = [[0, 0, 0, 255, 0, 0], [1, 2, 3, 0, 255, 51]]
    for fmt in ("binary_little_endian", "binary_big_endian", "ascii"):
        p = tmp_path / f"{fmt}.ply"
        _write_ply(p, fmt, props, rows)
        sc = read_splats(p)
        np.testing.assert_allclose(sc.means, [[0, 0, 0], [1, 2, 3]])
        np.testing.assert_allclose(sc.colors, [[1, 0, 0], [0, 1, 0.2]], atol=1e-6)
        np.testing.assert_allclose(sc.opacity, 1.0)
        np.testing.assert_allclose(sc.quats, [[1, 0, 0, 0]] * 2)
        assert np.all(sc.scales > 0)


def test_3dgs_ply_with_double_positions_and_unnormalised_quats(tmp_path):
    props = [
        ("x", "double"),
        ("y", "double"),
        ("z", "double"),
        ("f_dc_0", "float"),
        ("f_dc_1", "float"),
        ("f_dc_2", "float"),
        ("opacity", "float"),
        ("scale_0", "float"),
        ("scale_1", "float"),
        ("scale_2", "float"),
        ("rot_0", "float"),
        ("rot_1", "float"),
        ("rot_2", "float"),
        ("rot_3", "float"),
    ]
    rows = [[0.5, 0, 0, 0, 0, 0, 0, -2, -2, -2, 2, 0, 0, 0], [0, 0, 0, 10, -10, 0, -50, 0, 0, 0, 0, 0, 0, 0]]
    p = tmp_path / "g.ply"
    _write_ply(p, "binary_little_endian", props, rows)
    sc = read_splats(p)
    np.testing.assert_allclose(sc.colors, [[0.5] * 3, [1.0, 0.0, 0.5]], atol=1e-6)  # clipped to [0, 1]
    np.testing.assert_allclose(sc.opacity, [0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(sc.scales[0], np.exp(-2.0), rtol=1e-6)
    np.testing.assert_allclose(sc.quats, [[1, 0, 0, 0], [1, 0, 0, 0]])  # normalised / zero -> identity


def test_read_errors(tmp_path):
    with pytest.raises(InputError):
        read_splats(tmp_path / "missing.ply")
    bad = tmp_path / "bad.ply"
    bad.write_bytes(b"not a ply at all")
    with pytest.raises(S.SplatError):
        read_splats(bad)
    trunc = tmp_path / "trunc.ply"
    write_splats(random_cloud(10), trunc)
    trunc.write_bytes(trunc.read_bytes()[:-40])
    with pytest.raises(S.SplatError, match="truncated"):
        read_splats(trunc)
    odd = tmp_path / "odd.splat"
    odd.write_bytes(b"\x00" * 33)
    with pytest.raises(S.SplatError):
        read_splats(odd)
    other = tmp_path / "x.spz"
    other.write_bytes(b"\x00")
    with pytest.raises(InputError):
        read_splats(other)
    with pytest.raises(InputError):
        write_splats(random_cloud(3), tmp_path / "x.obj")
    wrong = random_cloud(3)
    wrong.scales = wrong.scales[:2]
    with pytest.raises(InputError):
        write_splats(wrong, tmp_path / "w.ply")


# --------------------------------------------------------------------------- decimation
def test_merging_two_gaussians_matches_analytic_moments():
    r2 = S.quat_to_rotmat(np.array([[0.9, 0.1, 0.3, 0.2]]))[0]
    sc = SplatCloud(
        means=np.array([[0, 0, 0], [0.1, 0.05, -0.02]], np.float32),
        scales=np.array([[0.1, 0.05, 0.02], [0.05, 0.08, 0.03]], np.float32),
        quats=S.rotmat_to_quat(np.stack([np.eye(3), r2])).astype(np.float32),
        colors=np.array([[1, 0, 0], [0, 0, 1]], np.float32),
        opacity=np.array([0.5, 0.8], np.float32),
    )
    m = decimate_splats(sc, 1)
    assert len(m) == 1
    w = sc.opacity.astype(np.float64) * np.prod(sc.scales.astype(np.float64), axis=1)
    mu = (w[:, None] * sc.means).sum(0) / w.sum()
    d = sc.means - mu
    cov = S.splat_covariances(sc)
    sigma = (w[:, None, None] * (cov + d[:, :, None] * d[:, None, :])).sum(0) / w.sum()
    np.testing.assert_allclose(m.means[0], mu, atol=1e-6)
    np.testing.assert_allclose(S.splat_covariances(m)[0], sigma, atol=1e-7, rtol=1e-4)
    assert m.opacity[0] == pytest.approx(1 - 0.5 * 0.2, abs=1e-6)
    np.testing.assert_allclose(m.colors[0], (w[:, None] * sc.colors).sum(0) / w.sum(), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(m.quats[0]), 1.0, atol=1e-6)


def _sphere_surface_cloud(n: int, seed: int = 1) -> SplatCloud:
    """Overlapping flat splats on a unit sphere (a typical surface-like 3DGS cloud)."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    t1 = np.cross(p, [0.0, 0.0, 1.0]) + np.cross(p, [0.0, 1.0, 0.0]) * 1e-3
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(p, t1)
    q = S.rotmat_to_quat(np.stack([t1, t2, p], axis=2))
    spacing = np.sqrt(4 * np.pi / n)
    return SplatCloud(
        p.astype(np.float32),
        np.tile([[spacing, spacing, spacing / 8]], (n, 1)).astype(np.float32),
        q.astype(np.float32),
        rng.random((n, 3)).astype(np.float32),
        rng.uniform(0.3, 1.0, n).astype(np.float32),
    )


@pytest.mark.parametrize("kind", ["surface", "volume"])
def test_decimate_count_mass_and_centroid(kind):
    sc = _sphere_surface_cloud(200_000) if kind == "surface" else random_cloud(200_000, seed=1)
    out = decimate_splats(sc, 20_000)
    assert 0.5 * 20_000 <= len(out) <= 20_000
    w_in, w_out = S.importance(sc), S.importance(out)
    c_in = (w_in[:, None] * sc.means).sum(0) / w_in.sum()
    c_out = (w_out[:, None] * out.means).sum(0) / w_out.sum()
    extent = np.ptp(sc.means, axis=0).max()
    assert np.linalg.norm(c_out - c_in) < 0.01 * extent

    # total coverage mass (opacity x projected area of the two largest axes) is kept within a
    # small factor: moment matching preserves the weights exactly, while the contract opacity
    # rule 1 - prod(1 - a) saturates when many overlapping splats merge
    def mass(c: SplatCloud) -> float:
        s = np.sort(c.scales.astype(np.float64), axis=1)
        return float((c.opacity * s[:, 1] * s[:, 2]).sum())

    assert 0.2 < mass(out) / mass(sc) < 5
    np.testing.assert_allclose(sc.means.min(0), out.means.min(0), atol=0.15 * extent)
    np.testing.assert_allclose(sc.means.max(0), out.means.max(0), atol=0.15 * extent)
    if kind == "surface":
        np.testing.assert_allclose(np.linalg.norm(out.means, axis=1), 1.0, atol=0.02)  # still on the sphere
    # sorted by importance, valid quaternions and opacities, DC colour only
    assert np.all(np.diff(w_out) <= 1e-12)
    np.testing.assert_allclose(np.linalg.norm(out.quats, axis=1), 1.0, atol=1e-5)
    assert out.opacity.min() >= S.MIN_OPACITY - 1e-6 and out.opacity.max() <= 1.0


def test_group_merge_preserves_mass_centroid_and_second_moment():
    """Moment matching: with W_k = sum of member weights, the merged splats carry exactly the
    total mass, the weighted centroid and the second moment of the input."""
    sc = random_cloud(5000, seed=8)
    rng = np.random.default_rng(0)
    m = 300
    labels = rng.integers(0, m, 5000)
    labels[:m] = np.arange(m)  # every label used
    centers = np.zeros((m, 3))
    out = S._merge_by_labels(sc, labels, m, centers, None)
    w = S.importance(sc)
    W = np.bincount(labels, weights=w, minlength=m)
    assert W.sum() == pytest.approx(w.sum())
    np.testing.assert_allclose(
        (W[:, None] * out.means).sum(0), (w[:, None] * sc.means).sum(0), rtol=1e-5, atol=1e-9
    )
    mm_in = (w[:, None, None] * (S.splat_covariances(sc) + sc.means[:, :, None] * sc.means[:, None, :])).sum(
        0
    )
    mm_out = (
        W[:, None, None]
        * (S.splat_covariances(out) + out.means[:, :, None].astype(np.float64) * out.means[:, None, :])
    ).sum(0)
    np.testing.assert_allclose(mm_out, mm_in, rtol=1e-4)


def test_decimate_prunes_transparent_and_huge_splats():
    sc = random_cloud(1000, sh=3, seed=2)
    sc.opacity[:100] = 0.01
    sc.scales[100:105] = 50.0  # absurd floaters
    out = decimate_splats(sc, 5000)  # no merging needed: only pruning + sorting
    assert len(out) == 895
    assert out.sh_rest is not None and out.sh_rest.shape == (895, 3, 3)
    assert out.scales.max() < 1.0 and out.opacity.min() >= 0.04
    kept = {tuple(x) for x in sc.means[105:]}
    assert {tuple(x) for x in out.means} == kept


def test_decimate_keeps_singletons_unchanged_and_drops_sh_when_merging():
    sc = random_cloud(50, sh=3, seed=3, spread=10.0)
    dup = random_cloud(50, sh=3, seed=4)
    dup.means = sc.means + 1e-4  # each splat gets a near-duplicate
    both = SplatCloud(
        *(
            np.concatenate([getattr(sc, k), getattr(dup, k)])
            for k in ("means", "scales", "quats", "colors", "opacity", "sh_rest")
        )
    )
    out = decimate_splats(both, 60)
    assert len(out) <= 60 and out.sh_rest is None


def test_decimate_validation_and_cancel():
    sc = random_cloud(100)
    with pytest.raises(InputError):
        decimate_splats(sc, 0)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        decimate_splats(random_cloud(5000), 100, cancel=tok)
    empty = SplatCloud(
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.float32),
        np.zeros((0, 4), np.float32),
        np.zeros((0, 3), np.float32),
        np.zeros(0, np.float32),
    )
    assert len(decimate_splats(empty, 10)) == 0


@pytest.mark.slow
def test_decimate_three_million_splats_is_fast():
    sc = random_cloud(3_000_000, seed=5)
    t = time.perf_counter()
    out = decimate_splats(sc, 750_000)
    dt = time.perf_counter() - t
    assert 0.6 * 750_000 <= len(out) <= 750_000
    assert dt < 60.0


# --------------------------------------------------------------------------- crop / orientation
def test_crop_splats_radius_and_aabb():
    sc = random_cloud(1000, seed=6)
    sc.means[:10] = 100.0  # outliers
    out = crop_splats(sc, radius_pct=98.0)
    assert np.abs(out.means).max() < 50 and 970 <= len(out) <= 990
    box = crop_splats(sc, radius_pct=None, aabb=[[-1, -1, -1], [1, 1, 1]])
    assert np.all(np.abs(box.means) <= 1) and len(box) > 100
    assert len(crop_splats(sc, radius_pct=100.0)) == 1000
    with pytest.raises(InputError):
        crop_splats(sc, aabb=[1, 2, 3])
    with pytest.raises(InputError):
        crop_splats(sc, radius_pct=-5.0)


def _rot(axis, deg) -> np.ndarray:
    a = np.radians(deg)
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _floor_and_person(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    floor = np.column_stack(
        [rng.uniform(-1.5, 1.5, 6000), rng.normal(0, 0.003, 6000), rng.uniform(-1.5, 1.5, 6000)]
    )
    person = np.column_stack([rng.normal(0, 0.15, 8000), rng.uniform(0, 1.7, 8000), rng.normal(0, 0.1, 8000)])
    return np.concatenate([floor, person])


def _cloud_from_points(p: np.ndarray, sh: int = 0) -> SplatCloud:
    n = len(p)
    rng = np.random.default_rng(1)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return SplatCloud(
        p.astype(np.float32),
        np.tile([[0.01, 0.02, 0.005]], (n, 1)).astype(np.float32),
        q.astype(np.float32),
        np.full((n, 3), 0.5, np.float32),
        np.ones(n, np.float32),
        rng.normal(size=(n, sh, 3)).astype(np.float32) if sh else None,
    )


@pytest.mark.parametrize(
    "R",
    [np.eye(3), _rot([1, 0.3, 0.2], 20), _rot([1, 0, 0], 180), _rot([1, 0, 0], 180) @ _rot([0, 0, 1], 15)],
    ids=["upright", "tilted", "y-down", "y-down-tilted"],
)
def test_normalize_orientation_finds_the_floor(R):
    p = _floor_and_person()
    sc = _cloud_from_points(p @ R.T + [3.0, -2.0, 5.0])
    out = normalize_orientation(sc)
    floor_y = out.means[:6000, 1]
    assert abs(floor_y.mean()) < 0.003 and floor_y.std() < 0.006
    assert out.means[6000:, 1].max() == pytest.approx(1.7, abs=0.02)
    np.testing.assert_allclose(np.median(out.means[:, [0, 2]], axis=0), 0.0, atol=1e-5)


def test_normalize_orientation_with_explicit_up_rotates_splats_and_sh():
    p = _floor_and_person()
    R = _rot([0.2, 0.1, 1.0], 35)
    sc = _cloud_from_points(p @ R.T, sh=15)
    up = R @ np.array([0.0, 1.0, 0.0])
    out = normalize_orientation(sc, up=up)
    assert abs(out.means[:, 1].min() - np.percentile(out.means[:, 1], 0)) < 1e-6
    assert abs(np.percentile(out.means[:, 1], 0.5)) < 1e-5
    # the recovered frame is the original up to a rotation about y + translation
    Rt = _rot(up, 0)  # identity helper for readability
    assert np.allclose(Rt, np.eye(3))
    # covariances rotate with the cloud: Sigma' = Q Sigma Q^T with Q the applied rotation
    q_apply = S._rotation_to_y(up)
    c0 = S.splat_covariances(sc)
    c1 = S.splat_covariances(out)
    np.testing.assert_allclose(c1[:50], q_apply @ c0[:50] @ q_apply.T, atol=1e-9)
    # view-dependent colour is rotated: SH evaluated at rotated directions matches
    d = S._fibonacci_sphere(40)
    before = np.einsum("mk,kc->mc", S._sh_basis(d, 15), sc.sh_rest[0].astype(np.float64))
    after = np.einsum("mk,kc->mc", S._sh_basis(d @ q_apply.T, 15), out.sh_rest[0].astype(np.float64))
    np.testing.assert_allclose(after, before, atol=1e-5)
    with pytest.raises(InputError):
        normalize_orientation(sc, up=np.zeros(3))


def test_normalize_orientation_without_floor_falls_back_to_lowest_points():
    rng = np.random.default_rng(3)
    blob = rng.normal(size=(5000, 3)) * [0.2, 0.8, 0.2]
    out = normalize_orientation(_cloud_from_points(blob))
    assert abs(np.percentile(out.means[:, 1], 0.5)) < 1e-5
    assert np.ptp(out.means[:, 1]) > 3 * np.ptp(out.means[:, 0])  # the long axis stays vertical


def test_quaternion_helpers_round_trip():
    rng = np.random.default_rng(7)
    q = rng.normal(size=(2000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[:4] = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]  # 180-degree and identity cases
    R = S.quat_to_rotmat(q)
    np.testing.assert_allclose(R @ np.swapaxes(R, 1, 2), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
    q2 = S.rotmat_to_quat(R)
    assert _quat_close(q2, q).max() < 1e-9
    a, b = q[10:20], q[20:30]
    np.testing.assert_allclose(
        S.quat_to_rotmat(S.quat_mul(a, b)), S.quat_to_rotmat(a) @ S.quat_to_rotmat(b), atol=1e-12
    )


def test_sh_rotation_matrix_is_exact_per_band():
    R = _rot([0.3, 0.5, 0.1], 70)
    for n in (3, 8, 15):
        M = S.sh_rotation_matrix(R, n)
        c = np.random.default_rng(n).normal(size=n)
        d = S._fibonacci_sphere(50)
        np.testing.assert_allclose(S._sh_basis(d @ R.T, n) @ (M @ c), S._sh_basis(d, n) @ c, atol=1e-10)
    assert S.sh_rotation_matrix(R, 5) is None


# --------------------------------------------------------------------------- mesh -> splats
def _quad(tex: np.ndarray, alpha_mode: str = "OPAQUE") -> Mesh:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    f = np.array([[0, 1, 2], [1, 3, 2]], np.uint32)
    uv = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], np.float32)
    return Mesh(v, f, uv, tex, alpha_mode=alpha_mode)


def test_splats_from_mesh_samples_surface_and_texture():
    tex = np.zeros((64, 64, 3), np.uint8)
    tex[:, :32] = (255, 0, 0)
    tex[:, 32:] = (0, 0, 255)
    sc = splats_from_mesh(_quad(tex), scale_m=0.012)
    spacing = 0.012 / 0.6
    assert len(sc) == pytest.approx(1.0 / spacing**2, rel=0.1)
    np.testing.assert_allclose(sc.means[:, 2], 0.0, atol=1e-7)
    assert sc.means[:, :2].min() >= 0 and sc.means[:, :2].max() <= 1
    left = sc.means[:, 0] < 0.45
    right = sc.means[:, 0] > 0.55
    assert np.allclose(sc.colors[left], [1, 0, 0], atol=0.02) and np.allclose(
        sc.colors[right], [0, 0, 1], atol=0.02
    )
    # flat discs: the thin axis is the face normal (+z)
    R = S.quat_to_rotmat(sc.quats)
    thin_axis = R[:, :, 2]
    np.testing.assert_allclose(np.abs(thin_axis[:, 2]), 1.0, atol=1e-6)
    np.testing.assert_allclose(sc.scales[:, :2], 0.012) and np.testing.assert_allclose(
        sc.scales[:, 2], 0.0012, rtol=1e-5
    )


def test_splats_from_mesh_respects_mask_and_default_scale():
    tex = np.zeros((32, 32, 4), np.uint8)
    tex[..., :3] = 200
    tex[:, 16:, 3] = 255  # right half opaque
    sc = splats_from_mesh(_quad(tex, "MASK"))
    assert sc.means[:, 0].min() > 0.45  # texture u >= 0.5 only
    assert len(sc) > 0 and np.allclose(sc.opacity, 1.0)
    blend = splats_from_mesh(_quad(tex, "BLEND"), scale_m=0.05)
    assert blend.opacity.min() == 0.0 and blend.opacity.max() == 1.0
    with pytest.raises(InputError):
        splats_from_mesh(_quad(tex), scale_m=-1.0)
    empty = Mesh(np.zeros((3, 3), np.float32), np.zeros((0, 3), np.uint32), np.zeros((3, 2), np.float32), tex)
    with pytest.raises(InputError):
        splats_from_mesh(empty)


def test_billboard_to_splats_round_trip(tmp_path, person_alpha):
    from vrphoto.core.types import DepthResult, Intrinsics
    from vrphoto.geometry.billboard import billboard

    h, w = person_alpha.shape
    img = np.full((h, w, 3), 128, np.uint8)
    front, back = billboard(
        img, person_alpha, DepthResult(np.full((h, w), 0.5, np.float32), "t"), Intrinsics.from_hfov(w, h, 60)
    )
    sc = SplatCloud(
        *(
            np.concatenate([getattr(a, k), getattr(b, k)])
            for a, b in [(splats_from_mesh(front), splats_from_mesh(back))]
            for k in ("means", "scales", "quats", "colors", "opacity")
        )
    )
    assert sc.means[:, 1].min() >= -1e-6 and sc.means[:, 1].max() <= 1.7 + 1e-6
    r = read_splats(write_splats(sc, tmp_path / "person.splat"))
    assert len(r) == len(sc)


# --------------------------------------------------------------------------- review additions
def test_decimate_is_robust_to_far_floaters():
    """A few splats a kilometre away must not blow up the voxel grid (which used to
    collapse 200k splats into a few hundred)."""
    sc = random_cloud(200_000, seed=11)
    sc.means[:5] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1]]) * 1e6
    out = decimate_splats(sc, 5000)
    assert 0.6 * 5000 <= len(out) <= 5000
    assert np.sum(np.abs(out.means).max(axis=1) > 1e5) == 5  # floaters kept as singletons
    w_in, w_out = S.importance(sc), S.importance(out)
    c_in = (w_in[5:, None] * sc.means[5:]).sum(0) / w_in[5:].sum()
    near = np.abs(out.means).max(axis=1) < 1e5
    c_out = (w_out[near, None] * out.means[near]).sum(0) / w_out[near].sum()
    assert np.linalg.norm(c_out - c_in) < 0.05


def test_voxel_keys_are_exact_when_the_grid_overflows():
    p = np.array([[0.0, 0.0, 0.0], [1e-9, 0, 0], [1e12, 1e12, 1e12], [1e12, 1e12, 1e12 + 1e-3]])
    keys, q = S._voxel_keys(p, p.min(axis=0), 1e-6)
    assert len(np.unique(keys)) == 3  # (0,0,0) twice, two distinct far voxels
    assert keys[0] == keys[1] and keys[2] != keys[3]


def _nan_cloud() -> SplatCloud:
    sc = random_cloud(2000, sh=3, seed=12)
    sc.means[:3] = np.nan
    sc.scales[3] = np.inf
    return sc


def test_non_finite_splats_are_dropped_not_crashing():
    out = decimate_splats(_nan_cloud(), 100)
    assert 0 < len(out) <= 100 and np.isfinite(out.means).all() and np.isfinite(out.scales).all()
    norm = normalize_orientation(_nan_cloud())
    assert len(norm) == 1996 and np.isfinite(norm.means).all()
    crop = crop_splats(_nan_cloud(), radius_pct=None)
    assert len(crop) == 1996
    with pytest.raises(InputError):
        crop_splats(_nan_cloud(), radius_pct=float("nan"))
    with pytest.raises(InputError):
        crop_splats(_nan_cloud(), aabb=[[0, 0, 0], [np.nan, 1, 1]])


def test_decimate_with_sh_and_merging_drops_sh():
    sc = random_cloud(20_000, sh=15, seed=13)
    out = decimate_splats(sc, 2000)
    assert len(out) <= 2000 and out.sh_rest is None
    assert sc.sh_rest is not None and sc.sh_rest.shape == (20_000, 15, 3)  # input untouched


@pytest.mark.parametrize(
    "header",
    [
        b"ply\nformat binary_little_endian 1.0\nelement vertex\nproperty float x\nend_header\n",
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float\nend_header\n",
        b"ply\nformat\nelement vertex 1\nproperty float x\nend_header\n",
        b"ply\nformat binary_little_endian 1.0\nelement vertex -5\nproperty float x\nend_header\n",
        b"ply\nformat binary_little_endian 1.0\nelement vertex x\nproperty float x\nend_header\n",
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty list uchar\nend_header\n",
    ],
    ids=["no-count", "no-name", "no-format", "negative", "nan-count", "short-list"],
)
def test_malformed_ply_headers_raise_splat_error(tmp_path, header):
    p = tmp_path / "bad.ply"
    p.write_bytes(header + b"\x00" * 16)
    with pytest.raises(S.SplatError):
        read_splats(p)


def test_ply_with_long_comment_header(tmp_path):
    sc = random_cloud(5)
    p = write_splats(sc, tmp_path / "c.ply")
    data = p.read_bytes()
    comments = b"".join(b"comment %d padding padding padding padding padding\n" % i for i in range(3000))
    i = data.index(b"element vertex")
    p.write_bytes(data[:i] + comments + data[i:])  # > 64 KiB header
    np.testing.assert_array_equal(read_splats(p).means, sc.means)


def test_ply_reader_lays_out_f_rest_channel_major(tmp_path):
    """Pins f_rest[c*K + k] -> sh_rest[k, c] on a hand-written file (reader side)."""
    k = 3
    props = [(n, "float") for n in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2")]
    props += [(f"f_rest_{i}", "float") for i in range(3 * k)]
    rows = [[0, 0, 0, 0, 0, 0, *range(9)]]
    p = tmp_path / "sh.ply"
    _write_ply(p, "binary_little_endian", props, rows)
    sh = read_splats(p).sh_rest
    assert sh is not None and sh.shape == (1, k, 3)
    for c in range(3):
        for kk in range(k):
            assert sh[0, kk, c] == c * k + kk


def test_splats_from_mesh_validation_and_coverage_cap():
    tex = np.full((8, 8, 3), 100, np.uint8)
    quad = _quad(tex)
    capped = splats_from_mesh(quad, scale_m=0.001, max_splats=1000)
    assert len(capped) <= 1100
    # fewer splats -> each one is larger so the unit quad stays covered
    assert capped.scales[0, 0] == pytest.approx(0.6 * np.sqrt(1.0 / 1000), rel=1e-5)
    bad_v = quad.vertices.copy()
    bad_v[3] = np.nan
    half = splats_from_mesh(Mesh(bad_v, quad.faces, quad.uvs, tex), scale_m=0.02)
    assert len(half) > 0 and np.isfinite(half.means).all()
    assert half.means[:, 0].max() <= 1.0 and (half.means[:, 0] + half.means[:, 1]).max() <= 1.0 + 1e-6
    with pytest.raises(InputError):
        splats_from_mesh(quad, max_splats=0)
    with pytest.raises(InputError):
        splats_from_mesh(Mesh(quad.vertices, np.array([[0, 1, 9]], np.uint32), quad.uvs, tex))
