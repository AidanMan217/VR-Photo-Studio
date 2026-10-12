"""3D Gaussian splat I/O and processing (algorithm J).

File formats (coordinates are stored verbatim - a :class:`SplatCloud` is in the glTF
frame, so files we write are y-up; trainer outputs in another frame should go through
:func:`normalize_orientation`):

* **3DGS PLY** (``.ply``, as written by the reference 3DGS / gsplat / Brush trainers):
  ``x y z [nx ny nz] f_dc_0..2 f_rest_* opacity scale_0..2 rot_0..3`` (float32, binary
  little endian).  ``colour = 0.5 + C0 * f_dc`` with ``C0 = 0.28209479177387814``,
  ``opacity`` is a logit, ``scale_*`` are log standard deviations, ``rot_*`` is the
  quaternion ``(w, x, y, z)``; ``f_rest`` holds the higher SH bands channel-major
  (``f_rest[c*K + k] = sh_rest[k, c]``).  Plain point clouds (``red green blue``) load
  with default scales.  Binary big endian and ASCII PLYs are read too.
* **.splat** (antimatter15 viewer): 32-byte records - position 3 x f32, linear scale
  3 x f32, RGBA u8 (A = opacity), quaternion u8 ``q*128+128`` (w first).

Decimation (:func:`decimate_splats`): prune splats with opacity < 0.04 and absurdly large
splats, then merge splats per voxel by moment matching with weights
``w = opacity * scale_x * scale_y * scale_z``: ``mu = sum(w mu_i)/W``,
``Sigma = sum(w (Sigma_i + (mu_i - mu)(mu_i - mu)^T))/W``, colour = weighted mean (DC
only - higher SH bands are dropped when splats are merged), opacity
``1 - prod(1 - alpha_i)``.  The voxel size is searched so the number of occupied voxels
is just below ``max_n``; singleton voxels keep their splat unchanged.  Everything is
vectorised with ``np.bincount`` (3 M splats take a few seconds and < 1 GB).

Building blocks for combining clouds (SHARP 360): :func:`transform_splats` (similarity
transform, rotating orientations and SH), :func:`scale_splats`, :func:`concat_splats`,
:func:`subset_splats` and :func:`fit_plane_ransac`.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import InputError, VRPhotoError
from ..core.fsutil import replace_with_retry
from ..core.log import get_logger
from ..core.types import CancelToken, Mesh, ProgressReporter, SplatCloud
from .mesh import _as_reporter, _check

log = get_logger(__name__)

C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

MIN_OPACITY = 0.04
_HUGE_FRAC = 0.25  # of the robust scene extent ...
_HUGE_MEDIAN = 20.0  # ... and this many times the median splat size
_SPLAT_DTYPE = np.dtype(
    [("pos", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("rot", "u1", (4,))]
)
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip
_CHUNK = 500_000
_RANSAC_CHUNK = 65_536  # points per block of the RANSAC inlier count


class SplatError(InputError):
    """A splat file is malformed or uses an unsupported variant."""


# --------------------------------------------------------------------------- quaternions
def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """``(N, 4)`` unit quaternions ``(w, x, y, z)`` -> ``(N, 3, 3)`` rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def rotmat_to_quat(r: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)`` rotation matrices -> unit quaternions ``(w, x, y, z)`` with w >= 0."""
    r = np.asarray(r, dtype=np.float64)
    m00, m11, m22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    # Shepperd: build all four candidates and keep the numerically largest
    k = np.stack(
        [
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ],
        axis=-1,
    )
    pick = np.argmax(k, axis=-1)
    q = np.empty(r.shape[:-2] + (4,))
    t = np.sqrt(np.maximum(np.take_along_axis(k, pick[..., None], -1)[..., 0], 1e-300))
    s = 0.5 / t
    c0 = pick == 0
    c1 = pick == 1
    c2 = pick == 2
    c3 = pick == 3
    q[c0] = np.stack(
        [
            0.5 * t,
            (r[..., 2, 1] - r[..., 1, 2]) * s,
            (r[..., 0, 2] - r[..., 2, 0]) * s,
            (r[..., 1, 0] - r[..., 0, 1]) * s,
        ],
        -1,
    )[c0]
    q[c1] = np.stack(
        [
            (r[..., 2, 1] - r[..., 1, 2]) * s,
            0.5 * t,
            (r[..., 0, 1] + r[..., 1, 0]) * s,
            (r[..., 0, 2] + r[..., 2, 0]) * s,
        ],
        -1,
    )[c1]
    q[c2] = np.stack(
        [
            (r[..., 0, 2] - r[..., 2, 0]) * s,
            (r[..., 0, 1] + r[..., 1, 0]) * s,
            0.5 * t,
            (r[..., 1, 2] + r[..., 2, 1]) * s,
        ],
        -1,
    )[c2]
    q[c3] = np.stack(
        [
            (r[..., 1, 0] - r[..., 0, 1]) * s,
            (r[..., 0, 2] + r[..., 2, 0]) * s,
            (r[..., 1, 2] + r[..., 2, 1]) * s,
            0.5 * t,
        ],
        -1,
    )[c3]
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q[q[..., 0] < 0] *= -1.0
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of ``(..., 4)`` quaternions (w first)."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def splat_covariances(sc: SplatCloud) -> np.ndarray:
    """``(N, 3, 3)`` covariance matrices ``R diag(s^2) R^T`` (float64)."""
    r = quat_to_rotmat(sc.quats)
    s2 = np.asarray(sc.scales, dtype=np.float64) ** 2
    return (r * s2[:, None, :]) @ np.swapaxes(r, 1, 2)


# --------------------------------------------------------------------------- SH rotation
def _sh_basis(dirs: np.ndarray, n_rest: int) -> np.ndarray:
    """Real SH basis (3DGS convention) for coefficients 1..n_rest at unit ``dirs``."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    cols = [-_SH_C1 * y, _SH_C1 * z, -_SH_C1 * x]
    if n_rest > 3:
        xx, yy, zz = x * x, y * y, z * z
        cols += [
            _SH_C2[0] * x * y,
            _SH_C2[1] * y * z,
            _SH_C2[2] * (2 * zz - xx - yy),
            _SH_C2[3] * x * z,
            _SH_C2[4] * (xx - yy),
        ]
        if n_rest > 8:
            cols += [
                _SH_C3[0] * y * (3 * xx - yy),
                _SH_C3[1] * x * y * z,
                _SH_C3[2] * y * (4 * zz - xx - yy),
                _SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                _SH_C3[4] * x * (4 * zz - xx - yy),
                _SH_C3[5] * z * (xx - yy),
                _SH_C3[6] * x * (xx - 3 * yy),
            ]
    return np.stack(cols[:n_rest], axis=1)


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def sh_rotation_matrix(R: np.ndarray, n_rest: int) -> np.ndarray | None:
    """Matrix ``M`` with ``sh_rest' = M @ sh_rest`` for a world rotation ``R``.

    Exact for complete bands (``n_rest`` in 3, 8, 15): the rotated function
    ``f'(d) = f(R^T d)`` is fitted in the same basis by least squares on 128 directions.
    Returns None for other coefficient counts."""
    if n_rest not in (3, 8, 15):
        return None
    d = _fibonacci_sphere(128)
    Y = _sh_basis(d, n_rest)
    Yr = _sh_basis(d @ np.asarray(R, dtype=np.float64), n_rest)
    return np.linalg.lstsq(Y, Yr, rcond=None)[0]


# --------------------------------------------------------------------------- helpers
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _subset(sc: SplatCloud, idx: np.ndarray) -> SplatCloud:
    return SplatCloud(
        means=np.ascontiguousarray(sc.means[idx], dtype=np.float32),
        scales=np.ascontiguousarray(sc.scales[idx], dtype=np.float32),
        quats=np.ascontiguousarray(sc.quats[idx], dtype=np.float32),
        colors=np.ascontiguousarray(sc.colors[idx], dtype=np.float32),
        opacity=np.ascontiguousarray(sc.opacity[idx], dtype=np.float32),
        sh_rest=None if sc.sh_rest is None else np.ascontiguousarray(sc.sh_rest[idx], dtype=np.float32),
    )


def _validate(sc: SplatCloud) -> int:
    n = len(sc)
    shapes = {
        "means": (sc.means, (n, 3)),
        "scales": (sc.scales, (n, 3)),
        "quats": (sc.quats, (n, 4)),
        "colors": (sc.colors, (n, 3)),
        "opacity": (sc.opacity, (n,)),
    }
    for name, (arr, shape) in shapes.items():
        if np.shape(arr) != shape:
            raise InputError(f"Splat {name} must have shape {shape} (got {np.shape(arr)}).")
    if sc.sh_rest is not None and (
        np.ndim(sc.sh_rest) != 3 or np.shape(sc.sh_rest)[0] != n or np.shape(sc.sh_rest)[2] != 3
    ):
        raise InputError(f"Splat sh_rest must have shape (N, K, 3) (got {np.shape(sc.sh_rest)}).")
    return n


def _finite_mask(sc: SplatCloud) -> np.ndarray:
    """Splats whose position and scale are finite (NaN / inf ones cannot be processed)."""
    return np.isfinite(np.asarray(sc.means)).all(axis=1) & np.isfinite(np.asarray(sc.scales)).all(axis=1)


def _drop_non_finite(sc: SplatCloud, what: str) -> SplatCloud:
    ok = _finite_mask(sc)
    if ok.all():
        return sc
    log.warning("%s: ignoring %d splats with non-finite position or scale", what, int((~ok).sum()))
    return _subset(sc, np.nonzero(ok)[0])


def _normalized_quats(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    bad = ~np.isfinite(n[:, 0]) | (n[:, 0] < 1e-12)
    out = q / np.where(bad[:, None], 1.0, n)
    out[bad] = (1.0, 0.0, 0.0, 0.0)
    return out.astype(np.float32)


def importance(sc: SplatCloud) -> np.ndarray:
    """Per-splat weight ``opacity * scale_x * scale_y * scale_z`` (float64)."""
    return np.asarray(sc.opacity, dtype=np.float64) * np.prod(np.asarray(sc.scales, dtype=np.float64), axis=1)


def _sorted_by_importance(sc: SplatCloud) -> SplatCloud:
    order = np.argsort(-importance(sc), kind="stable")
    return _subset(sc, order)


def _atomic_replace(tmp: Path, path: Path) -> None:
    try:
        replace_with_retry(tmp, path)  # waits out antivirus / indexer locks on Windows
    except OSError as e:
        tmp.unlink(missing_ok=True)
        hint = (
            "The file is busy (another program or antivirus); try again."
            if isinstance(e, PermissionError)
            else None
        )
        raise VRPhotoError(f"Cannot write {path.name}: {e}", hint=hint) from e


# --------------------------------------------------------------------------- PLY
def _parse_ply_header(
    data: bytes, name: str
) -> tuple[str, list[tuple[str, int, list[tuple[str, Any]]]], int]:
    m = re.search(rb"end_header[ \t]*\r?\n", data)
    if not data.startswith(b"ply") or m is None:
        raise SplatError(f"{name} is not a PLY file (missing 'ply' / 'end_header').")
    header = data[: m.start()].decode("ascii", errors="replace").splitlines()
    fmt = ""
    elements: list[tuple[str, int, list[tuple[str, Any]]]] = []
    for line in header[1:]:
        parts = line.split()
        if not parts or parts[0] in ("comment", "obj_info"):
            continue
        try:
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                count = int(parts[2])
                if count < 0:
                    raise ValueError("negative element count")
                elements.append((parts[1], count, []))
            elif parts[0] == "property":
                if not elements:
                    raise SplatError(f"{name}: PLY property before any element.")
                if parts[1] == "list":
                    elements[-1][2].append((parts[4], ("list", parts[2], parts[3])))
                else:
                    t = _PLY_TYPES.get(parts[1])
                    if t is None:
                        raise SplatError(f"{name}: unsupported PLY property type {parts[1]!r}.")
                    elements[-1][2].append((parts[2], t))
        except (IndexError, ValueError) as e:
            raise SplatError(f"{name}: malformed PLY header line {line.strip()!r}.") from e
    if fmt not in ("binary_little_endian", "binary_big_endian", "ascii"):
        raise SplatError(f"{name}: unsupported PLY format {fmt!r}.")
    return fmt, elements, m.end()


def _read_ply(path: Path) -> SplatCloud:
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)  # headers with many comments can exceed 64 KiB
    fmt, elements, hlen = _parse_ply_header(head, path.name)
    names = [e[0] for e in elements]
    if "vertex" not in names:
        raise SplatError(f"{path.name} has no 'vertex' element.")
    if any(e[0] in ("chunk",) for e in elements):
        raise SplatError(
            f"{path.name} is a compressed (SuperSplat) PLY, which is not supported.",
            hint="Decompress it to a standard 3DGS PLY first.",
        )
    endian = "<" if fmt == "binary_little_endian" else ">"
    offset = hlen
    vdtype = None
    count = 0
    data = None
    if fmt == "ascii":
        text = path.read_bytes()[hlen:].decode("ascii", errors="replace").splitlines()
        line = 0
        for ename, ecount, props in elements:
            if ename == "vertex":
                if any(isinstance(t, tuple) for _, t in props):
                    raise SplatError(f"{path.name}: list properties in 'vertex' are not supported.")
                rows = [ln for ln in text[line : line + ecount]]
                arr = np.loadtxt(rows, dtype=np.float64, ndmin=2) if ecount else np.zeros((0, len(props)))
                if arr.shape != (ecount, len(props)):
                    raise SplatError(f"{path.name}: the ASCII vertex data is truncated or malformed.")
                data = {p: arr[:, i] for i, (p, _) in enumerate(props)}
                count = ecount
                break
            line += ecount
    else:
        for ename, ecount, props in elements:
            if any(isinstance(t, tuple) for _, t in props):
                raise SplatError(f"{path.name}: PLY list properties are not supported in splat files.")
            dt = np.dtype([(p, endian + t) for p, t in props])
            if ename == "vertex":
                vdtype, count = dt, ecount
                break
            offset += dt.itemsize * ecount
        assert vdtype is not None
        need = offset + vdtype.itemsize * count
        size = path.stat().st_size
        if size < need:
            raise SplatError(f"{path.name} is truncated ({size} of {need} bytes).")
        rec = np.fromfile(path, dtype=vdtype, count=count, offset=offset)
        data = {p: rec[p] for p in vdtype.names or ()}
    assert data is not None
    return _cloud_from_fields(data, count, path.name)


def _cloud_from_fields(data: dict[str, np.ndarray], n: int, name: str) -> SplatCloud:
    def col(key: str) -> np.ndarray:
        return np.asarray(data[key], dtype=np.float32)

    if not all(k in data for k in ("x", "y", "z")):
        raise SplatError(f"{name} has no x/y/z vertex positions.")
    means = np.stack([col("x"), col("y"), col("z")], axis=1)
    if all(f"f_dc_{i}" in data for i in range(3)):
        colors = 0.5 + C0 * np.stack([col(f"f_dc_{i}") for i in range(3)], axis=1)
    elif all(k in data for k in ("red", "green", "blue")):
        rgb = np.stack([col("red"), col("green"), col("blue")], axis=1)
        colors = (
            rgb / 255.0 if np.asarray(data["red"]).dtype.kind in "iu" or rgb.max(initial=0) > 1.0 else rgb
        )
    else:
        colors = np.full((n, 3), 0.5, np.float32)
    opacity = _sigmoid(col("opacity")) if "opacity" in data else np.ones(n, np.float32)
    if all(f"scale_{i}" in data for i in range(3)):
        scales = np.exp(np.clip(np.stack([col(f"scale_{i}") for i in range(3)], axis=1), -60.0, 30.0))
    else:
        ext = np.ptp(means, axis=0) if n else np.ones(3)
        vol = float(np.prod(np.maximum(ext, 1e-6)))
        s0 = 0.5 * (vol / max(n, 1)) ** (1.0 / 3.0)
        scales = np.full((n, 3), s0, np.float32)
    if all(f"rot_{i}" in data for i in range(4)):
        quats = _normalized_quats(np.stack([col(f"rot_{i}") for i in range(4)], axis=1))
    else:
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    rest_keys = sorted((k for k in data if k.startswith("f_rest_")), key=lambda k: int(k.split("_")[-1]))
    sh_rest = None
    if rest_keys:
        if len(rest_keys) % 3:
            log.warning(
                "%s: %d f_rest coefficients is not a multiple of 3; ignoring them", name, len(rest_keys)
            )
        else:
            k = len(rest_keys) // 3
            # fill (N, K, 3) column by column: one output-sized allocation (3M splats with
            # 45 coefficients are 540 MB, so no stacked / transposed temporaries)
            sh_rest = np.empty((n, k, 3), np.float32)
            for i, key in enumerate(rest_keys):
                sh_rest[:, i % k, i // k] = col(key)
            np.nan_to_num(sh_rest, copy=False)
    sc = SplatCloud(
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        quats=quats.astype(np.float32),
        colors=np.clip(np.nan_to_num(colors, nan=0.5), 0.0, 1.0).astype(np.float32),
        opacity=np.clip(np.nan_to_num(opacity), 0.0, 1.0).astype(np.float32),
        sh_rest=sh_rest,
    )
    return _drop_non_finite(sc, name)


def _write_ply(sc: SplatCloud, path: Path) -> None:
    n = len(sc)
    k = 0 if sc.sh_rest is None else int(sc.sh_rest.shape[1])
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(3 * k)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = ["ply", "format binary_little_endian 1.0", "comment VR Photo Studio", f"element vertex {n}"]
    header += [f"property float {p}" for p in props]
    header.append("end_header")
    dt = np.dtype([(p, "<f4") for p in props])
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        for a in range(0, n, _CHUNK):
            b = min(n, a + _CHUNK)
            rec = np.zeros(b - a, dtype=dt)
            m = sc.means[a:b]
            rec["x"], rec["y"], rec["z"] = m[:, 0], m[:, 1], m[:, 2]
            dc = (np.asarray(sc.colors[a:b], dtype=np.float64) - 0.5) / C0
            rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
            if k:
                rest = np.asarray(sc.sh_rest[a:b]).transpose(0, 2, 1).reshape(b - a, 3 * k)  # type: ignore[index]
                for i in range(3 * k):
                    rec[f"f_rest_{i}"] = rest[:, i]
            rec["opacity"] = _logit(np.asarray(sc.opacity[a:b]))
            ls = np.log(np.maximum(np.asarray(sc.scales[a:b], dtype=np.float64), 1e-30))
            rec["scale_0"], rec["scale_1"], rec["scale_2"] = ls[:, 0], ls[:, 1], ls[:, 2]
            q = _normalized_quats(sc.quats[a:b])
            rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            fh.write(rec.tobytes())
    _atomic_replace(tmp, path)


# --------------------------------------------------------------------------- .splat
def _read_dot_splat(path: Path) -> SplatCloud:
    size = path.stat().st_size
    if size % _SPLAT_DTYPE.itemsize:
        raise SplatError(f"{path.name} is not a valid .splat file (size is not a multiple of 32 bytes).")
    rec = np.fromfile(path, dtype=_SPLAT_DTYPE)
    q = (rec["rot"].astype(np.float32) - 128.0) / 128.0
    sc = SplatCloud(
        means=np.ascontiguousarray(rec["pos"], dtype=np.float32),
        scales=np.ascontiguousarray(np.abs(rec["scale"]), dtype=np.float32),
        quats=_normalized_quats(q),
        colors=rec["rgba"][:, :3].astype(np.float32) / 255.0,
        opacity=rec["rgba"][:, 3].astype(np.float32) / 255.0,
    )
    return _drop_non_finite(sc, path.name)


def _write_dot_splat(sc: SplatCloud, path: Path) -> None:
    n = len(sc)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        for a in range(0, n, _CHUNK):
            b = min(n, a + _CHUNK)
            rec = np.zeros(b - a, dtype=_SPLAT_DTYPE)
            rec["pos"] = sc.means[a:b]
            rec["scale"] = sc.scales[a:b]
            rgba = np.concatenate([np.asarray(sc.colors[a:b]), np.asarray(sc.opacity[a:b])[:, None]], axis=1)
            rec["rgba"] = np.clip(np.rint(np.clip(rgba, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
            q = _normalized_quats(sc.quats[a:b]).astype(np.float64)
            q[q[:, 0] < 0] *= -1.0
            rec["rot"] = np.clip(np.rint(q * 128.0 + 128.0), 0, 255).astype(np.uint8)
            fh.write(rec.tobytes())
    _atomic_replace(tmp, path)


# --------------------------------------------------------------------------- public I/O
def read_splats(path: str | Path) -> SplatCloud:
    """Read a 3DGS ``.ply`` or antimatter15 ``.splat`` file (coordinates verbatim)."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"Splat file not found: {path}")
    ext = path.suffix.lower()
    try:
        if ext == ".ply":
            sc = _read_ply(path)
        elif ext == ".splat":
            sc = _read_dot_splat(path)
        else:
            raise InputError(f"Unsupported splat format {ext or path.name!r}; use .ply or .splat.")
    except OSError as e:
        raise InputError(f"Cannot read {path.name}: {e}") from e
    except MemoryError as e:
        raise SplatError(f"{path.name} is too large to load into memory.") from e
    except (ValueError, KeyError, IndexError) as e:
        raise SplatError(f"{path.name} is not a valid splat file: {e}") from e
    log.info("Read %d splats from %s", len(sc), path.name)
    return sc


def write_splats(sc: SplatCloud, path: str | Path) -> Path:
    """Write ``sc`` as 3DGS ``.ply`` (keeps ``sh_rest``) or ``.splat`` (DC colour only)."""
    _validate(sc)
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in (".ply", ".splat"):
        raise InputError(f"Unsupported splat format {ext or path.name!r}; use .ply or .splat.")
    tmp = path.with_name(path.name + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if ext == ".ply":
            _write_ply(sc, path)
        else:
            _write_dot_splat(sc, path)
    except OSError as e:
        raise VRPhotoError(f"Cannot write {path.name}: {e}") from e
    finally:
        try:  # a failed write must not leave a half-written temporary file behind
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    log.info("Wrote %d splats to %s", len(sc), path.name)
    return path


# --------------------------------------------------------------------------- decimation
_KEY_LIMIT = float(2**62)


def _pack_keys(q: np.ndarray) -> np.ndarray | None:
    """Exact int64 key per voxel row of ``q`` (N x 3, >= 0), or None if it would overflow."""
    dims = q.max(axis=0) + 1 if len(q) else np.ones(3, np.int64)
    if float(dims[0]) * float(dims[1]) * float(dims[2]) >= _KEY_LIMIT:
        return None
    return (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]


def _voxel_keys(p: np.ndarray, lo: np.ndarray, v: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer voxel coordinates ``q`` of ``p`` and one int64 key per occupied voxel.

    Keys are packed densely when the voxel grid fits into 62 bits.  When a few far
    outliers blow up the grid (common with 3DGS floaters), each axis is first
    rank-compressed to its occupied voxel indices, which is still exact; only in the
    pathological case where even that overflows are the rows uniqued directly."""
    q = np.floor((p - lo) / v).astype(np.int64)
    np.maximum(q, 0, out=q)
    keys = _pack_keys(q)
    if keys is None:
        ranks = np.empty_like(q)
        for i in range(3):
            ranks[:, i] = np.unique(q[:, i], return_inverse=True)[1].reshape(-1)
        keys = _pack_keys(ranks)
        if keys is None:  # pragma: no cover - needs > 2**62 distinct rank triples
            keys = np.unique(ranks, axis=0, return_inverse=True)[1].reshape(-1).astype(np.int64)
    return keys, q


def _count_voxels(p: np.ndarray, lo: np.ndarray, v: float) -> int:
    keys = _voxel_keys(p, lo, v)[0]
    keys.sort()
    return int(1 + np.count_nonzero(np.diff(keys))) if len(keys) else 0


def _find_voxel(p: np.ndarray, max_n: int, cancel: CancelToken | None) -> float:
    """Voxel edge such that the number of occupied voxels is <= ``max_n`` (and close to it).

    Bracketing search in log space; new guesses assume ``count ~ voxel**-a`` with ``a``
    fitted to the bracket (``a = 2`` - a surface - before a bracket exists)."""
    lo, hi = p.min(axis=0), p.max(axis=0)
    ext = float(max((hi - lo).max(), 1e-9))
    v_min = ext / float(1 << 40)  # integer voxel coordinates stay far inside int64
    target = 0.95 * max_n
    v = max(ext / math.sqrt(max_n), v_min)
    fine_v = fine_c = None  # largest voxel seen with count > max_n
    ok_v = ok_c = None  # smallest voxel seen with count <= max_n
    prev = None
    for _ in range(30):
        _check(cancel)
        c = _count_voxels(p, lo, v)
        if c <= max_n:
            if ok_v is None or v < ok_v:
                ok_v, ok_c = v, c
            if c >= 0.9 * max_n:
                break
        elif fine_v is None or v > fine_v:
            fine_v, fine_c = v, c
        if fine_v is not None and ok_v is not None:
            if ok_v / fine_v < 1.002:
                break
            a = math.log(fine_c / max(ok_c, 1)) / math.log(ok_v / fine_v)
            nv = fine_v * (fine_c / target) ** (1.0 / a) if a > 0.1 else math.sqrt(fine_v * ok_v)
            nv = min(max(nv, fine_v * 1.0005), ok_v / 1.0005)
        elif ok_v is not None:
            nv = ok_v * (max(ok_c, 1) / target) ** 0.5
        else:
            nv = fine_v * max((fine_c / target) ** 0.5, 1.05)
        nv = max(nv, v_min)
        if prev is not None and nv == v:
            break
        prev, v = v, nv
    return float(ok_v) if ok_v is not None else ext * 1.01


def _merge_by_labels(
    sc: SplatCloud, labels: np.ndarray, m: int, centers: np.ndarray, cancel: CancelToken | None
) -> SplatCloud:
    """Moment-matching merge of splats sharing a label (``centers``: per-label reference point)."""
    n = len(sc)
    counts = np.bincount(labels, minlength=m)
    w = importance(sc) + 1e-30
    W = np.bincount(labels, weights=w, minlength=m)
    multi = counts > 1
    first = np.unique(labels, return_index=True)[1]  # labels are 0..m-1
    out_means = np.asarray(sc.means, dtype=np.float32)[first].copy()
    out_scales = np.asarray(sc.scales, dtype=np.float32)[first].copy()
    out_quats = np.asarray(sc.quats, dtype=np.float32)[first].copy()
    out_colors = np.asarray(sc.colors, dtype=np.float32)[first].copy()
    out_opacity = np.asarray(sc.opacity, dtype=np.float32)[first].copy()
    if multi.any():
        s1 = np.zeros((m, 3))
        s2 = np.zeros((m, 6))
        col = np.zeros((m, 3))
        logt = np.zeros(m)
        pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        for a in range(0, n, _CHUNK):
            _check(cancel)
            b = min(n, a + _CHUNK)
            lab = labels[a:b]
            wc = w[a:b]
            d = np.asarray(sc.means[a:b], dtype=np.float64) - centers[lab]
            rot = quat_to_rotmat(sc.quats[a:b])
            s2v = np.asarray(sc.scales[a:b], dtype=np.float64) ** 2
            for i in range(3):
                s1[:, i] += np.bincount(lab, weights=wc * d[:, i], minlength=m)
                col[:, i] += np.bincount(
                    lab, weights=wc * np.asarray(sc.colors[a:b, i], dtype=np.float64), minlength=m
                )
            for j, (r, c) in enumerate(pairs):
                cov_rc = np.einsum("nj,nj,nj->n", rot[:, r, :], rot[:, c, :], s2v)
                s2[:, j] += np.bincount(lab, weights=wc * (cov_rc + d[:, r] * d[:, c]), minlength=m)
            alpha = np.clip(np.asarray(sc.opacity[a:b], dtype=np.float64), 0.0, 1.0 - 1e-7)
            logt += np.bincount(lab, weights=np.log1p(-alpha), minlength=m)
        idx = np.nonzero(multi)[0]
        Wm = W[idx]
        mu = s1[idx] / Wm[:, None]
        sec = s2[idx] / Wm[:, None]
        cov = np.empty((len(idx), 3, 3))
        for j, (r, c) in enumerate(pairs):
            val = sec[:, j] - mu[:, r] * mu[:, c]
            cov[:, r, c] = val
            cov[:, c, r] = val
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 1e-20)
        det = np.linalg.det(evecs)
        evecs[det < 0, :, 2] *= -1.0
        out_means[idx] = (centers[idx] + mu).astype(np.float32)
        out_scales[idx] = np.sqrt(evals).astype(np.float32)
        out_quats[idx] = rotmat_to_quat(evecs).astype(np.float32)
        out_colors[idx] = np.clip(col[idx] / Wm[:, None], 0.0, 1.0).astype(np.float32)
        out_opacity[idx] = (1.0 - np.exp(logt[idx])).astype(np.float32)
    sh = None
    if sc.sh_rest is not None and not multi.any():
        sh = np.asarray(sc.sh_rest, dtype=np.float32)[first]
    return SplatCloud(out_means, out_scales, out_quats, out_colors, out_opacity, sh)


def decimate_splats(
    sc: SplatCloud,
    max_n: int = 750_000,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> SplatCloud:
    """Reduce ``sc`` to at most ``max_n`` splats (algorithm J); output sorted by importance.

    Splats with a non-finite position/scale, opacity < 0.04 and absurdly large ones
    (larger than 25 % of the robust scene extent *and* 20x the median size) are pruned
    first; if more than ``max_n`` remain they are merged per voxel by moment matching
    (see module docstring; far-away floaters do not coarsen the voxel grid).
    """
    rep = _as_reporter(progress)
    if not isinstance(max_n, (int, np.integer)) or int(max_n) < 1:
        raise InputError(f"max_n must be a positive integer (got {max_n!r}).")
    max_n = int(max_n)
    n = _validate(sc)
    if n == 0:
        return sc
    rep(0.02, "Pruning splats")
    op = np.asarray(sc.opacity, dtype=np.float32)
    ms = np.asarray(sc.scales, dtype=np.float32).max(axis=1)
    keep = _finite_mask(sc) & np.isfinite(op) & (op >= MIN_OPACITY)
    if keep.sum() >= 16:
        pts = np.asarray(sc.means)[keep]
        p2, p98 = np.percentile(pts, [2, 98], axis=0)
        ext = float(np.linalg.norm(p98 - p2))
        med = float(np.median(ms[keep]))
        keep &= ~((ms > _HUGE_FRAC * ext) & (ms > _HUGE_MEDIAN * med))
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        log.warning("Every splat is (nearly) transparent; nothing left after pruning")
    if len(idx) > max_n and sc.sh_rest is not None:
        # merging is unavoidable and always drops the higher SH bands: do not copy them
        sc = replace(sc, sh_rest=None)
    pruned = _subset(sc, idx) if len(idx) < n else sc
    log.info("Splat pruning: %d -> %d", n, len(pruned))
    _check(cancel)
    if len(pruned) <= max_n:
        rep(1.0, "Splats ready")
        return _sorted_by_importance(pruned)
    rep(0.1, "Choosing voxel size")
    p = np.asarray(pruned.means, dtype=np.float64)
    v = _find_voxel(p, max_n, cancel)
    rep(0.5, "Merging splats")
    lo = p.min(axis=0)
    keys, q = _voxel_keys(p, lo, v)
    uniq, labels = np.unique(keys, return_inverse=True)
    labels = labels.reshape(-1)
    m = len(uniq)
    centers = np.empty((m, 3))
    centers[labels] = lo + (q + 0.5) * v
    del q, keys
    _check(cancel)
    merged = _merge_by_labels(pruned, labels, m, centers, cancel)
    log.info("Splat decimation: %d -> %d (voxel %.4g m)", len(pruned), len(merged), v)
    rep(1.0, "Splats ready")
    return _sorted_by_importance(merged)


# --------------------------------------------------------------------------- cropping / orientation
def crop_splats(sc: SplatCloud, *, radius_pct: float | None = 98.0, aabb: Any | None = None) -> SplatCloud:
    """Remove outliers: keep splats inside ``aabb`` (``[[xmin, ymin, zmin], [xmax, ymax,
    zmax]]``) if given, then within the ``radius_pct`` percentile of the distance from
    the median position (``None`` or >= 100 disables the radius test).  Splats with a
    non-finite position or scale are always removed."""
    _validate(sc)
    if radius_pct is not None and not (isinstance(radius_pct, (int, float, np.number)) and radius_pct > 0):
        raise InputError(f"radius_pct must be in (0, 100] (got {radius_pct!r}).")
    keep = _finite_mask(sc)
    p = np.asarray(sc.means, dtype=np.float64)
    if aabb is not None:
        box = np.asarray(aabb, dtype=np.float64)
        if box.shape != (2, 3) or not np.isfinite(box).all():
            raise InputError("aabb must be [[xmin, ymin, zmin], [xmax, ymax, zmax]].")
        keep &= np.all(p >= box[0], axis=1) & np.all(p <= box[1], axis=1)
    if radius_pct is not None and radius_pct < 100 and keep.any():
        c = np.median(p[keep], axis=0)
        d = np.linalg.norm(p - c, axis=1)
        keep &= d <= np.percentile(d[keep], radius_pct)
    return _subset(sc, np.nonzero(keep)[0])


def _rotation_to_y(up: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector ``up`` onto +y."""
    u = up / np.linalg.norm(up)
    y = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(u, y))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])  # 180 deg about x
    axis = np.cross(u, y)
    s = np.linalg.norm(axis)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * (kx @ kx)


def _ransac_plane(
    pts: np.ndarray, tol: float, rng: np.random.Generator, iters: int = 256
) -> tuple[np.ndarray, float, float]:
    """Best plane (unit normal, offset, inlier fraction) through ``pts``."""
    n = len(pts)
    if n < 3:
        return np.array([0.0, 1.0, 0.0]), 0.0, 0.0
    tri = rng.integers(0, n, size=(iters, 3))
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    if not ok.any():
        return np.array([0.0, 1.0, 0.0]), 0.0, 0.0
    nrm = nrm[ok] / ln[ok, None]
    off = np.einsum("ij,ij->i", nrm, a[ok])
    p32, n32, o32 = pts.astype(np.float32), nrm.T.astype(np.float32), off.astype(np.float32)
    if n <= _RANSAC_CHUNK:
        dist = np.abs(p32 @ n32 - o32)
        inl = (dist < tol).sum(axis=0)
        best = int(np.argmax(inl))
        mask = dist[:, best] < tol
    else:  # the N x iters distance matrix of a big cloud would not fit in memory
        inl = np.zeros(len(o32), np.int64)
        for s in range(0, n, _RANSAC_CHUNK):
            inl += (np.abs(p32[s : s + _RANSAC_CHUNK] @ n32 - o32) < tol).sum(axis=0)
        best = int(np.argmax(inl))
        mask = np.abs(p32 @ n32[:, best] - o32[best]) < tol
    q = pts[mask]
    centroid = q.mean(axis=0)
    normal = np.linalg.svd(q - centroid, full_matrices=False)[2][-1]
    if np.dot(normal, nrm[best]) < 0:
        normal = -normal
    return normal, float(np.dot(normal, centroid)), float(mask.mean())


def _estimate_floor(p: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Up vector (+ floor offset along it, or None) by PCA + RANSAC on the lowest 10 %."""
    rng = np.random.default_rng(0)
    if len(p) > 200_000:
        p = p[rng.choice(len(p), 200_000, replace=False)]
    c = p.mean(axis=0)
    evecs = np.linalg.eigh(np.cov((p - c).T))[1]
    axis = evecs[:, int(np.argmax(np.abs(evecs[1, :])))]  # PCA axis closest to +-y
    if axis[1] < 0:
        axis = -axis
    lo, hi = np.percentile(p, [2, 98], axis=0)
    tol = max(0.005 * float(np.linalg.norm(hi - lo)), 1e-6)
    best = (axis, None, 0.0)
    scores = []
    for sgn in (1.0, -1.0):
        d = sgn * axis
        h = p @ d
        low = p[h <= np.percentile(h, 10)]
        if len(low) > 20_000:
            low = low[rng.choice(len(low), 20_000, replace=False)]
        normal, off, frac = _ransac_plane(low, tol, rng)
        if np.dot(normal, d) < 0:  # orient the floor normal away from the floor side
            normal, off = -normal, -off
        if np.dot(normal, d) < math.cos(math.radians(30)):
            frac = 0.0
        scores.append((frac, sgn, normal, off))
    scores.sort(key=lambda s: s[0] + (0.1 if s[1] > 0 else 0.0), reverse=True)
    frac, sgn, normal, off = scores[0]
    if frac >= 0.25:
        best = (normal, off, frac)
    return best[0], best[1]


def normalize_orientation(sc: SplatCloud, up: np.ndarray | None = None) -> SplatCloud:
    """Rotate so ``up`` (or an estimated up) becomes +y, put the floor at ``y = 0`` and
    centre the cloud on the y axis (median x/z).

    Without ``up`` the vertical is the PCA axis closest to +-y (input assumed roughly
    upright up to sign), refined by a RANSAC plane fit on the lowest 10 % of points at
    either end; the end with the better floor plane (>= 25 % inliers, normal within 30
    degrees of the axis) wins and the floor height is that plane.  Otherwise the floor is
    the 0.5th height percentile.  Splat orientations and higher SH bands are rotated too.
    """
    _validate(sc)
    sc = _drop_non_finite(sc, "normalize_orientation")
    n = len(sc)
    if n == 0:
        return sc
    p = np.asarray(sc.means, dtype=np.float64)
    floor: float | None = None
    if up is not None:
        u = np.asarray(up, dtype=np.float64).reshape(-1)
        if u.shape != (3,) or not np.isfinite(u).all() or np.linalg.norm(u) < 1e-12:
            raise InputError("up must be a non-zero 3-vector.")
        u = u / np.linalg.norm(u)
    elif n >= 10:
        u, floor = _estimate_floor(p)
    else:  # too few points for PCA / RANSAC: keep the frame, only re-floor and centre
        u = np.array([0.0, 1.0, 0.0])
    R = _rotation_to_y(u)
    pr = p @ R.T
    if floor is None:
        floor = float(np.percentile(pr[:, 1], 0.5))
    shift = np.array([np.median(pr[:, 0]), floor, np.median(pr[:, 2])])
    pr -= shift
    qr = rotmat_to_quat(R[None])[0]
    quats = quat_mul(qr[None, :], np.asarray(sc.quats, dtype=np.float64))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return replace(
        sc,
        means=pr.astype(np.float32),
        quats=quats.astype(np.float32),
        sh_rest=_rotate_sh(sc.sh_rest, R),
        scales=np.asarray(sc.scales, dtype=np.float32),
        colors=np.asarray(sc.colors, dtype=np.float32),
        opacity=np.asarray(sc.opacity, dtype=np.float32),
    )


def _rotate_sh(sh: np.ndarray | None, R: np.ndarray) -> np.ndarray | None:
    """``sh_rest`` (N x K x 3) for a world rotation ``R``; None (with a warning) when ``K``
    is not a complete set of bands."""
    if sh is None or np.allclose(R, np.eye(3)):
        return sh
    M = sh_rotation_matrix(R, int(sh.shape[1]))
    if M is None:
        log.warning("Cannot rotate %d SH coefficients; dropping view-dependent colour", sh.shape[1])
        return None
    # chunked float32 matmul: no float64 copies of a (possibly 540 MB) SH array
    src = np.asarray(sh)
    m32 = M.astype(np.float32)
    out = np.empty(src.shape, np.float32)
    for a in range(0, len(src), _CHUNK):
        b = min(len(src), a + _CHUNK)
        out[a:b] = np.matmul(m32, src[a:b].astype(np.float32, copy=False))
    return out


# --------------------------------------------------------------------------- transforms / combining
def _rotation_arg(R: Any) -> np.ndarray:
    r = np.asarray(R, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise InputError("R must be a finite 3x3 rotation matrix.")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-4) or np.linalg.det(r) <= 0:
        # a reflection (det -1) has no quaternion, so splat orientations could not follow it
        raise InputError("R must be a proper rotation (orthonormal, determinant +1).")
    return r


def _positive(x: Any, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = float("nan")
    if not (math.isfinite(v) and v > 0):
        raise InputError(f"{what} must be a positive number (got {x!r}).")
    return v


def transform_splats(sc: SplatCloud, R: np.ndarray, t: np.ndarray, s: float = 1.0) -> SplatCloud:
    """Similarity transform ``P' = s * R @ P + t`` of a whole cloud.

    Orientations follow the rotation (quaternions left-multiplied by ``R``'s quaternion and
    normalised), sizes scale by ``s`` and the higher SH bands are rotated like
    :func:`normalize_orientation` does (dropped with a warning when their count is not a
    complete set of bands).  Colours and opacities are unchanged.  ``R`` must be a proper
    rotation (a reflection cannot be expressed with quaternions) and ``s > 0``; raises
    :class:`InputError` otherwise.  Returns a new float32 cloud (unchanged arrays may be
    shared with ``sc``)."""
    _validate(sc)
    r = _rotation_arg(R)
    tv = np.asarray(t, dtype=np.float64).reshape(-1)
    if tv.shape != (3,) or not np.isfinite(tv).all():
        raise InputError("t must be a finite 3-vector.")
    k = _positive(s, "The scale s")
    p = np.asarray(sc.means, dtype=np.float64)
    means = (k * (p @ r.T) + tv).astype(np.float32)
    qr = rotmat_to_quat(r[None])[0]
    quats = quat_mul(qr[None, :], np.asarray(sc.quats, dtype=np.float64))
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    return SplatCloud(
        means=means,
        scales=(np.asarray(sc.scales, dtype=np.float64) * k).astype(np.float32),
        quats=quats.astype(np.float32),
        colors=np.asarray(sc.colors, dtype=np.float32),
        opacity=np.asarray(sc.opacity, dtype=np.float32),
        sh_rest=_rotate_sh(sc.sh_rest, r),
    )


def scale_splats(sc: SplatCloud, k: float) -> SplatCloud:
    """Uniform scale about the origin: positions and sizes times ``k`` (> 0); orientations,
    colours and SH are unchanged.  Raises :class:`InputError` for a bad ``k``."""
    _validate(sc)
    f = _positive(k, "The scale factor")
    return replace(
        sc,
        means=(np.asarray(sc.means, dtype=np.float64) * f).astype(np.float32),
        scales=(np.asarray(sc.scales, dtype=np.float64) * f).astype(np.float32),
        quats=np.asarray(sc.quats, dtype=np.float32),
        colors=np.asarray(sc.colors, dtype=np.float32),
        opacity=np.asarray(sc.opacity, dtype=np.float32),
    )


def concat_splats(clouds: Any) -> SplatCloud:
    """Concatenate clouds in order (float32).

    ``sh_rest`` is kept only when every non-empty cloud has it with the same number of
    coefficients; otherwise the higher SH bands are dropped (DC colour stays) and that is
    logged.  An empty sequence raises :class:`InputError`."""
    parts = list(clouds)
    if not parts:
        raise InputError("There are no splat clouds to combine.")
    for c in parts:
        _validate(c)
    # empty clouds carry no SH information, so they must not force the SH to be dropped
    decide = [c for c in parts if len(c)] or parts
    ks = {None if c.sh_rest is None else int(c.sh_rest.shape[1]) for c in decide}
    k = next(iter(ks)) if len(ks) == 1 else None
    if k is None and any(c.sh_rest is not None for c in parts):
        log.info(
            "concat_splats: SH coefficient counts differ (%s); keeping DC colour only", sorted(map(str, ks))
        )

    def cat(get: Any, width: tuple[int, ...]) -> np.ndarray:
        arrs = [np.asarray(get(c), dtype=np.float32).reshape((len(c), *width)) for c in parts]
        return np.ascontiguousarray(np.concatenate(arrs, axis=0), dtype=np.float32)

    sh = None
    if k is not None:
        sh = cat(lambda c: c.sh_rest if c.sh_rest is not None else np.zeros((0, k, 3), np.float32), (k, 3))
    return SplatCloud(
        means=cat(lambda c: c.means, (3,)),
        scales=cat(lambda c: c.scales, (3,)),
        quats=cat(lambda c: c.quats, (4,)),
        colors=cat(lambda c: c.colors, (3,)),
        opacity=cat(lambda c: c.opacity, ()),
        sh_rest=sh,
    )


def subset_splats(sc: SplatCloud, idx: Any) -> SplatCloud:
    """The splats ``idx`` (integer indices, in that order, or a boolean mask of length
    ``len(sc)``) as a new contiguous float32 cloud (``sh_rest`` kept)."""
    sel = np.asarray(idx)
    if sel.dtype == bool:
        if sel.shape != (len(sc),):
            raise InputError(f"The selection mask must have {len(sc)} entries (got shape {sel.shape}).")
        sel = np.flatnonzero(sel)
    elif sel.size == 0:
        sel = np.zeros(0, np.int64)
    elif sel.dtype.kind not in "iu":
        raise InputError("Splat indices must be integers or a boolean mask.")
    return _subset(sc, sel)


def fit_plane_ransac(
    pts: np.ndarray, tol: float, *, seed: int = 0, iters: int = 256
) -> tuple[np.ndarray, float, float]:
    """RANSAC plane through ``pts`` (N x 3): ``iters`` random triangles, the one with most
    points within ``tol`` wins, then a least-squares refit on its inliers.

    Returns ``(normal, offset, inlier_frac)`` with ``normal . p == offset`` on the plane;
    the sign of ``normal`` is arbitrary (orient it yourself).  Rows with non-finite values
    are ignored; fewer than 3 points give ``((0, 1, 0), 0.0, 0.0)``.  Deterministic for a
    given ``seed``.  Memory is chunked, but runtime grows with ``N * iters``, so subsample
    very large inputs."""
    p = np.asarray(pts, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise InputError(f"Plane fitting needs N x 3 points (got shape {p.shape}).")
    tol_v = _positive(tol, "The plane tolerance")
    if not (isinstance(iters, (int, np.integer)) and int(iters) >= 1):
        raise InputError(f"iters must be a positive integer (got {iters!r}).")
    p = p[np.isfinite(p).all(axis=1)]
    return _ransac_plane(p, tol_v, np.random.default_rng(seed), int(iters))


# --------------------------------------------------------------------------- mesh -> splats
def _sample_texture(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear texture lookup (clamp) at glTF UVs -> float32 (N, C) in [0, 1]."""
    t = np.asarray(tex)
    h, w = t.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip(uv[:, 1] * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    tf = t.reshape(h, w, -1)
    c = (
        tf[y0, x0] * (1 - fx) * (1 - fy)
        + tf[y0, x1] * fx * (1 - fy)
        + tf[y1, x0] * (1 - fx) * fy
        + tf[y1, x1] * fx * fy
    )
    return (c / 255.0 if t.dtype == np.uint8 else c).astype(np.float32)


def splats_from_mesh(mesh: Mesh, *, scale_m: float | None = None, max_splats: int = 1_000_000) -> SplatCloud:
    """Sample a textured mesh into flat Gaussian splats (discs tangent to the surface).

    Splats are spaced ``scale_m / 0.6`` apart (area-proportional, stratified per face)
    with standard deviation ``scale_m`` in-plane and ``scale_m / 10`` along the normal;
    colour (and opacity, for MASK/BLEND materials) comes from the texture; MASK samples
    below the cutoff are dropped.  ``scale_m=None`` picks the spacing so the mesh yields
    about two splats per face (at most ``max_splats``); an explicit ``scale_m`` that
    would exceed ``max_splats`` is enlarged so the surface stays covered.  Faces with
    non-finite vertices are skipped.  The output is in the mesh's (glTF) frame.
    """
    if not (isinstance(max_splats, (int, np.integer)) and int(max_splats) >= 1):
        raise InputError(f"max_splats must be a positive integer (got {max_splats!r}).")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    uv = np.asarray(mesh.uvs, dtype=np.float64)
    if len(f) == 0:
        raise InputError(f"Mesh '{mesh.name}' has no faces to sample.")
    if v.ndim != 2 or v.shape[1] != 3 or uv.shape != (len(v), 2):
        raise InputError(f"Mesh '{mesh.name}' needs Nx3 vertices and matching Nx2 UVs.")
    if f.min() < 0 or f.max() >= len(v):
        raise InputError(f"Mesh '{mesh.name}' has face indices outside its vertex array.")
    ok_v = np.isfinite(v).all(axis=1) & np.isfinite(uv).all(axis=1)
    f = f[ok_v[f].all(axis=1)]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    area = 0.5 * area2
    total = float(area.sum())
    if not total > 0:
        raise InputError(f"Mesh '{mesh.name}' has zero surface area.")
    if scale_m is None:
        n_target = min(int(max_splats), max(1, 2 * len(f)))
        spacing = math.sqrt(total / n_target)
        scale = 0.6 * spacing
    else:
        if not (math.isfinite(scale_m) and scale_m > 0):
            raise InputError(f"scale_m must be a positive size in metres (got {scale_m!r}).")
        scale = float(scale_m)
        spacing = scale / 0.6
        if total / spacing**2 > max_splats:
            spacing = math.sqrt(total / max_splats)
            scale = 0.6 * spacing  # keep the surface covered with fewer, larger splats
            log.info("splats_from_mesh: splats enlarged to %.4g m to stay under %d splats", scale, max_splats)
    rng = np.random.default_rng(0)
    expect = area / spacing**2
    counts = np.floor(expect).astype(np.int64)
    counts += rng.random(len(f)) < (expect - counts)
    fi = np.repeat(np.arange(len(f)), counts)
    if len(fi) == 0:
        fi = np.array([int(np.argmax(area))])
    r1 = np.sqrt(rng.random(len(fi)))
    r2 = rng.random(len(fi))
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
    pos = a[fi] * w0[:, None] + b[fi] * w1[:, None] + c[fi] * w2[:, None]
    tuv = uv[f[fi, 0]] * w0[:, None] + uv[f[fi, 1]] * w1[:, None] + uv[f[fi, 2]] * w2[:, None]
    tex = _sample_texture(np.asarray(mesh.texture), tuv)
    colors = tex[:, :3] if tex.shape[1] >= 3 else np.repeat(tex[:, :1], 3, axis=1)
    opacity = np.ones(len(fi), np.float32)
    if tex.shape[1] == 4 and mesh.alpha_mode != "OPAQUE":
        if mesh.alpha_mode == "MASK":
            ok = tex[:, 3] >= mesh.alpha_cutoff
            pos, colors, fi = pos[ok], colors[ok], fi[ok]
            opacity = np.ones(len(fi), np.float32)
        else:
            opacity = tex[:, 3].astype(np.float32)
    nrm = cr[fi] / np.maximum(area2[fi], 1e-30)[:, None]
    t1 = b[fi] - a[fi]
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-30)
    t2 = np.cross(nrm, t1)
    rot = np.stack([t1, t2, nrm], axis=2)  # columns = local axes
    quats = rotmat_to_quat(rot)
    scales = np.tile(np.array([scale, scale, scale * 0.1], np.float32), (len(fi), 1))
    return SplatCloud(
        means=pos.astype(np.float32),
        scales=scales,
        quats=quats.astype(np.float32),
        colors=np.clip(colors, 0.0, 1.0).astype(np.float32),
        opacity=np.clip(opacity, 0.0, 1.0).astype(np.float32),
    )


__all__ = [
    "read_splats",
    "write_splats",
    "decimate_splats",
    "crop_splats",
    "normalize_orientation",
    "transform_splats",
    "scale_splats",
    "concat_splats",
    "subset_splats",
    "fit_plane_ransac",
    "splats_from_mesh",
    "importance",
    "splat_covariances",
    "quat_to_rotmat",
    "rotmat_to_quat",
    "quat_mul",
    "sh_rotation_matrix",
    "SplatError",
    "C0",
    "MIN_OPACITY",
]
