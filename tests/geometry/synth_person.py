"""Analytic synthetic subject for the SHARP 360 tests (docs/design/SHARP360.md section 5.1).

A person made of ellipsoids stands on a floor in front of a back wall.  Everything is
ray-cast in closed form (vectorised numpy, no renderer), so the tests get exact ground
truth: the photo and its matte, a SHARP-like Gaussian cloud with labelled layers and
"cards", the cleaned front (:class:`FrontModel`) and the upright frame (:class:`Canon`) as
SHARP 360 should find them, and orthographic views from any azimuth.

Frames (metres):

* **C** - the photo camera / SHARP's frame: OpenCV axes (x right, y down, z forward),
  camera at the origin; continuous pixel coordinates ``u = f*X/Z + W/2``,
  ``v = f*Y/Z + H/2``, pixel index ``floor(u)`` (pixel centres at ``i + 0.5``).
* **W** - gravity-aligned world at the camera: ``x_W`` right, ``y_W`` down, ``z_W`` the
  camera's horizontal viewing direction; ``P_C = R_cw @ P_W``.  The camera looks down by
  ``pitch_deg`` (its forward axis is ``(0, sin p, cos p)`` in W).  Floor ``y_W =
  cam_height``, back wall ``z_W = wall_z`` (both infinite planes).
* **B** - the body: origin on the floor under the body axis, i.e. ``(0, cam_height,
  distance)`` in W; ``x_B`` = the body's own LEFT (image right when it faces the camera),
  ``y_B`` down, ``z_B`` = the body's back direction (away from the camera at yaw 0).
  ``R_wb`` turns the body about the vertical by ``yaw_deg``; positive yaw turns its front
  toward ``+x_W`` (the camera's right).
* **S** (ground truth :class:`Canon`) - ``x^, y^, z^`` = ``x_W, y_W, z_W`` expressed in C;
  the origin ``c`` is on the vertical through the body's bounding-box centre.

The body (heights above the floor): torso, pelvis, shoulders, neck and head (top at
1.70 m), two legs (1 cm above the floor), the LEFT arm raised sideways 20 deg above the
horizontal (tip at ``x_B ~ +0.77``), the RIGHT arm hanging with its forearm across the belly
3-9 cm in front of the torso (a torso-behind-arm disocclusion region), and a bag at the
RIGHT hip (``x_B ~ -0.34 .. -0.18``, slightly behind the body) on a strap over the right
shoulder.  It is asymmetric on purpose: a mirrored back puts the raised arm and the bag on
the wrong sides.

Colours (flat, no shading) by the outward normal in B, *dominant horizontal axis*: if
``|n_z| >= |n_x|`` the front (``n_z < 0``) is red and the back blue, else the body-left
(``n_x > 0``) is green and the body-right yellow.  Floor brown, wall light grey, nothing
(sky) black.

Only the ``splat_fusion`` dataclasses are used (constructed directly); all projection math
is local, so the fixture does not depend on the code under test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from vrphoto.core.fill import push_pull_fill
from vrphoto.core.types import SplatCloud
from vrphoto.geometry.splat import rotmat_to_quat
from vrphoto.geometry.splat_fusion import Canon, FrontModel, MouldGrid, SharpCam

# --------------------------------------------------------------------------- palette / labels
FRONT_RGB = np.array([220, 40, 40], np.uint8)  # red: faces the camera (yaw 0)
BACK_RGB = np.array([40, 70, 220], np.uint8)  # blue
LEFT_RGB = np.array([40, 190, 70], np.uint8)  # green: the body's own left (+x_B)
RIGHT_RGB = np.array([235, 205, 40], np.uint8)  # yellow: the body's own right (-x_B)
FLOOR_RGB = np.array([120, 95, 70], np.uint8)
WALL_RGB = np.array([200, 200, 200], np.uint8)
SKY_RGB = np.array([0, 0, 0], np.uint8)  # photo pixels where a ray hits nothing
VIEW_BG_RGB = np.array([128, 128, 128], np.uint8)  # ortho_view background
PALETTE: dict[str, np.ndarray] = {
    "front": FRONT_RGB,
    "back": BACK_RGB,
    "left": LEFT_RGB,
    "right": RIGHT_RGB,
    "floor": FLOOR_RGB,
    "wall": WALL_RGB,
    "sky": SKY_RGB,
    "view_bg": VIEW_BG_RGB,
}
SIDES = ("front", "back", "left", "right")

# per-Gaussian kinds of :func:`sharp_like_scene`
KIND_BODY = 0  # layer 0: visible body surface
KIND_FLOOR = 1  # layer 0: visible floor
KIND_WALL = 2  # layer 0: visible wall
KIND_CARD = 3  # layer 0: ray-aligned "card" at a body silhouette depth jump (replaces KIND_BODY)
KIND_HIDDEN_BODY = 4  # layer 1: body hidden behind another body part (disocclusion fill: keep)
KIND_HIDDEN_BG = 5  # layer 1: floor / wall hidden behind the body ("wall blobs": remove)
KIND_BG_LAYER1 = 6  # layer 1: 3 cm behind a visible floor / wall point (background: remove)
KIND_NAMES = ("body", "floor", "wall", "card", "hidden_body", "hidden_bg", "bg_layer1")
OPACITY = {
    KIND_BODY: 0.95,
    KIND_FLOOR: 0.95,
    KIND_WALL: 0.95,
    KIND_CARD: 0.9,
    KIND_HIDDEN_BODY: 0.85,
    KIND_HIDDEN_BG: 0.85,
    KIND_BG_LAYER1: 0.5,
}
_OPACITY_LUT = np.array([OPACITY[k] for k in range(len(KIND_NAMES))])

#: Default photo: portrait 120 x 160 px, 40 deg horizontal FOV (f = 164.8 px): the 1.69 m
#: body at 2.5 m is ~111 px tall (rows ~24-136) and the floor shows below its feet.
PHOTO_W, PHOTO_H, PHOTO_HFOV = 120, 160, 40.0
BODY_HEIGHT = 1.69  # head top 1.70 m, feet 1 cm above the floor


def _b(x: float, h: float, z: float) -> np.ndarray:
    """B-frame point from (x, height above the floor, z)."""
    return np.array([x, -h, z], np.float64)


def _frame_along(axis: np.ndarray) -> np.ndarray:
    """Rotation whose first column is the unit ``axis``."""
    a = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b = ref - (ref @ a) * a
    b /= np.linalg.norm(b)
    return np.column_stack([a, b, np.cross(a, b)])


def _seg(name: str, p0: tuple, p1: tuple, r: float, extra: float) -> tuple:
    """Ellipsoid along the segment ``p0 -> p1`` (``(x, h, z)``), radius ``r``, its long
    semi-axis ``extra`` longer than half the segment (so joints overlap)."""
    a, b = _b(*p0), _b(*p1)
    half = 0.5 * float(np.linalg.norm(b - a))
    return name, 0.5 * (a + b), (half + extra, r, r), _frame_along(b - a)


def _parts() -> list[tuple]:
    eye = np.eye(3)
    a20 = math.radians(20.0)
    sh_l = (0.17, 1.38, 0.0)
    hand_l = (0.17 + 0.62 * math.cos(a20), 1.38 + 0.62 * math.sin(a20), 0.0)
    return [
        ("torso", _b(0, 1.12, 0), (0.18, 0.36, 0.11), eye),
        ("pelvis", _b(0, 0.88, 0), (0.16, 0.12, 0.10), eye),
        ("shoulders", _b(0, 1.36, 0), (0.21, 0.07, 0.09), eye),
        ("neck", _b(0, 1.47, 0), (0.05, 0.06, 0.05), eye),
        ("head", _b(0, 1.585, 0), (0.09, 0.115, 0.10), eye),
        ("leg_left", _b(0.085, 0.43, 0), (0.07, 0.42, 0.075), eye),
        ("leg_right", _b(-0.085, 0.43, 0), (0.07, 0.42, 0.075), eye),
        _seg("arm_left", sh_l, hand_l, 0.05, 0.02),
        _seg("arm_right_upper", (-0.19, 1.37, -0.01), (-0.22, 1.10, -0.05), 0.05, 0.03),
        _seg("arm_right_fore", (-0.22, 1.10, -0.05), (-0.02, 1.00, -0.24), 0.045, 0.02),
        ("bag", _b(-0.26, 1.00, 0.07), (0.08, 0.14, 0.13), eye),
        _seg("strap", (-0.24, 1.13, 0.08), (-0.12, 1.43, 0.0), 0.018, 0.0),
    ]


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# --------------------------------------------------------------------------- body
@dataclass
class Body:
    """The scene: ellipsoid body parts (in C) plus the floor and back-wall planes.

    Build it with :func:`make_body`.  Ray methods (:meth:`raycast`, :meth:`raycast_part`,
    :meth:`raycast_background`, :meth:`raycast_scene`, :meth:`intervals`,
    :meth:`first_two`) take ``origins`` (``Mx3`` or one ``3``-vector) and ``dirs`` (``Mx3``,
    need not be unit) in C and return the ray parameter ``t`` with the hit at ``o + t*d``
    (with ``z = 1`` directions from the camera ``t`` is the depth z, with unit directions
    the distance).  Body part ``k`` is the ellipsoid ``|diag(1/radii[k]) rot[k]^T (P -
    centers[k])| = 1``; ``names[k]`` is its name (:meth:`part` gives the index)."""

    names: tuple[str, ...]
    centers: np.ndarray  # (K, 3) ellipsoid centres in C
    rot: np.ndarray  # (K, 3, 3) columns = the ellipsoid's local axes in C
    radii: np.ndarray  # (K, 3) semi-axes (m)
    R_cw: np.ndarray  # (3, 3) W -> C (columns = x_W, y_W, z_W in C)
    R_cb: np.ndarray  # (3, 3) B -> C (columns = x_B, y_B, z_B in C)
    origin: np.ndarray  # (3,) B origin (floor point under the body axis) in C
    distance: float  # horizontal distance camera -> body axis
    pitch_deg: float  # camera looks down by this much
    yaw_deg: float
    cam_height: float  # camera height above the floor
    wall_z: float  # wall distance along z_W
    lateral: float = 0.0  # body axis offset along x_W
    _A: np.ndarray = field(init=False, repr=False)  # (K,3,3) C -> unit-sphere coordinates
    _G: np.ndarray = field(init=False, repr=False)  # (K,3,3) normal matrices

    def __post_init__(self) -> None:
        inv = 1.0 / self.radii
        self._A = inv[:, :, None] * np.swapaxes(self.rot, 1, 2)
        self._G = (self.rot * (inv**2)[:, None, :]) @ np.swapaxes(self.rot, 1, 2)

    # ---- frames
    @property
    def up_C(self) -> np.ndarray:
        """World up in C (``-y_W``)."""
        return -self.R_cw[:, 1].copy()

    @property
    def floor_normal(self) -> np.ndarray:
        return self.up_C

    @property
    def wall_normal(self) -> np.ndarray:
        """Wall normal in C, facing the camera (``-z_W``)."""
        return -self.R_cw[:, 2].copy()

    def part(self, name: str) -> int:
        return self.names.index(name)

    def to_B(self, P: np.ndarray) -> np.ndarray:
        """``...x3`` points from C to the body frame."""
        return (np.asarray(P, np.float64) - self.origin) @ self.R_cb

    def from_B(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P, np.float64) @ self.R_cb.T + self.origin

    def extent(self, axis: np.ndarray) -> tuple[float, float]:
        """Exact ``(min, max)`` of ``axis . P`` over the body (union of ellipsoids)."""
        a = np.asarray(axis, np.float64)
        a = a / np.linalg.norm(a)
        local = np.einsum("kji,j->ki", self.rot, a) * self.radii  # rot^T a scaled by the radii
        half = np.linalg.norm(local, axis=1)
        c = self.centers @ a
        return float((c - half).min()), float((c + half).max())

    def colour_of(self, normal: np.ndarray) -> np.ndarray:
        """Side colour (uint8 ``Mx3``) of outward normals in C (see module docstring)."""
        nb = np.asarray(normal, np.float64) @ self.R_cb
        fb = np.abs(nb[:, 2]) >= np.abs(nb[:, 0])
        out = np.where((nb[:, 0] > 0)[:, None], LEFT_RGB, RIGHT_RGB)
        out = np.where((fb & (nb[:, 2] < 0))[:, None], FRONT_RGB, out)
        out = np.where((fb & (nb[:, 2] >= 0))[:, None], BACK_RGB, out)
        return out.astype(np.uint8)

    def side_of(self, normal: np.ndarray) -> np.ndarray:
        """Side name (``"front" / "back" / "left" / "right"``) of outward normals in C."""
        return classify_rgb(self.colour_of(normal))

    def normal_at(self, P: np.ndarray, part: np.ndarray) -> np.ndarray:
        """Unit outward normals (C) at surface points ``P`` of parts ``part``."""
        g = np.einsum("mij,mj->mi", self._G[part], np.asarray(P, np.float64) - self.centers[part])
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300)

    # ---- ray casting
    @staticmethod
    def _rays(origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(dirs, np.float64).reshape(-1, 3)
        o = np.broadcast_to(np.asarray(origins, np.float64), d.shape)
        return o, d

    def intervals(self, origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per ray and part (``M x K``) the entry / exit parameters ``t_in <= t_out`` of the
        ellipsoid in front of the origin (``t_in`` clamped at 0), ``inf`` where missed."""
        o, d = self._rays(origins, dirs)
        shared = np.ndim(origins) == 1  # one origin (the camera) for every ray
        o0 = np.asarray(origins, np.float64)
        m, k = len(d), len(self.radii)
        t_in = np.full((m, k), np.inf)
        t_out = np.full((m, k), np.inf)
        dl2 = np.maximum((d * d).sum(1), 1e-300)
        for j in range(k):  # one part at a time, only on the rays near its bounding sphere
            rel = self.centers[j] - (o0 if shared else o)
            along = d @ rel if shared else (d * rel).sum(1)
            rel2 = rel @ rel if shared else (rel * rel).sum(1)
            rb2 = (1.001 * float(self.radii[j].max())) ** 2
            idx = np.flatnonzero((rel2 - along * along / dl2 <= rb2) & ((along > 0) | (rel2 <= rb2)))
            if not len(idx):
                continue
            A = self._A[j]
            op = ((o0 if shared else o[idx]) - self.centers[j]) @ A.T
            dp = d[idx] @ A.T
            a = (dp * dp).sum(1)
            b = 2.0 * (dp @ op if shared else (op * dp).sum(1))
            c = (op @ op if shared else (op * op).sum(1)) - 1.0
            disc = b * b - 4.0 * a * c
            ok = (disc > 0) & (a > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                sq = np.sqrt(np.where(ok, disc, 0.0))
                t0 = (-b - sq) / (2.0 * a)
                t1 = (-b + sq) / (2.0 * a)
            ok &= t1 > 0
            t_in[idx, j] = np.where(ok, np.maximum(t0, 0.0), np.inf)
            t_out[idx, j] = np.where(ok, t1, np.inf)
        return t_in, t_out

    def first_two(self, origins: np.ndarray, dirs: np.ndarray) -> dict[str, np.ndarray]:
        """The first two surfaces of the body (union of ellipsoids) along each ray:
        ``s0`` / ``part0`` = first entry, ``e0`` = where the ray leaves the body again
        (overlapping parts merged), ``s1`` / ``part1`` = the next entry after that (a body
        part hidden behind the first one), ``inf`` / ``-1`` where none."""
        t_in, t_out = self.intervals(origins, dirs)
        order = np.argsort(t_in, axis=1, kind="stable")
        ti = np.take_along_axis(t_in, order, 1)
        to = np.take_along_axis(t_out, order, 1)
        m = len(ti)
        s0, e0 = ti[:, 0].copy(), to[:, 0].copy()
        p0 = np.where(np.isfinite(s0), order[:, 0], -1)
        s1 = np.full(m, np.inf)
        p1 = np.full(m, -1, np.int64)
        open_ = np.isfinite(s0)
        for j in range(1, ti.shape[1]):  # sweep the sorted intervals (K is ~12)
            valid = open_ & np.isfinite(ti[:, j])
            merge = valid & (ti[:, j] <= e0)
            gap = valid & ~merge
            e0 = np.where(merge, np.maximum(e0, to[:, j]), e0)
            s1 = np.where(gap, ti[:, j], s1)
            p1 = np.where(gap, order[:, j], p1)
            open_ &= ~gap
        return {"s0": s0, "part0": p0, "e0": e0, "s1": s1, "part1": p1}

    def raycast(
        self, origins: np.ndarray, dirs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """First body hit per ray: ``(t, normal, rgb, hit)`` - ``t`` (``inf`` on a miss),
        unit outward normal in C (NaN on a miss), side colour uint8 (0 on a miss), bool."""
        o, d = self._rays(origins, dirs)
        t_in, _ = self.intervals(o, d)
        part = np.argmin(t_in, axis=1)
        t = t_in[np.arange(len(d)), part]
        hit = np.isfinite(t)
        normal = np.full(d.shape, np.nan)
        rgb = np.zeros(d.shape, np.uint8)
        if hit.any():
            P = o[hit] + t[hit, None] * d[hit]
            normal[hit] = self.normal_at(P, part[hit])
            rgb[hit] = self.colour_of(normal[hit])
        return t, normal, rgb, hit

    def raycast_part(self, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Index of the first body part hit per ray (-1 on a miss)."""
        t_in, _ = self.intervals(origins, dirs)
        part = np.argmin(t_in, axis=1)
        return np.where(np.isfinite(t_in.min(axis=1)), part, -1)

    def raycast_background(
        self, origins: np.ndarray, dirs: np.ndarray, *, floor: bool = True, wall: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """First floor / wall hit, ignoring the body: ``(t, normal, rgb, kind)`` with
        ``kind`` = :data:`KIND_FLOOR` / :data:`KIND_WALL` / -1 (nothing: ``t = inf``,
        normal NaN, rgb :data:`SKY_RGB`)."""
        o, d = self._rays(origins, dirs)
        m = len(d)
        best = np.full(m, np.inf)
        kind = np.full(m, -1, np.int64)
        normal = np.full((m, 3), np.nan)
        rgb = np.broadcast_to(SKY_RGB, (m, 3)).copy()
        planes = []
        if floor:  # up . P_C = -cam_height on the floor
            planes.append((KIND_FLOOR, self.floor_normal, -self.cam_height, FLOOR_RGB))
        if wall:  # (-z_W) . P_C = -wall_z on the wall
            planes.append((KIND_WALL, self.wall_normal, -self.wall_z, WALL_RGB))
        for k, n, off, col in planes:
            dn = d @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (off - o @ n) / dn
            t = np.where(np.isfinite(t) & (t > 1e-9), t, np.inf)
            win = t < best
            best = np.where(win, t, best)
            kind[win] = k
            normal[win] = n
            rgb[win] = col
        return best, normal, rgb, kind

    def raycast_scene(
        self, origins: np.ndarray, dirs: np.ndarray, *, floor: bool = True, wall: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """First hit of body, floor or wall: ``(t, normal, rgb, kind)`` with ``kind`` =
        :data:`KIND_BODY` / :data:`KIND_FLOOR` / :data:`KIND_WALL` / -1."""
        tb, nb, cb, hb = self.raycast(origins, dirs)
        tg, ng, cg, kg = self.raycast_background(origins, dirs, floor=floor, wall=wall)
        body = hb & (tb <= tg)
        return (
            np.where(body, tb, tg),
            np.where(body[:, None], nb, ng),
            np.where(body[:, None], cb, cg).astype(np.uint8),
            np.where(body, KIND_BODY, kg),
        )


def make_body(
    *,
    distance: float = 2.5,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    cam_height: float | None = None,
    wall_z: float | None = None,
    lateral: float = 0.0,
) -> Body:
    """The synthetic person ``distance`` m in front of the camera (horizontally, to the body
    axis) and ``lateral`` m to its right, facing it at ``yaw_deg = 0``; the camera looks
    down by ``pitch_deg``.  ``cam_height=None`` puts the camera at ``0.85 + distance *
    tan(pitch)`` so its optical axis passes through the body's mid-height (the body stays
    centred in the photo); a higher camera moves the body down in the photo.  The back
    wall is at ``wall_z`` (default ``distance + 1``, i.e. 3.5 m)."""
    p = math.radians(pitch_deg)
    if cam_height is None:
        cam_height = 0.85 + distance * math.tan(p)
    if wall_z is None:
        wall_z = distance + 1.0
    # rows = the camera's right, down and forward axes in W (looking down by p)
    R_wc = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(p), -math.sin(p)], [0.0, math.sin(p), math.cos(p)]]).T
    R_cw = R_wc.T
    R_wb = _rot_y(-math.radians(yaw_deg))
    R_cb = R_cw @ R_wb
    origin = R_cw @ np.array([lateral, cam_height, distance])
    parts = _parts()
    names = tuple(p_[0] for p_ in parts)
    centers = np.array([origin + R_cb @ p_[1] for p_ in parts])
    rot = np.array([R_cb @ p_[3] for p_ in parts])
    radii = np.array([p_[2] for p_ in parts], np.float64)
    return Body(
        names=names,
        centers=centers,
        rot=rot,
        radii=radii,
        R_cw=R_cw,
        R_cb=R_cb,
        origin=origin,
        distance=float(distance),
        pitch_deg=float(pitch_deg),
        yaw_deg=float(yaw_deg),
        cam_height=float(cam_height),
        wall_z=float(wall_z),
        lateral=float(lateral),
    )


def make_cam(
    width: int = PHOTO_W,
    height: int = PHOTO_H,
    hfov_deg: float = PHOTO_HFOV,
    *,
    grid: int = 96,
    layers: int = 2,
) -> SharpCam:
    """SHARP-style pinhole camera (principal point at the image centre)."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return SharpCam(f=f, width=int(width), height=int(height), grid=int(grid), layers=int(layers))


# --------------------------------------------------------------------------- camera helpers
def project_points(cam: SharpCam, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous pixel coordinates ``(u, v, z)`` of ``...x3`` points in C (NaN for z <= 1e-6)."""
    P = np.asarray(P, np.float64)
    z = P[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        zs = np.where(z > 1e-6, z, np.nan)
        return cam.f * P[..., 0] / zs + cam.width / 2.0, cam.f * P[..., 1] / zs + cam.height / 2.0, z


def pixel_rays(cam: SharpCam, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ray directions with ``z = 1`` through continuous pixel coordinates (``...x3``)."""
    u = np.asarray(u, np.float64)
    v = np.asarray(v, np.float64)
    return np.stack([(u - cam.width / 2.0) / cam.f, (v - cam.height / 2.0) / cam.f, np.ones_like(u)], -1)


def mould_grid(width: int, height: int, long_side: int) -> MouldGrid:
    """Isotropic grid with ``max(Wm, Hm) = min(long_side, max(width, height))`` cells (cell
    ``cell_px = max(W, H) / long`` photo pixels; the short side is rounded)."""
    L = min(int(long_side), max(width, height))
    cell = max(width, height) / L
    return MouldGrid(Wm=max(1, round(width / cell)), Hm=max(1, round(height / cell)), cell_px=cell)


def grid_centers(grid: MouldGrid) -> tuple[np.ndarray, np.ndarray]:
    """Photo coordinates ``(u, v)`` (each ``Hm x Wm``) of the mould-cell centres."""
    rows, cols = np.mgrid[0 : grid.Hm, 0 : grid.Wm]
    return (cols + 0.5) * grid.cell_px, (rows + 0.5) * grid.cell_px


def classify_rgb(rgb: np.ndarray, names: tuple[str, ...] = tuple(PALETTE)) -> np.ndarray:
    """Name of the nearest palette colour (``PALETTE`` keys in ``names``) per ``...x3`` colour
    (uint8, or float in [0, 1]); returns a string array of the same leading shape."""
    c = np.asarray(rgb)
    c = c.astype(np.float64) * (255.0 if c.dtype.kind == "f" else 1.0)
    pal = np.array([PALETTE[n] for n in names], np.float64)
    d = ((c[..., None, :] - pal) ** 2).sum(-1)
    return np.array(names)[np.argmin(d, axis=-1)]


def _subsamples(ss: int) -> np.ndarray:
    return (np.arange(ss) + 0.5) / ss


def _render(cast, H: int, W: int, ss: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive supersampling: ``cast(u, v) -> (rgb uint8 Mx3, covered bool M, t M)`` at
    continuous pixel coordinates.  Pixels whose 8 neighbours all have the same centre
    colour and coverage keep the centre sample; the others average ``ss x ss`` sub-pixel
    rays.  Returns ``(rgb, coverage, t and covered at the pixel centres)``."""
    rows, cols = np.mgrid[0:H, 0:W]
    col_c, cov_c, t_c = cast((cols + 0.5).ravel(), (rows + 0.5).ravel())
    rgb = col_c.reshape(H, W, 3).copy()
    alpha = cov_c.reshape(H, W).astype(np.float32)
    if ss > 1:
        key = (rgb.astype(np.int64) @ np.array([65536, 256, 1])) * 2 + cov_c.reshape(H, W)
        pad = np.pad(key, 1, mode="edge")
        edge = np.zeros((H, W), bool)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                edge |= pad[dy : dy + H, dx : dx + W] != key
        ey, ex = np.nonzero(edge)
        off = _subsamples(ss)
        su = np.broadcast_to(ex[:, None, None] + off[None, None, :], (len(ex), ss, ss)).reshape(-1)
        sv = np.broadcast_to(ey[:, None, None] + off[None, :, None], (len(ey), ss, ss)).reshape(-1)
        col, cov, _ = cast(su, sv)
        alpha[ey, ex] = cov.reshape(-1, ss * ss).mean(1)
        rgb[ey, ex] = np.rint(col.reshape(-1, ss * ss, 3).astype(np.float64).mean(1)).astype(np.uint8)
    return rgb, alpha, t_c.reshape(H, W), cov_c.reshape(H, W)


# --------------------------------------------------------------------------- photo
def render_photo(
    body: Body, cam: SharpCam, *, ss: int = 3, floor: bool = True, wall: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The photo ``(rgb, alpha, depth, body_depth)`` at ``cam.width x cam.height``:

    * ``rgb`` HxWx3 uint8 - first-hit colour, averaged over ``ss x ss`` sub-pixel rays on
      silhouette / colour edges (adaptive supersampling, exact elsewhere);
    * ``alpha`` HxW float32 - the matte: fraction of those rays hitting the body (its
      ``>= 0.5`` part is the body's projection; ``ss=1`` gives a hard 0/1 matte from the
      pixel-centre rays);
    * ``depth`` HxW float32 - z of the first scene hit at the pixel centre (inf: nothing);
    * ``body_depth`` HxW float32 - z of the first body hit at the pixel centre (NaN: none)."""
    zero = np.zeros(3)

    def cast(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t, _, col, kind = body.raycast_scene(zero, pixel_rays(cam, u, v), floor=floor, wall=wall)
        return col, kind == KIND_BODY, t  # z = 1 rays: t is the depth

    rgb, alpha, t, on_body = _render(cast, cam.height, cam.width, ss)
    depth = t.astype(np.float32)
    return rgb, alpha, depth, np.where(on_body, depth, np.float32(np.nan)).astype(np.float32)


# --------------------------------------------------------------------------- SHARP-like cloud
@dataclass
class SharpScene:
    """Output of :func:`sharp_like_scene`; ``sc`` and the label arrays share row order:
    every layer-0 Gaussian (cells row-major), then every layer-1 Gaussian."""

    sc: SplatCloud  # SHARP-like whole-photo cloud, frame C, no sh_rest
    alpha: np.ndarray  # (H, W) float32 exact matte (see render_photo)
    rgb: np.ndarray  # (H, W, 3) uint8 photo
    cam: SharpCam  # with n_raw = len(sc), grid = G, layers
    layer: np.ndarray  # (N,) int8 0 / 1
    kind: np.ndarray  # (N,) int8 KIND_*
    part: np.ndarray  # (N,) int16 body part index into body.names (-1 = floor / wall)
    cell: np.ndarray  # (N,) int64 SHARP-grid cell row * G + col of the Gaussian's ray
    normal: np.ndarray  # (N, 3) true unit surface normal in C (card: the body's at its edge)
    surface_z: np.ndarray  # (N,) z of the true surface point (card: body edge; bg_layer1: bg)
    depth: np.ndarray  # (H, W) float32 z of the first scene hit at pixel centres
    body_depth: np.ndarray  # (H, W) float32 z of the first body hit at pixel centres (NaN)

    def mask(self, *kinds: int) -> np.ndarray:
        """Rows of the given kinds."""
        return np.isin(self.kind, kinds)


def _tangent_frames(n: np.ndarray) -> np.ndarray:
    """``(M, 3, 3)`` rotations with columns ``t1, t2, n`` (det +1), ``t1`` = the camera x axis
    projected onto the tangent plane."""
    ref = np.where((np.abs(n[:, 0]) > 0.95)[:, None], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    t1 = ref - (ref * n).sum(1, keepdims=True) * n
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=2)


def _surface_scales(frames: np.ndarray, rdir: np.ndarray, fp_u: np.ndarray, fp_v: np.ndarray) -> np.ndarray:
    """Flat Gaussians covering their pixel footprint (``0.6 * fp`` on screen, stretched by the
    surface tilt up to 1/0.3), ``s3 = 0.1 * min``."""
    proj1 = np.sqrt(np.maximum(1.0 - (frames[:, :, 0] * rdir).sum(1) ** 2, 0.0))
    proj2 = np.sqrt(np.maximum(1.0 - (frames[:, :, 1] * rdir).sum(1) ** 2, 0.0))
    s1 = 0.6 * fp_u / np.maximum(proj1, 0.3)
    s2 = 0.6 * fp_v / np.maximum(proj2, 0.3)
    return np.stack([s1, s2, 0.1 * np.minimum(s1, s2)], axis=1)


def sharp_like_scene(
    body: Body,
    cam: SharpCam,
    G: int = 96,
    layers: int = 2,
    wall: bool = True,
    floor: bool = True,
    cards: bool = True,
    seed: int = 0,
    *,
    jitter: float = 0.2,
    card_frac: float = 0.5,
    ss: int = 3,
) -> SharpScene:
    """Imitate SHARP's output for the photo of ``body`` (see :class:`SharpScene`).

    One ray per cell of a ``G x G`` grid over the photo (anisotropic: cells are ``W/G x
    H/G`` px), through the cell centre jittered by up to ``jitter`` cells (``seed``):

    * layer 0 - the first hit: body (:data:`KIND_BODY`), else floor / wall.  Flat Gaussians
      along the true normal, on-screen size ``0.6 * z * cell / f`` (more on tilted
      surfaces), opacity 0.95, colour = the surface colour.
    * cards (``cards=True``) - a layer-0 body cell with a non-body 4-neighbour whose next
      surface along its own ray (the layer-1 surface: background, or a body part behind)
      is far behind is eligible (``s1 > 4.5 s2`` and ``s1 > 1.2 cm`` below); a random
      ``card_frac`` of them (``seed``) become ray-aligned cards (:data:`KIND_CARD`) instead
      of surface Gaussians: centred halfway between the body edge and that surface,
      ``s1 = gap / 4`` along the ray (+-2 sigma spans edge to background), ``s2 = 0.6 *
      fp``, ``s3 = 0.3 * fp`` (``fp`` = cell footprint at that depth), opacity 0.9.  In a
      card cell the only other Gaussian is the layer-1 one behind it, as in SHARP.
    * layer 1 (``layers=2``) - body cells: the next surface behind the body part hit first,
      either another body part (:data:`KIND_HIDDEN_BODY`, e.g. the torso behind the
      forearm) or the floor / wall hidden by the body (:data:`KIND_HIDDEN_BG`, the "wall
      blobs"), opacity 0.85; background cells: a copy 3 cm farther along the ray
      (:data:`KIND_BG_LAYER1`, opacity 0.5).

    ``wall`` / ``floor`` switch those planes off for both the cloud and the photo; cells
    whose ray hits nothing get no Gaussian, so ``len(sc) == layers * G * G`` exactly only
    when every ray hits something (the defaults).  ``ss`` = sub-pixel rays per pixel side
    for the photo and matte."""
    if layers not in (1, 2):
        raise ValueError("layers must be 1 or 2")
    W, H, f = cam.width, cam.height, float(cam.f)
    cu, cv_ = W / G, H / G
    rows, cols = np.divmod(np.arange(G * G), G)
    rng = np.random.default_rng(seed)
    jit = rng.uniform(-jitter, jitter, size=(G * G, 2))
    pick = rng.random(G * G) < card_frac
    d = pixel_rays(cam, (cols + 0.5 + jit[:, 0]) * cu, (rows + 0.5 + jit[:, 1]) * cv_)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)  # unit: t is the distance
    zero = np.zeros(3)
    ft = body.first_two(zero, dn)
    tb, nb, cb, kb = body.raycast_background(zero, dn, floor=floor, wall=wall)
    s0, p0 = ft["s0"], ft["part0"]
    is_body = np.isfinite(s0) & (s0 < tb)
    is_bg = ~is_body & np.isfinite(tb)

    def surf(
        idx: np.ndarray,
        t: np.ndarray,
        normal: np.ndarray,
        rgb: np.ndarray,
        kind: int | np.ndarray,
        part: int | np.ndarray,
        layer: int,
        push: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Flat surface Gaussians on the rays ``idx`` at ``t (+ push)``."""
        P = dn[idx] * (t + push)[:, None]
        frames = _tangent_frames(normal)
        kinds = np.broadcast_to(np.asarray(kind, np.int64), idx.shape)
        return {
            "means": P,
            "scales": _surface_scales(frames, dn[idx], P[:, 2] * cu / f, P[:, 2] * cv_ / f),
            "frames": frames,
            "colors": rgb.astype(np.float64) / 255.0,
            "opacity": _OPACITY_LUT[kinds],
            "layer": np.full(idx.shape, layer),
            "kind": kinds,
            "part": np.broadcast_to(np.asarray(part, np.int64), idx.shape),
            "cell": idx,
            "normal": normal,
            "surface_z": dn[idx, 2] * t,
        }

    # ---- layer 0: first hits, body silhouette cells possibly as cards
    ib = np.flatnonzero(is_body)
    ig = np.flatnonzero(is_bg)
    nrm_b = body.normal_at(dn[ib] * s0[ib, None], p0[ib])
    col_b = body.colour_of(nrm_b)
    card = np.zeros(len(ib), bool)
    gap = np.zeros(len(ib))
    if cards and len(ib):
        b2 = is_body.reshape(G, G)
        edge = np.zeros_like(b2)  # a 4-neighbour inside the grid is not body
        edge[1:, :] |= ~b2[:-1, :]
        edge[:-1, :] |= ~b2[1:, :]
        edge[:, 1:] |= ~b2[:, :-1]
        edge[:, :-1] |= ~b2[:, 1:]
        t_far = np.minimum(ft["s1"][ib], tb[ib])  # the next surface behind the edge
        cand = edge.ravel()[ib] & np.isfinite(t_far) & pick[ib]
        gap = np.where(cand, t_far - s0[ib], 0.0)
        fp = dn[ib, 2] * (s0[ib] + 0.5 * gap) * math.sqrt(cu * cv_) / f
        card = cand & (gap / 4.0 > 4.5 * 0.6 * fp) & (gap / 4.0 > 0.012)
    k = ~card
    groups0 = [
        surf(ib[k], s0[ib][k], nrm_b[k], col_b[k], KIND_BODY, p0[ib][k], 0),
        surf(ig, tb[ig], nb[ig], cb[ig], kb[ig], -1, 0),
    ]
    if card.any():
        ic = ib[card]
        a1 = dn[ic]
        ref = np.where((np.abs(a1[:, 0]) > 0.95)[:, None], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
        a2 = ref - (ref * a1).sum(1, keepdims=True) * a1
        a2 /= np.linalg.norm(a2, axis=1, keepdims=True)
        g = gap[card]
        tm = s0[ic] + 0.5 * g
        fp_c = dn[ic, 2] * tm * math.sqrt(cu * cv_) / f
        groups0.append(
            {
                "means": a1 * tm[:, None],
                "scales": np.stack([g / 4.0, 0.6 * fp_c, 0.3 * fp_c], axis=1),
                "frames": np.stack([a1, a2, np.cross(a1, a2)], axis=2),
                "colors": col_b[card].astype(np.float64) / 255.0,
                "opacity": np.full(len(ic), OPACITY[KIND_CARD]),
                "layer": np.zeros(len(ic)),
                "kind": np.full(len(ic), KIND_CARD),
                "part": p0[ic],
                "cell": ic,
                "normal": nrm_b[card],
                "surface_z": dn[ic, 2] * s0[ic],
            }
        )
    out = [_by_cell(_cat(groups0))]

    # ---- layer 1: what lies behind the first body part / a copy behind the background
    if layers == 2:
        s1, p1, e0 = ft["s1"], ft["part1"], ft["e0"]
        ih = np.flatnonzero(is_body & np.isfinite(s1) & (s1 < tb))
        iw = np.flatnonzero(is_body & ~(np.isfinite(s1) & (s1 < tb)) & np.isfinite(tb) & (tb > e0))
        n_h = body.normal_at(dn[ih] * s1[ih, None], p1[ih])
        groups1 = [
            surf(ih, s1[ih], n_h, body.colour_of(n_h), KIND_HIDDEN_BODY, p1[ih], 1),
            surf(iw, tb[iw], nb[iw], cb[iw], KIND_HIDDEN_BG, -1, 1),
            surf(ig, tb[ig], nb[ig], cb[ig], KIND_BG_LAYER1, -1, 1, push=0.03),
        ]
        out.append(_by_cell(_cat(groups1)))
    allg = _cat(out)

    sc = SplatCloud(
        means=allg["means"].astype(np.float32),
        scales=allg["scales"].astype(np.float32),
        quats=rotmat_to_quat(allg["frames"]).astype(np.float32),
        colors=allg["colors"].astype(np.float32),
        opacity=allg["opacity"].astype(np.float32),
    )
    rgb, alpha, depth, body_depth = render_photo(body, cam, ss=ss, floor=floor, wall=wall)
    return SharpScene(
        sc=sc,
        alpha=alpha,
        rgb=rgb,
        cam=SharpCam(f=f, width=W, height=H, n_raw=len(sc), grid=int(G), layers=int(layers)),
        layer=allg["layer"].astype(np.int8),
        kind=allg["kind"].astype(np.int8),
        part=allg["part"].astype(np.int16),
        cell=allg["cell"].astype(np.int64),
        normal=allg["normal"].astype(np.float64),
        surface_z=allg["surface_z"].astype(np.float64),
        depth=depth,
        body_depth=body_depth,
    )


_SHAPES = {"means": (3,), "scales": (3,), "frames": (3, 3), "colors": (3,), "normal": (3,)}
_KEYS = (
    "means",
    "scales",
    "frames",
    "colors",
    "opacity",
    "layer",
    "kind",
    "part",
    "cell",
    "normal",
    "surface_z",
)


def _cat(groups: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        k: np.concatenate([np.asarray(g[k]).reshape(-1, *_SHAPES.get(k, ())) for g in groups]) for k in _KEYS
    }


def _by_cell(g: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = np.argsort(g["cell"], kind="stable")
    return {k: v[order] for k, v in g.items()}


def sharp_like_cloud(
    body: Body,
    cam: SharpCam,
    G: int = 96,
    layers: int = 2,
    wall: bool = True,
    floor: bool = True,
    cards: bool = True,
    seed: int = 0,
) -> tuple[SplatCloud, np.ndarray, np.ndarray]:
    """``(sc, alpha, rgb)`` of :func:`sharp_like_scene` (cloud in C, HxW float32 matte, HxWx3
    uint8 photo); use :func:`sharp_like_scene` for the per-Gaussian ground-truth labels."""
    s = sharp_like_scene(body, cam, G, layers, wall, floor, cards, seed)
    return s.sc, s.alpha, s.rgb


# --------------------------------------------------------------------------- ground truth
def _dist_cells(mask: np.ndarray) -> np.ndarray:
    """Distance (cells) of inside cells to the nearest outside cell; borders count as
    outside (like ``billboard._dist``)."""
    pad = np.pad(mask.astype(np.uint8), 1)
    return cv2.distanceTransform(pad, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1].astype(np.float32)


def synth_front_model(
    body: Body,
    cam: SharpCam,
    grid_long: int = 128,
    *,
    stride: int = 1,
    ss: int = 3,
    floor: bool = True,
    wall: bool = True,
) -> FrontModel:
    """Ground-truth :class:`FrontModel` (what ``sharp_front`` should produce), by ray casting
    at the mould-cell centres of :func:`mould_grid` ``(cam.width, cam.height, grid_long)``:

    * ``matte`` - the cell-centre ray hits the body; ``ring`` - the matte dilated by an
      elliptic kernel of radius ``max(1, round(r_px / cell_px))`` cells with ``r_px =
      max(2, round(1 % of the photo's long side))`` (``run._subject_splats``' dilation),
      minus the matte;
    * ``D_f`` - true z of the first body hit in the matte, the push-pull extrapolation of it
      in the ring, NaN elsewhere; ``D_near`` / ``D_far`` - ``cv2.erode`` 5x5 / ``cv2.dilate``
      9x9 of the extrapolated depth, on matte | ring (NaN elsewhere); ``D_ext`` - push-pull
      extrapolation of ``D_near`` from the matte to the whole grid (no NaN);
    * ``A_f`` - 1 in the matte, 0 elsewhere; ``normals`` - true unit normals (facing the
      camera) in the matte, NaN elsewhere; ``cos_theta = n . (-unit ray)`` (NaN outside);
      ``dt`` - ``cv2.distanceTransform`` of the matte in cells (borders count as outside, 0
      outside); ``rgb`` - the :func:`render_photo` photo resized with ``INTER_AREA``;
    * ``sc`` - one flat body Gaussian per matte cell with ``row % stride == col % stride ==
      0``, at the true surface point on the cell-centre ray (so it projects exactly to the
      cell centre), on-screen size ``0.6 * stride * cell``, side colour, opacity 0.95;
      ``cell`` = its flat mould cell, ``theta_deg`` from ``cos_theta`` of that cell,
      ``rim = dt <= max(2, 0.4 % of max(Hm, Wm))``.
    ``cam`` is stored as given; ``stats = {"synthetic": True, "fallback": False, ...}``."""
    grid = mould_grid(cam.width, cam.height, grid_long)
    Hm, Wm, f = grid.Hm, grid.Wm, float(cam.f)
    u, v = grid_centers(grid)
    d = pixel_rays(cam, u, v).reshape(-1, 3)
    t, n, col, hit = body.raycast(np.zeros(3), d)  # z = 1 rays: t is the depth
    matte = hit.reshape(Hm, Wm)
    if not matte.any():
        raise ValueError("the body is not in the photo")
    Z = np.where(hit, t, 0.0).reshape(Hm, Wm).astype(np.float32)
    Z_fill = push_pull_fill(Z, matte)
    r_px = max(2, int(round(0.01 * max(cam.width, cam.height))))
    r = max(1, int(round(r_px / grid.cell_px)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring = (cv2.dilate(matte.astype(np.uint8), ker) > 0) & ~matte
    band = matte | ring
    nan = np.float32(np.nan)
    near_full = cv2.erode(Z_fill, np.ones((5, 5), np.uint8))
    D_f = np.where(band, Z_fill, nan).astype(np.float32)
    D_near = np.where(band, near_full, nan).astype(np.float32)
    D_far = np.where(band, cv2.dilate(Z_fill, np.ones((9, 9), np.uint8)), nan).astype(np.float32)
    D_ext = push_pull_fill(np.where(matte, near_full, 0.0).astype(np.float32), matte)
    normals = np.where(matte[..., None], n.reshape(Hm, Wm, 3), np.nan).astype(np.float32)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    cos_t = np.where(matte, -(n * dn).sum(1).reshape(Hm, Wm), np.nan).astype(np.float32)
    dt = _dist_cells(matte)
    photo = render_photo(body, cam, ss=ss, floor=floor, wall=wall)[0]
    rgb = cv2.resize(photo, (Wm, Hm), interpolation=cv2.INTER_AREA)

    rr, cc = np.mgrid[0:Hm, 0:Wm]
    sel = matte & (rr % stride == 0) & (cc % stride == 0)
    idx = np.flatnonzero(sel)
    P = d[idx] * t[idx, None]
    frames = _tangent_frames(n[idx])
    fp = P[:, 2] * stride * grid.cell_px / f
    scales = _surface_scales(frames, dn[idx], fp, fp)
    sc = SplatCloud(
        means=P.astype(np.float32),
        scales=scales.astype(np.float32),
        quats=rotmat_to_quat(frames).astype(np.float32),
        colors=(col[idx].astype(np.float32) / 255.0),
        opacity=np.full(len(idx), 0.95, np.float32),
    )
    ct = cos_t.ravel()[idx].astype(np.float64)
    theta = np.degrees(np.arccos(np.clip(np.nan_to_num(ct, nan=0.0), -1.0, 1.0))).astype(np.float32)
    rim = dt.ravel()[idx] <= max(2.0, 0.004 * max(Hm, Wm))
    return FrontModel(
        sc=sc,
        cam=cam,
        grid=grid,
        matte=matte,
        ring=ring,
        D_f=D_f,
        D_near=D_near,
        D_far=D_far,
        D_ext=D_ext,
        A_f=matte.astype(np.float32),
        normals=normals,
        cos_theta=cos_t,
        dt=dt,
        rgb=rgb,
        cell=idx.astype(np.int64),
        theta_deg=theta,
        rim=rim,
        stats={"synthetic": True, "fallback": False, "kept": int(len(idx)), "stride": int(stride)},
    )


def true_back_depth(body: Body, cam: SharpCam, grid_long: int = 128) -> np.ndarray:
    """``(Hm, Wm)`` float32 z where each mould-cell-centre ray finally leaves the body (the
    farthest exit over the parts it crosses: the person's back as seen along the photo's
    rays), NaN where the ray misses the body.  Minus ``synth_front_model(...).D_f`` it is
    the true thickness along the rays."""
    grid = mould_grid(cam.width, cam.height, grid_long)
    u, v = grid_centers(grid)
    _, t_out = body.intervals(np.zeros(3), pixel_rays(cam, u, v).reshape(-1, 3))  # z = 1 rays
    back = np.where(np.isfinite(t_out), t_out, -np.inf).max(axis=1)
    return np.where(np.isfinite(back), back, np.nan).reshape(grid.Hm, grid.Wm).astype(np.float32)


def synth_canon(body: Body, cam: SharpCam, thickness: float = 0.11) -> Canon:
    """Ground-truth :class:`Canon`: ``R`` = ``[x_W y_W z_W]`` in C (so ``up_C = -y^`` is the
    camera's ``-y`` tilted by the pitch), ``x0`` / ``y0`` = centre of the body's exact
    bounding box along ``x^`` / ``y^``, ``z_axis`` = depth of the body axis along ``z^``
    (= ``distance``), ``c = R @ [x0, y0, z_axis]``, ``height_s`` / ``width_s`` = the exact
    vertical / ``x^`` extents of the body (1.69 m / ~1.11 m), ``T_prior = thickness *
    height_s``, ``pitch_deg`` = the camera pitch, ``up_source = "truth"``; ``cut_top`` /
    ``cut_bottom`` by the design rule (matte pixels within 1 % of that border spanning more
    than 15 % of the subject width) on the :func:`render_photo` matte."""
    R = body.R_cw.copy()
    xh, yh, zh = R[:, 0], R[:, 1], R[:, 2]
    x_lo, x_hi = body.extent(xh)
    y_lo, y_hi = body.extent(yh)
    x0, y0 = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    z_axis = float(body.origin @ zh)
    height = y_hi - y_lo
    alpha = render_photo(body, cam, ss=1)[1] >= 0.5
    cut = {"top": False, "bottom": False}
    if alpha.any():
        cols = np.flatnonzero(alpha.any(axis=0))
        width_px = float(cols[-1] - cols[0] + 1)
        k = max(1, math.ceil(0.01 * cam.height))
        for name, band in (("top", alpha[:k]), ("bottom", alpha[-k:])):
            bc = np.flatnonzero(band.any(axis=0))
            cut[name] = bool(len(bc) and (bc[-1] - bc[0] + 1) > 0.15 * width_px)
    return Canon(
        R=R,
        c=R @ np.array([x0, y0, z_axis]),
        up_C=-yh.copy(),
        up_source="truth",
        height_s=float(height),
        width_s=float(x_hi - x_lo),
        T_prior=float(thickness * height),
        z_axis=z_axis,
        pitch_deg=float(body.pitch_deg),
        cut_top=cut["top"],
        cut_bottom=cut["bottom"],
    )


# --------------------------------------------------------------------------- orthographic views
def ortho_axes(canon: Canon, phi_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(right, down, forward)`` of the orthographic view at azimuth ``phi`` in C (design
    3.0): ``forward = cos(phi) z^ - sin(phi) x^``, ``right = cos(phi) x^ + sin(phi) z^``,
    ``down = y^``.  ``phi = 0`` is the photo's side, 90 = "you walked to your right" (the
    camera is on the ``+x^`` side), 180 = behind (mirror image)."""
    p = math.radians(phi_deg)
    xh, yh, zh = canon.R[:, 0], canon.R[:, 1], canon.R[:, 2]
    return math.cos(p) * xh + math.sin(p) * zh, yh.copy(), math.cos(p) * zh - math.sin(p) * xh


def ortho_project(
    canon: Canon, phi_deg: float, m: float, N: int, P: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous pixel ``(u, v)`` of ``...x3`` points in C in :func:`ortho_view`
    (``u = (P - c).right / m + N/2``, ``v = (P - c).down / m + N/2``; the S origin ``c``
    maps to the image centre) and their depth ``(P - c).forward``."""
    right, down, fwd = ortho_axes(canon, phi_deg)
    rel = np.asarray(P, np.float64) - canon.c
    return rel @ right / m + N / 2.0, rel @ down / m + N / 2.0, rel @ fwd


def ortho_view(
    body: Body,
    canon: Canon,
    phi_deg: float,
    m: float = 0.01,
    N: int = 200,
    *,
    ss: int = 2,
    bg: np.ndarray = VIEW_BG_RGB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthographic view of the body alone (no floor / wall) at azimuth ``phi_deg``, ``m``
    metres per pixel, ``N x N`` pixels (pixel ``(i, j)`` covers ``u in [j, j+1)``,
    ``v in [i, i+1)``, see :func:`ortho_project`).  Returns ``(rgb, alpha, depth)``: NxNx3
    uint8 composited over ``bg``, NxN float32 coverage (``ss x ss`` sub-pixel rays on
    edges, as :func:`render_photo`), NxN float32 ``(P_hit - c).forward`` at the pixel centre
    (NaN where it misses)."""
    right, down, fwd = ortho_axes(canon, phi_deg)
    far = 20.0  # rays start this far in front of the axis, well outside the body
    bg_c = np.asarray(bg, np.uint8)

    def cast(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        o = canon.c + ((u - N / 2.0) * m)[:, None] * right + ((v - N / 2.0) * m)[:, None] * down - far * fwd
        t, _, col, hit = body.raycast(o, np.broadcast_to(fwd, o.shape))
        return np.where(hit[:, None], col, bg_c), hit, t - far

    rgb, alpha, depth, _ = _render(cast, N, N, ss)
    return rgb, alpha, np.where(np.isfinite(depth), depth, np.nan).astype(np.float32)


__all__ = [
    "BACK_RGB",
    "BODY_HEIGHT",
    "FLOOR_RGB",
    "FRONT_RGB",
    "KIND_BG_LAYER1",
    "KIND_BODY",
    "KIND_CARD",
    "KIND_FLOOR",
    "KIND_HIDDEN_BG",
    "KIND_HIDDEN_BODY",
    "KIND_NAMES",
    "KIND_WALL",
    "LEFT_RGB",
    "OPACITY",
    "PALETTE",
    "PHOTO_H",
    "PHOTO_HFOV",
    "PHOTO_W",
    "RIGHT_RGB",
    "SIDES",
    "SKY_RGB",
    "VIEW_BG_RGB",
    "WALL_RGB",
    "Body",
    "SharpScene",
    "classify_rgb",
    "grid_centers",
    "make_body",
    "make_cam",
    "mould_grid",
    "ortho_axes",
    "ortho_project",
    "ortho_view",
    "pixel_rays",
    "project_points",
    "render_photo",
    "sharp_like_cloud",
    "sharp_like_scene",
    "synth_canon",
    "synth_front_model",
    "true_back_depth",
]


# --------------------------------------------------------------------------- SHARP 360 views (views_lift tests)
def make_body_pose(left_arm_deg: float = 20.0, **kw) -> Body:
    """:func:`make_body` (same keywords) with the raised LEFT arm at ``left_arm_deg`` above the
    horizontal instead of 20 deg - another pose of the same person, e.g. for a back photo that
    does not match the front."""
    from dataclasses import replace as _replace

    body = make_body(**kw)
    a = math.radians(left_arm_deg)
    sh_l = (0.17, 1.38, 0.0)
    hand_l = (0.17 + 0.62 * math.cos(a), 1.38 + 0.62 * math.sin(a), 0.0)
    _, ctr, radii, rot = _seg("arm_left", sh_l, hand_l, 0.05, 0.02)
    k = body.part("arm_left")
    centers, rots, rad = body.centers.copy(), body.rot.copy(), body.radii.copy()
    centers[k] = body.origin + body.R_cb @ ctr
    rots[k] = body.R_cb @ rot
    rad[k] = radii
    return _replace(body, centers=centers, rot=rots, radii=rad)


def ortho_view_parts(body: Body, canon: Canon, phi_deg: float, m: float = 0.01, N: int = 200) -> np.ndarray:
    """``(N, N)`` int index (into ``body.names``) of the first body part hit at each pixel centre
    of :func:`ortho_view` with the same arguments (-1 where the ray misses the body)."""
    right, down, fwd = ortho_axes(canon, phi_deg)
    rows, cols = np.mgrid[0:N, 0:N]
    u = (cols + 0.5).ravel()
    v = (rows + 0.5).ravel()
    far = 20.0
    o = canon.c + ((u - N / 2.0) * m)[:, None] * right + ((v - N / 2.0) * m)[:, None] * down - far * fwd
    return body.raycast_part(o, np.broadcast_to(fwd, o.shape)).reshape(N, N)


__all__ += ["make_body_pose", "ortho_view_parts"]


# --------------------------------------------------------------------------- MV-Adapter-like view sets (M4)
#: Azimuths of MV-Adapter's six orthographic views (design 2 / 4.3), in the app's convention.
MVADAPTER_AZIMUTHS = (0.0, 45.0, 90.0, 180.0, 270.0, 315.0)


@dataclass
class MVSet:
    """Output of :func:`mvadapter_like_set`: what the MV-Adapter worker would write.

    ``images`` / ``alphas`` / ``phis`` are keyed by file name (``view_0.png`` ...): the RGB
    picture composited over the background, the true coverage, and the TRUE azimuth of the
    render (the camera file may label it differently, see ``flip_sign``).  ``cameras`` is the
    ``cameras.json`` dict (design 4.3)."""

    images: dict[str, np.ndarray]
    alphas: dict[str, np.ndarray]
    phis: dict[str, float]
    cameras: dict

    def write(self, folder) -> object:
        """Write the pictures (PNG) and ``cameras.json`` into ``folder``; returns the JSON path."""
        import json
        from pathlib import Path

        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        for name, rgb in self.images.items():
            cv2.imwrite(str(out / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        path = out / "cameras.json"
        path.write_text(json.dumps(self.cameras, indent=1), encoding="utf-8")
        return path


def _c2w_opencv(phi_deg: float, distance: float) -> list[list[float]]:
    """4 x 4 camera-to-world matrix (OpenCV axes: x right, y down, z forward) of the view at
    ``phi`` in a generator-style world that is NOT frame S: y up, and turned by 90 deg about
    it (MV-Adapter's ``az - 90``), so a reader must not assume the world axes."""
    p = math.radians(phi_deg)
    right = np.array([math.cos(p), 0.0, math.sin(p)])  # in S
    down = np.array([0.0, 1.0, 0.0])
    fwd = np.array([-math.sin(p), 0.0, math.cos(p)])
    W = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])  # S -> world, det +1
    M = np.eye(4)
    M[:3, :3] = W @ np.column_stack([right, down, fwd])
    M[:3, 3] = W @ (-distance * fwd)
    return M.tolist()


def mvadapter_like_set(
    body: Body,
    canon: Canon,
    ortho_input,
    *,
    scale: float = 1.0,
    offset_px: tuple[float, float] = (0.0, 0.0),
    flip_sign: bool = False,
    grey_bg: bool = True,
    size: int = 512,
    view_scale: dict[float, float] | None = None,
    axis_dz: float = 0.0,
    azimuths: tuple[float, ...] = MVADAPTER_AZIMUTHS,
    ss: int = 2,
    paint: dict[str, np.ndarray] | None = None,
) -> MVSet:
    """Six :func:`ortho_view` renders of ``body`` at MV-Adapter's azimuths with the app's
    framing - what an ideal generator returns for the levelled input ``ortho_input``
    (``views_lift.OrthoInput``): ``size`` px, the input's pixel scale (``m = m_in * S /
    size``), the orbit axis at the picture centre, i.e. at ``x = center_S[0]`` and ``z =
    axis_dz`` in ``canon``'s frame S (``canon`` = the TRUE frame), rows centred on
    ``center_S[1]``.

    Perturbations (all optional): ``scale`` / ``offset_px`` - every picture scaled about its
    centre and shifted by ``(dx, dy)`` px (a generator that re-framed its output); with
    ``view_scale`` ``{phi: s}`` one view is scaled on its own as well; ``flip_sign`` - the
    camera file labels the view at ``phi`` as ``360 - phi`` (the azimuth sign convention
    reversed, consistently in ``phi_deg``, ``azimuth_deg`` and ``c2w``); ``grey_bg=False``
    composites over white instead of 0.5 grey; ``paint`` recolours body parts (name ->
    colour) in every view to track them.  Returns an :class:`MVSet`."""
    import dataclasses as _dc

    S_in = int(ortho_input.size)
    m = float(ortho_input.m_in) * S_in / int(size)
    cx, cy = (float(t) for t in np.asarray(ortho_input.center_S, np.float64)[:2])
    bg = VIEW_BG_RGB if grey_bg else np.array([255, 255, 255], np.uint8)
    dx, dy = (float(t) for t in offset_px)
    axis = canon.c + canon.R @ np.array([cx, cy, float(axis_dz)])
    images: dict[str, np.ndarray] = {}
    alphas: dict[str, np.ndarray] = {}
    phis: dict[str, float] = {}
    entries = []
    distance = 1.8
    for i, phi in enumerate(azimuths):
        k = float(scale) * float((view_scale or {}).get(float(phi), 1.0))
        mv = m / k  # metres per pixel of this picture
        right, down, _ = ortho_axes(canon, phi)
        c_view = axis - dx * mv * right - dy * mv * down  # the axis lands at (N/2 + dx, N/2 + dy)
        view_canon = _dc.replace(canon, c=c_view)
        rgb, alpha, _ = ortho_view(body, view_canon, phi, m=mv, N=int(size), ss=ss, bg=bg)
        if paint:
            parts = ortho_view_parts(body, view_canon, phi, m=mv, N=int(size))
            for part, colour in paint.items():
                rgb[parts == body.part(part)] = colour
        name = f"view_{i}.png"
        label = (360.0 - phi) % 360.0 if flip_sign else float(phi)
        images[name], alphas[name], phis[name] = rgb, alpha, float(phi)
        entries.append(
            {
                "file": name,
                "phi_deg": label,
                "azimuth_deg": label,
                "elevation_deg": 0.0,
                "c2w": _c2w_opencv(label, distance),
            }
        )
    cameras = {
        "projection": "orthographic",
        "ortho_half_extent": 0.55,
        "distance": distance,
        "width": int(size),
        "height": int(size),
        "input_fill": 0.9,
        "pixel_center": 0.5,
        "convention": "c2w_opencv",
        "views": entries,
    }
    return MVSet(images=images, alphas=alphas, phis=phis, cameras=cameras)


__all__ += ["MVADAPTER_AZIMUTHS", "MVSet", "mvadapter_like_set"]
