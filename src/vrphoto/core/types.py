"""Shared data types: the frozen contract between all VR Photo Studio modules.

Coordinate conventions (see docs/CONTRACTS.md):

* Images are numpy arrays, row-major, ``HxWx3 uint8`` RGB (never BGR) unless noted.
* Source camera frame: x right, y DOWN, z FORWARD, units metres (OpenCV convention).
* Exported meshes / splats / viewer scenes use glTF convention: x right, y UP,
  z BACKWARD (camera looks down -z), metres.  Convert with
  ``geometry.camera.cv_to_gltf``.
* Disparity / inverse depth: larger = nearer.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .errors import Cancelled

RGB = np.ndarray  # HxWx3 uint8
RGBA = np.ndarray  # HxWx4 uint8
FloatMap = np.ndarray  # HxW float32


# --------------------------------------------------------------------------- camera
@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels for an image of ``width`` x ``height``."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @staticmethod
    def from_hfov(width: int, height: int, hfov_deg: float) -> Intrinsics:
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return Intrinsics(f, f, width / 2.0, height / 2.0, int(width), int(height))

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def scaled(self, s: float) -> Intrinsics:
        """Intrinsics for the same camera with the image resized by factor ``s``."""
        return Intrinsics(
            self.fx * s,
            self.fy * s,
            self.cx * s,
            self.cy * s,
            max(1, int(round(self.width * s))),
            max(1, int(round(self.height * s))),
        )

    def resized(self, width: int, height: int) -> Intrinsics:
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, int(width), int(height))

    def with_principal(self, cx: float, cy: float) -> Intrinsics:
        return Intrinsics(self.fx, self.fy, cx, cy, self.width, self.height)


def _rot_xyz(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Camera rotation in the y-down source frame.  Columns are the new camera axes.

    yaw > 0 turns right (forward moves toward +x), pitch > 0 looks UP (forward moves
    toward -y), roll > 0 rotates clockwise as seen by the viewer."""
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


@dataclass
class Pose:
    """Pose of a NEW (virtual) camera expressed in the SOURCE camera frame.

    A point ``P_src`` in source-camera coordinates maps into the new camera as
    ``P_new = R.T @ (P_src - t)``.  ``t`` is the new camera centre in source coords
    (metres), columns of ``R`` are the new camera axes in source coords.
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @staticmethod
    def identity() -> Pose:
        return Pose(np.eye(3), np.zeros(3))

    @staticmethod
    def translate(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Pose:
        return Pose(np.eye(3), np.array([x, y, z], dtype=np.float64))

    @staticmethod
    def look(
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        t: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> Pose:
        return Pose(_rot_xyz(yaw_deg, pitch_deg, roll_deg), np.asarray(t, dtype=np.float64))

    def world_to_cam(self, P: np.ndarray) -> np.ndarray:
        """Map ``...x3`` source-frame points into this camera's frame."""
        return (P - self.t) @ self.R  # == R.T @ (P - t) for row vectors


# --------------------------------------------------------------------------- depth
@dataclass
class DepthResult:
    """Output of a depth estimator, at the resolution of the input image."""

    disparity: FloatMap  # float32 HxW in [0, 1], 1 = nearest (relative)
    model_id: str
    metric_m: FloatMap | None = None  # float32 HxW metric depth (z) in metres if known
    intrinsics: Intrinsics | None = None  # estimated camera if the model provides it
    raw: FloatMap | None = None  # raw model output (resized), for debugging


@dataclass
class DepthScale:
    """Maps relative disparity d in [0,1] to inverse depth: inv_z = 1/far + d*(1/near - 1/far)."""

    near_m: float = 1.0
    far_m: float = 12.0

    def to_inv_z(self, disparity: FloatMap) -> FloatMap:
        a = 1.0 / self.far_m
        b = 1.0 / self.near_m
        return (a + np.clip(disparity, 0.0, 1.0) * (b - a)).astype(np.float32)


# --------------------------------------------------------------------------- layers
@dataclass
class Layer:
    rgb: RGB  # HxWx3 uint8
    inv_z: FloatMap  # float32 HxW, 1/metres, > 0
    alpha: FloatMap  # float32 HxW in [0,1]


@dataclass
class LayeredImage:
    """Layered depth image.  ``layers`` are ordered back -> front.  All layers share ``K``."""

    layers: list[Layer]
    K: Intrinsics

    @property
    def front(self) -> Layer:
        return self.layers[-1]


@dataclass
class StereoParams:
    ipd_m: float = 0.063
    strength: float = 1.0  # multiplies baseline in flat mode
    convergence_m: float | None = None  # flat mode: zero-parallax distance (None = auto)
    max_parallax_pct: float = 3.0  # flat mode: clamp |parallax| to this % of width (uncrossed side
    # additionally capped by ipd_m/screen_width_m, see below)
    mode: Literal["flat", "vr"] = "flat"
    swap_eyes: bool = False
    # flat mode: width (m) of the virtual screen the stereo image is shown on (viewer default
    # 3.0 m).  Uncrossed parallax is limited so the background never exceeds the eye
    # separation on that screen (no divergence).
    screen_width_m: float = 3.0


@dataclass
class RenderResult:
    rgb: RGB
    alpha: FloatMap  # coverage in [0,1]; < 0.5 = hole


# --------------------------------------------------------------------------- 3D
@dataclass
class Mesh:
    """Triangle mesh in glTF frame (y up, metres)."""

    vertices: np.ndarray  # float32 Nx3
    faces: np.ndarray  # uint32 Mx3 (counter-clockwise = front)
    uvs: np.ndarray  # float32 Nx2, (0,0) = top-left of texture (glTF convention)
    texture: np.ndarray  # HxWx3 or HxWx4 uint8
    name: str = "mesh"
    alpha_mode: Literal["OPAQUE", "MASK", "BLEND"] = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    normals: np.ndarray | None = None  # float32 Nx3 optional


@dataclass
class SplatCloud:
    """3D Gaussian splats in glTF frame (y up, metres)."""

    means: np.ndarray  # float32 Nx3
    scales: np.ndarray  # float32 Nx3 linear standard deviations in metres
    quats: np.ndarray  # float32 Nx4 (w, x, y, z), normalised
    colors: np.ndarray  # float32 Nx3 linear-ish sRGB in [0,1]
    opacity: np.ndarray  # float32 N in [0,1]
    sh_rest: np.ndarray | None = None  # float32 NxKx3 higher-order SH, optional

    def __len__(self) -> int:
        return int(self.means.shape[0])


# --------------------------------------------------------------------------- progress / cancel
class CancelToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def check(self) -> None:
        if self._ev.is_set():
            raise Cancelled("Operation cancelled")


ProgressCallback = Callable[[float, str], None]


class ProgressReporter:
    """Reports progress fraction in [0,1] with a message.  ``sub(a, b)`` returns a
    reporter mapping its own [0,1] onto [a,b] of this reporter."""

    def __init__(self, callback: ProgressCallback | None = None, lo: float = 0.0, hi: float = 1.0):
        self._cb = callback
        self._lo = lo
        self._hi = hi

    def __call__(self, frac: float, msg: str = "") -> None:
        if self._cb is None:
            return
        f = self._lo + (self._hi - self._lo) * min(1.0, max(0.0, float(frac)))
        self._cb(f, msg)

    def sub(self, a: float, b: float) -> ProgressReporter:
        span = self._hi - self._lo
        return ProgressReporter(self._cb, self._lo + span * a, self._lo + span * b)

    @staticmethod
    def null() -> ProgressReporter:
        return ProgressReporter(None)


__all__ = [
    "RGB",
    "RGBA",
    "FloatMap",
    "Intrinsics",
    "Pose",
    "DepthResult",
    "DepthScale",
    "Layer",
    "LayeredImage",
    "StereoParams",
    "RenderResult",
    "Mesh",
    "SplatCloud",
    "CancelToken",
    "ProgressReporter",
    "ProgressCallback",
]
