"""Virtual camera paths for "living photo" parallax videos.

A :class:`CameraPath` is a list of :class:`~vrphoto.core.types.Pose` (one per output frame,
new camera expressed in the SOURCE camera frame: x right, y down, z forward, metres) plus
the frame rate.  :func:`make_path` builds the stock paths offered in the GUI:

``dolly``     in/out along the optical axis (z in [-A, +A]).
``circle``    small circle in the image plane (x/y), radius A.
``swing``     left-right arc around the focus point with a slight yaw toward it.
``zoom``      dolly toward the focus point (a FOV change is not representable by a pose).
``kenburns``  slow diagonal translation plus a slight dolly.
``orbit``     arc around the focus point with full yaw, so the subject stays centred.
``custom``    Catmull-Rom spline through user keyframes ``{t, x, y, z, yaw, pitch, roll}``.

Timing: with ``loop=True`` every path is periodic with period ``n_frames`` (frame
``n_frames`` would equal frame 0), so the encoded video loops seamlessly; the motion is
sinusoidal, i.e. eased at every turning point.  With ``loop=False`` the path runs once
from start to end with a cosine ease-in/ease-out (zero velocity at both ends).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import InputError
from ..core.log import get_logger
from ..core.types import Pose

log = get_logger(__name__)

PATH_KINDS: list[str] = ["dolly", "circle", "swing", "zoom", "kenburns", "orbit", "custom"]

#: Focus distance used when the caller does not provide one (metres).
DEFAULT_FOCUS_M = 2.0
#: Fraction of the full look-at yaw used by the "swing" path.
SWING_YAW_FRACTION = 0.5
#: Hard limit on frames per path (about 9 minutes at 60 fps) - protects against typos.
MAX_FRAMES = 32_768

_KEYFRAME_FIELDS = ("x", "y", "z", "yaw", "pitch", "roll")


@dataclass
class CameraPath:
    """Per-frame virtual camera poses and the playback frame rate."""

    poses: list[Pose] = field(default_factory=list)
    fps: float = 30.0

    @property
    def n_frames(self) -> int:
        """Number of frames (poses) in the path."""
        return len(self.poses)

    @property
    def duration(self) -> float:
        """Playback duration in seconds."""
        return len(self.poses) / float(self.fps) if self.fps > 0 else 0.0

    def translations(self) -> np.ndarray:
        """``Nx3`` array of camera centres (source frame, metres)."""
        if not self.poses:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([np.asarray(p.t, dtype=np.float64) for p in self.poses])

    def __len__(self) -> int:
        return len(self.poses)


# --------------------------------------------------------------------------- timing helpers
def _ease(u: np.ndarray) -> np.ndarray:
    """Cosine ease-in/ease-out on [0, 1] (zero slope at both ends)."""
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * u)


def _phases(n: int, loop: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(osc, ramp)`` drivers for ``n`` frames.

    ``osc`` goes -1 -> +1 (-> -1 when looping) with zero velocity at the turning points:
    ``-cos(2*pi*i/n)`` when looping, an eased ramp otherwise.  ``ramp`` goes 0 -> 1
    (-> 0 when looping) likewise.
    """
    i = np.arange(n, dtype=np.float64)
    if loop:
        theta = 2.0 * math.pi * i / max(n, 1)
        osc = -np.cos(theta)
    else:
        u = i / (n - 1) if n > 1 else np.zeros(n)
        osc = 2.0 * _ease(u) - 1.0
    return osc, 0.5 * (osc + 1.0)


def _angle(n: int, loop: bool) -> np.ndarray:
    """One full revolution: ``2*pi*i/n`` when looping, eased 0 -> 2*pi otherwise."""
    i = np.arange(n, dtype=np.float64)
    if loop:
        return 2.0 * math.pi * i / max(n, 1)
    u = i / (n - 1) if n > 1 else np.zeros(n)
    return 2.0 * math.pi * _ease(u)


def _poses(xyz: np.ndarray, ypr_deg: np.ndarray | None = None) -> list[Pose]:
    """Build poses from ``Nx3`` translations and optional ``Nx3`` yaw/pitch/roll (degrees)."""
    out: list[Pose] = []
    for k in range(xyz.shape[0]):
        t = np.asarray(xyz[k], dtype=np.float64).copy()
        if ypr_deg is None or not np.any(ypr_deg[k]):
            out.append(Pose(np.eye(3), t))
        else:
            y, p, r = (float(a) for a in ypr_deg[k])
            out.append(Pose.look(y, p, r, t))
    return out


def _look_yaw_deg(x: np.ndarray, z: np.ndarray, focus: float) -> np.ndarray:
    """Yaw (degrees, > 0 turns right) that points a camera at (x, 0, z) at (0, 0, focus)."""
    return np.degrees(np.arctan2(-x, focus - z))


# --------------------------------------------------------------------------- stock paths
def _dolly(n: int, a: float, loop: bool, focus: float) -> list[Pose]:
    osc, _ = _phases(n, loop)
    xyz = np.zeros((n, 3))
    xyz[:, 2] = a * osc
    return _poses(xyz)


def _circle(n: int, a: float, loop: bool, focus: float) -> list[Pose]:
    th = _angle(n, loop)
    xyz = np.zeros((n, 3))
    xyz[:, 0] = a * np.cos(th)
    xyz[:, 1] = a * np.sin(th)
    return _poses(xyz)


def _arc(n: int, a: float, loop: bool, focus: float, yaw_fraction: float) -> list[Pose]:
    """Camera on a circle of radius ``focus`` around (0, 0, focus), lateral offset ``a*osc``."""
    osc, _ = _phases(n, loop)
    if loop:
        osc = np.sin(2.0 * math.pi * np.arange(n) / max(n, 1))  # start centred, swing right first
    x = a * osc
    x = np.clip(x, -0.95 * focus, 0.95 * focus)
    z = focus - np.sqrt(np.maximum(focus * focus - x * x, 0.0))
    xyz = np.stack([x, np.zeros(n), z], axis=1)
    ypr = np.zeros((n, 3))
    ypr[:, 0] = yaw_fraction * _look_yaw_deg(x, z, focus)
    return _poses(xyz, ypr)


def _swing(n: int, a: float, loop: bool, focus: float) -> list[Pose]:
    return _arc(n, a, loop, focus, SWING_YAW_FRACTION)


def _orbit(n: int, a: float, loop: bool, focus: float) -> list[Pose]:
    return _arc(n, a, loop, focus, 1.0)


def _zoom(n: int, a: float, loop: bool, focus: float) -> list[Pose]:
    # forward travel: twice the amplitude, never more than 40% of the way to the subject
    travel = min(2.0 * a, 0.4 * focus)
    _, ramp = _phases(n, loop)
    xyz = np.zeros((n, 3))
    xyz[:, 2] = travel * ramp
    return _poses(xyz)


def _kenburns(n: int, a: float, loop: bool, focus: float) -> list[Pose]:
    osc, ramp = _phases(n, loop)
    xyz = np.zeros((n, 3))
    xyz[:, 0] = a * osc  # left -> right
    xyz[:, 1] = -0.6 * a * osc  # bottom -> top (y is down)
    xyz[:, 2] = 0.5 * a * ramp  # slight push in
    return _poses(xyz)


_STOCK = {
    "dolly": _dolly,
    "circle": _circle,
    "swing": _swing,
    "zoom": _zoom,
    "kenburns": _kenburns,
    "orbit": _orbit,
}


# --------------------------------------------------------------------------- custom keyframes
def _parse_keyframes(keyframes: list[dict[str, Any]] | None) -> tuple[np.ndarray, np.ndarray]:
    """Validate keyframes -> ``(t[K], values[K, 6])`` sorted by time."""
    if not keyframes:
        raise InputError("The custom camera path needs at least one keyframe.")
    if not isinstance(keyframes, (list, tuple)):
        raise InputError("Custom keyframes must be a list of {t, x, y, z, yaw, pitch, roll} objects.")
    k = len(keyframes)
    ts = np.empty(k, dtype=np.float64)
    vals = np.zeros((k, len(_KEYFRAME_FIELDS)), dtype=np.float64)
    for i, kf in enumerate(keyframes):
        if not isinstance(kf, dict):
            raise InputError(f"Keyframe {i + 1} must be an object with t, x, y, z, yaw, pitch, roll.")
        unknown = set(kf) - set(_KEYFRAME_FIELDS) - {"t"}
        if unknown:
            raise InputError(f"Keyframe {i + 1} has unknown field(s): {', '.join(sorted(unknown))}.")
        try:
            t = float(kf["t"]) if "t" in kf else (i / (k - 1) if k > 1 else 0.0)
            row = [float(kf.get(name, 0.0)) for name in _KEYFRAME_FIELDS]
        except (TypeError, ValueError) as exc:
            raise InputError(f"Keyframe {i + 1} contains a non-numeric value.") from exc
        if not all(math.isfinite(v) for v in (t, *row)):
            raise InputError(f"Keyframe {i + 1} contains a non-finite value.")
        if not 0.0 <= t <= 1.0:
            raise InputError(f"Keyframe {i + 1}: time t must be between 0 and 1 (got {t}).")
        ts[i] = t
        vals[i] = row
    order = np.argsort(ts, kind="stable")
    ts, vals = ts[order], vals[order]
    if k > 1 and np.any(np.diff(ts) <= 1e-9):
        raise InputError("Two custom keyframes have the same time t; every keyframe needs its own time.")
    return ts, vals


def _hermite(
    p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, dt: float, s: np.ndarray
) -> np.ndarray:
    """Cubic Hermite interpolation on one segment (``s`` in [0,1], ``m`` per unit time)."""
    s = s[:, None]
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return h00 * p0 + h10 * dt * m0 + h01 * p1 + h11 * dt * m1


def _catmull_rom(ts: np.ndarray, vals: np.ndarray, tq: np.ndarray, loop: bool) -> np.ndarray:
    """Non-uniform Catmull-Rom (finite-difference tangents) evaluated at times ``tq``.

    Looping: the keyframes repeat with period 1.  Otherwise the curve holds the first /
    last keyframe outside ``[t0, tK]`` and has zero tangents at both ends (ease in/out).
    """
    k = ts.size
    if k == 1:
        return np.repeat(vals[:1], tq.size, axis=0)
    if loop:
        # extend periodically: one keyframe before, two after
        # k >= 2 here: wrap one keyframe before and two after (t + 1)
        t_ext = np.concatenate([[ts[-1] - 1.0], ts, [ts[0] + 1.0, ts[1] + 1.0]])
        v_ext = np.concatenate([vals[-1:], vals, vals[:1], vals[1:2]])
        m = np.zeros_like(v_ext)
        m[1:-1] = (v_ext[2:] - v_ext[:-2]) / (t_ext[2:] - t_ext[:-2])[:, None]
        tq = np.mod(tq - ts[0], 1.0) + ts[0]  # map into [t0, t0 + 1)
        knots, pts, tans = t_ext[1:-1], v_ext[1:-1], m[1:-1]  # ts..., t0+1
    else:
        m = np.zeros_like(vals)
        if k > 2:
            m[1:-1] = (vals[2:] - vals[:-2]) / (ts[2:] - ts[:-2])[:, None]
        knots, pts, tans = ts, vals, m
        tq = np.clip(tq, ts[0], ts[-1])
    seg = np.clip(np.searchsorted(knots, tq, side="right") - 1, 0, knots.size - 2)
    out = np.empty((tq.size, vals.shape[1]), dtype=np.float64)
    for j in np.unique(seg):
        sel = seg == j
        dt = float(knots[j + 1] - knots[j])
        s = (tq[sel] - knots[j]) / dt
        out[sel] = _hermite(pts[j], pts[j + 1], tans[j], tans[j + 1], dt, s)
    return out


def _custom(n: int, loop: bool, keyframes: list[dict[str, Any]] | None) -> list[Pose]:
    ts, vals = _parse_keyframes(keyframes)
    if loop and ts.size > 1 and ts[-1] >= 1.0 - 1e-9:
        if ts[0] <= 1e-9:
            if not np.allclose(vals[-1], vals[0], atol=1e-9):
                log.warning("Looping custom path: keyframe at t=1 differs from t=0; using the t=0 keyframe")
            ts, vals = ts[:-1], vals[:-1]
        else:
            ts = ts.copy()
            ts[-1] = 0.0  # t = 1 is the same instant as t = 0 in a loop
            order = np.argsort(ts)
            ts, vals = ts[order], vals[order]
    i = np.arange(n, dtype=np.float64)
    tq = i / n if loop else (i / (n - 1) if n > 1 else np.zeros(n))
    v = _catmull_rom(ts, vals, tq, loop)
    return _poses(v[:, :3], v[:, 3:])


# --------------------------------------------------------------------------- public API
def make_path(
    kind: str,
    *,
    n_frames: int,
    amplitude_m: float = 0.05,
    fps: float = 30.0,
    loop: bool = True,
    keyframes: list[dict] | None = None,
    focus_m: float | None = None,
) -> CameraPath:
    """Build a virtual camera path of ``n_frames`` poses.

    Args:
        kind: one of :data:`PATH_KINDS`.
        n_frames: number of frames (>= 1).
        amplitude_m: size of the motion in metres (radius / half travel).  Ignored by
            ``custom`` (keyframes carry absolute offsets).
        fps: playback frame rate stored in the result.
        loop: periodic path (period ``n_frames``) for seamless looping video; otherwise
            a single eased pass from start to end.
        keyframes: ``custom`` only - list of ``{"t": 0..1, "x", "y", "z" (metres), "yaw",
            "pitch", "roll" (degrees)}``; missing fields default to 0, missing ``t`` means
            evenly spaced.  Interpolated with a Catmull-Rom spline.  Source camera frame:
            x > 0 right, y > 0 DOWN, z > 0 forward; yaw > 0 turns right, pitch > 0 looks
            up, roll > 0 rotates clockwise (see ``Pose.look``).
        focus_m: distance of the subject (pivot of ``swing``/``orbit``, target of
            ``zoom``); default :data:`DEFAULT_FOCUS_M`.

    Raises:
        InputError: unknown kind or invalid parameters.
    """
    name = str(kind).strip().lower()
    if name not in PATH_KINDS:
        raise InputError(f"Unknown camera path {kind!r}. Choose one of: {', '.join(PATH_KINDS)}.")
    try:
        n = int(n_frames)
    except (TypeError, ValueError) as exc:
        raise InputError(f"Number of frames must be an integer (got {n_frames!r}).") from exc
    if n < 1:
        raise InputError(f"A camera path needs at least one frame (got {n_frames}).")
    if n > MAX_FRAMES:
        raise InputError(f"Camera path too long: {n} frames (maximum {MAX_FRAMES}).")
    fps = float(fps)
    if not (math.isfinite(fps) and fps > 0):
        raise InputError(f"Frame rate must be positive (got {fps}).")
    a = float(amplitude_m)
    if not (math.isfinite(a) and a >= 0):
        raise InputError(f"Motion amplitude must be zero or positive (got {amplitude_m} m).")
    focus = DEFAULT_FOCUS_M if focus_m is None else float(focus_m)
    if not (math.isfinite(focus) and focus > 0):
        raise InputError(f"Focus distance must be positive (got {focus_m} m).")
    if name != "custom" and keyframes:
        log.debug("Keyframes ignored for camera path %r", name)

    poses = _custom(n, bool(loop), keyframes) if name == "custom" else _STOCK[name](n, a, bool(loop), focus)
    return CameraPath(poses=poses, fps=fps)


__all__ = ["CameraPath", "make_path", "PATH_KINDS", "DEFAULT_FOCUS_M"]
