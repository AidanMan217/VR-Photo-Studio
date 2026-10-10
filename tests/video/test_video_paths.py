"""Camera paths: periodicity, smoothness, per-kind geometry and keyframe interpolation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vrphoto.core.errors import InputError
from vrphoto.core.types import Pose
from vrphoto.video.paths import DEFAULT_FOCUS_M, PATH_KINDS, CameraPath, make_path

KEYFRAMES = [
    {"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
    {"t": 0.3, "x": 0.05, "yaw": 3.0},
    {"t": 0.7, "y": 0.03, "z": 0.04, "pitch": -2.0},
    {"t": 1.0, "x": 0.0},
]


def _path(kind: str, n: int, loop: bool = True, **kw) -> CameraPath:
    if kind == "custom":
        kw.setdefault("keyframes", KEYFRAMES)
    return make_path(kind, n_frames=n, loop=loop, **kw)


def _yaw_deg(p: Pose) -> float:
    return math.degrees(math.atan2(p.R[0, 2], p.R[2, 2]))


def _state(path: CameraPath) -> np.ndarray:
    """N x 12 array: translation + flattened rotation (for smoothness checks)."""
    return np.stack([np.concatenate([np.asarray(p.t), np.asarray(p.R).ravel() * 0.05]) for p in path.poses])


def test_kinds_list_matches_contract():
    assert PATH_KINDS == ["dolly", "circle", "swing", "zoom", "kenburns", "orbit", "custom"]


@pytest.mark.parametrize("kind", PATH_KINDS)
@pytest.mark.parametrize("loop", [True, False])
def test_basic_shape(kind, loop):
    p = _path(kind, 48, loop, fps=24.0)
    assert isinstance(p, CameraPath)
    assert p.n_frames == len(p) == 48
    assert p.fps == 24.0
    assert p.duration == pytest.approx(2.0)
    for pose in p.poses:
        R = np.asarray(pose.R)
        assert R.shape == (3, 3) and np.asarray(pose.t).shape == (3,)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", PATH_KINDS)
def test_loop_is_periodic_and_consistent(kind):
    """loop=True: sample i of an N-frame path == sample 2i of a 2N-frame path, and the
    wrap from the last frame back to frame 0 is as smooth as any other step."""
    n = 40
    a, b = _state(_path(kind, n)), _state(_path(kind, 2 * n))
    np.testing.assert_allclose(a, b[::2], atol=1e-12)
    # midpoint between frame n-1 and frame n (== frame 0) lies on the curve (no jump)
    mid = 0.5 * (a[-1] + a[0])
    assert np.abs(b[-1] - mid).max() < 2e-3
    steps = np.linalg.norm(np.diff(np.vstack([a, a[:1]]), axis=0), axis=1)
    assert steps.max() <= 1.6 * np.median(steps[steps > 0]) + 1e-9 or kind == "custom"
    wrap = np.linalg.norm(a[0] - a[-1])
    assert wrap <= 1.05 * steps[:-1].max() + 1e-12


@pytest.mark.parametrize("kind", PATH_KINDS)
@pytest.mark.parametrize("loop", [True, False])
def test_smooth_no_jumps(kind, loop):
    n = 90
    s = _state(_path(kind, n, loop))
    seq = np.vstack([s, s[:2]]) if loop else s
    d1 = np.diff(seq, axis=0)
    d2 = np.diff(seq, n=2, axis=0)
    step = np.linalg.norm(d1, axis=1).max()
    accel = np.linalg.norm(d2, axis=1).max()
    assert step < 0.012  # amplitude 5 cm over 90 frames
    assert accel < 0.25 * step + 1e-9  # second differences small vs first -> no kinks


@pytest.mark.parametrize("kind", [k for k in PATH_KINDS if k != "circle"])
def test_non_loop_eases_in_and_out(kind):
    s = _path(kind, 101, loop=False).translations()
    steps = np.linalg.norm(np.diff(s, axis=0), axis=1)
    assert steps.max() > 0
    assert steps[0] < 0.1 * steps.max()
    assert steps[-1] < 0.1 * steps.max()


def test_dolly_moves_along_z_only():
    T = make_path("dolly", n_frames=60, amplitude_m=0.08).translations()
    assert np.abs(T[:, :2]).max() == 0
    assert T[:, 2].min() == pytest.approx(-0.08)
    assert T[:, 2].max() == pytest.approx(0.08, abs=1e-4)
    T2 = make_path("dolly", n_frames=60, amplitude_m=0.08, loop=False).translations()
    assert T2[0, 2] == pytest.approx(-0.08) and T2[-1, 2] == pytest.approx(0.08)
    assert np.all(np.diff(T2[:, 2]) >= -1e-12)


def test_circle_radius_and_plane():
    p = make_path("circle", n_frames=72, amplitude_m=0.03)
    T = p.translations()
    np.testing.assert_allclose(np.linalg.norm(T[:, :2], axis=1), 0.03, atol=1e-12)
    assert np.abs(T[:, 2]).max() == 0
    assert all(np.allclose(q.R, np.eye(3)) for q in p.poses)


def test_swing_yaws_toward_focus():
    focus = 1.5
    p = make_path("swing", n_frames=40, amplitude_m=0.06, focus_m=focus)
    T = p.translations()
    assert np.abs(T[:, 1]).max() == 0
    assert T[:, 0].max() == pytest.approx(0.06, abs=1e-3) and T[:, 0].min() == pytest.approx(-0.06, abs=1e-3)
    for pose, t in zip(p.poses, T):
        full = math.degrees(math.atan2(-t[0], focus - t[2]))
        assert _yaw_deg(pose) == pytest.approx(0.5 * full, abs=1e-9)
        if abs(t[0]) > 1e-6:
            assert np.sign(_yaw_deg(pose)) == -np.sign(t[0])  # camera right -> turn left


@pytest.mark.parametrize("focus", [None, 1.2, 3.0])
def test_orbit_keeps_focus_point_centred(focus):
    """Golden geometric check: the focus point projects to the principal point."""
    f = DEFAULT_FOCUS_M if focus is None else focus
    p = make_path("orbit", n_frames=36, amplitude_m=0.1, focus_m=focus)
    F = np.array([0.0, 0.0, f])
    for pose in p.poses:
        Pc = pose.world_to_cam(F)
        assert abs(Pc[0]) < 1e-9 and abs(Pc[1]) < 1e-9
        assert Pc[2] == pytest.approx(f, rel=1e-9)  # stays on the circle around the focus
    assert max(abs(_yaw_deg(q)) for q in p.poses) > 1.0


def test_zoom_moves_toward_focus_and_is_capped():
    p = make_path("zoom", n_frames=50, amplitude_m=0.05, loop=False, focus_m=2.0)
    z = p.translations()[:, 2]
    assert z[0] == pytest.approx(0.0) and z[-1] == pytest.approx(0.1)
    assert np.all(np.diff(z) >= -1e-12)
    capped = make_path("zoom", n_frames=50, amplitude_m=1.0, loop=False, focus_m=1.0).translations()
    assert capped[:, 2].max() == pytest.approx(0.4)
    looped = make_path("zoom", n_frames=50).translations()[:, 2]
    assert looped[0] == pytest.approx(0.0) and looped.min() >= 0.0


def test_kenburns_diagonal_with_slight_dolly():
    T = make_path("kenburns", n_frames=60, amplitude_m=0.05, loop=False).translations()
    assert T[-1, 0] - T[0, 0] == pytest.approx(0.1)
    assert T[-1, 1] - T[0, 1] == pytest.approx(-0.06)  # moves up (y is down)
    assert 0 < T[:, 2].max() <= 0.05 * 0.5 + 1e-12


def test_zero_amplitude_is_static_identity():
    for kind in PATH_KINDS[:-1]:
        for pose in make_path(kind, n_frames=10, amplitude_m=0.0).poses:
            np.testing.assert_allclose(pose.t, 0.0, atol=1e-15)
            np.testing.assert_allclose(pose.R, np.eye(3), atol=1e-12)


def test_single_frame():
    for kind in PATH_KINDS:
        assert len(_path(kind, 1)) == 1
        assert len(_path(kind, 1, loop=False)) == 1


# --------------------------------------------------------------------------- custom
def test_custom_passes_through_keyframes():
    n = 101  # t * (n - 1) is an integer for t in {0, .3, .7, 1}
    p = make_path("custom", n_frames=n, loop=False, keyframes=KEYFRAMES)
    for kf in KEYFRAMES:
        pose = p.poses[round(kf["t"] * (n - 1))]
        np.testing.assert_allclose(pose.t, [kf.get("x", 0), kf.get("y", 0), kf.get("z", 0)], atol=1e-12)
        expect = Pose.look(kf.get("yaw", 0.0), kf.get("pitch", 0.0), kf.get("roll", 0.0))
        np.testing.assert_allclose(pose.R, expect.R, atol=1e-12)


def test_custom_catmull_rom_is_c1_between_keyframes():
    kfs = [
        {"t": 0, "x": 0},
        {"t": 0.25, "x": 0.1},
        {"t": 0.5, "x": 0.0},
        {"t": 0.75, "x": -0.1},
        {"t": 1, "x": 0},
    ]
    n = 401
    x = make_path("custom", n_frames=n, loop=False, keyframes=kfs).translations()[:, 0]
    v = np.diff(x)
    # velocity continuous at interior keyframes (finite-difference tangent = 0 at the peaks)
    for t in (0.25, 0.5, 0.75):
        k = round(t * (n - 1))
        assert abs(v[k] - v[k - 1]) < 2e-4
    # at the middle keyframe the tangent is (P3 - P1) / (t3 - t1) = -0.2 / 0.5 per unit t
    k = round(0.5 * (n - 1))
    assert (x[k + 1] - x[k - 1]) / (2 / (n - 1)) == pytest.approx(-0.4, rel=0.02)


def test_custom_loop_wraps_t1_onto_t0():
    kfs = [{"t": 0, "x": 0.02}, {"t": 0.5, "x": -0.02}, {"t": 1.0, "x": 0.02}]
    T = make_path("custom", n_frames=50, loop=True, keyframes=kfs).translations()
    assert T[0, 0] == pytest.approx(0.02)
    assert T[25, 0] == pytest.approx(-0.02)
    # symmetric periodic curve -> sinusoid-like and periodic
    assert np.abs(T[1, 0] - T[-1, 0]) < 1e-12


def test_custom_missing_t_spaced_evenly_and_unsorted_input():
    a = make_path("custom", n_frames=21, loop=False, keyframes=[{"x": 0}, {"x": 0.1}, {"x": 0.2}])
    b = make_path("custom", n_frames=21, loop=False,
                  keyframes=[{"t": 1, "x": 0.2}, {"t": 0, "x": 0}, {"t": 0.5, "x": 0.1}])  # fmt: skip
    np.testing.assert_allclose(a.translations(), b.translations(), atol=1e-12)


def test_custom_single_keyframe_is_static():
    p = make_path("custom", n_frames=5, keyframes=[{"t": 0.4, "x": 0.01, "yaw": 2}])
    for pose in p.poses:
        np.testing.assert_allclose(pose.t, [0.01, 0, 0])
        assert _yaw_deg(pose) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "kfs",
    [
        None,
        [],
        [{"t": 0.5}, {"t": 0.5}],
        [{"t": 1.5}],
        [{"t": -0.1}],
        [{"t": 0, "speed": 3}],
        [{"t": 0, "x": "abc"}],
        [{"t": 0, "x": float("nan")}],
        ["not a dict"],
    ],
)
def test_custom_invalid_keyframes(kfs):
    with pytest.raises(InputError):
        make_path("custom", n_frames=10, keyframes=kfs)


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "spiral", "n_frames": 10},
        {"kind": "dolly", "n_frames": 0},
        {"kind": "dolly", "n_frames": "x"},
        {"kind": "dolly", "n_frames": 10**6},
        {"kind": "dolly", "n_frames": 10, "amplitude_m": -1},
        {"kind": "dolly", "n_frames": 10, "fps": 0},
        {"kind": "orbit", "n_frames": 10, "focus_m": 0},
        {"kind": "orbit", "n_frames": 10, "focus_m": float("inf")},
    ],
)
def test_invalid_arguments(kwargs):
    kind = kwargs.pop("kind")
    with pytest.raises(InputError):
        make_path(kind, **kwargs)


def test_kind_is_case_insensitive():
    assert len(make_path(" Circle ", n_frames=3)) == 3


def test_loop_start_positions_and_direction():
    """Pins where looping stock paths start (frame 0) and which way they move first."""
    a = 0.05
    T = {k: make_path(k, n_frames=40, amplitude_m=a).translations() for k in PATH_KINDS[:-1]}
    np.testing.assert_allclose(T["dolly"][0], [0, 0, -a], atol=1e-12)  # pulled back, moves in
    assert T["dolly"][1, 2] > T["dolly"][0, 2]
    np.testing.assert_allclose(T["circle"][0], [a, 0, 0], atol=1e-12)  # right, then down (+y)
    assert T["circle"][1, 1] > 0
    for k in ("swing", "orbit"):
        np.testing.assert_allclose(T[k][0], 0, atol=1e-12)  # centred, then to the right
        assert T[k][1, 0] > 0
    np.testing.assert_allclose(T["zoom"][0], 0, atol=1e-12)
    np.testing.assert_allclose(T["kenburns"][0], [-a, 0.6 * a, 0], atol=1e-12)


def test_custom_keyframe_axis_conventions():
    """x right, y down, z forward (cv frame); yaw > 0 turns right, pitch > 0 looks up."""
    kf = [{"t": 0, "x": 0.01, "y": 0.02, "z": 0.03, "yaw": 10.0}]
    pose = make_path("custom", n_frames=1, loop=False, keyframes=kf).poses[0]
    np.testing.assert_allclose(pose.t, [0.01, 0.02, 0.03])
    fwd = np.asarray(pose.R)[:, 2]  # camera forward axis in the source frame
    assert fwd[0] > 0 and abs(fwd[1]) < 1e-12  # yaw right
    up = make_path("custom", n_frames=1, loop=False, keyframes=[{"t": 0, "pitch": 10.0}]).poses[0]
    assert np.asarray(up.R)[1, 2] < 0  # pitch up: forward moves toward -y (up in a y-down frame)
