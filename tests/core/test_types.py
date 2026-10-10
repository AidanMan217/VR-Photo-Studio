import math

import numpy as np
import pytest

from vrphoto.core.errors import Cancelled
from vrphoto.core.types import CancelToken, DepthScale, Intrinsics, Pose, ProgressReporter


def test_intrinsics_hfov_roundtrip():
    K = Intrinsics.from_hfov(640, 480, 65.0)
    assert math.isclose(K.hfov_deg, 65.0, rel_tol=1e-9)
    K2 = K.scaled(0.5)
    assert K2.width == 320 and K2.height == 240
    assert math.isclose(K2.hfov_deg, 65.0, rel_tol=1e-6)


def test_pose_conventions():
    # moving the camera right: a point straight ahead ends up at negative x in the new camera
    p = Pose.translate(0.1, 0, 0)
    P = np.array([[0.0, 0.0, 2.0]])
    assert p.world_to_cam(P)[0, 0] == pytest.approx(-0.1)
    # yaw right: forward axis points toward +x
    r = Pose.look(yaw_deg=90)
    assert np.allclose(r.R[:, 2], [1, 0, 0], atol=1e-9)
    # pitch up: forward axis points toward -y (y is down)
    u = Pose.look(pitch_deg=90)
    assert np.allclose(u.R[:, 2], [0, -1, 0], atol=1e-9)
    assert np.allclose(u.R.T @ u.R, np.eye(3))


def test_depth_scale():
    ds = DepthScale(1.0, 10.0)
    inv = ds.to_inv_z(np.array([0.0, 1.0], np.float32))
    assert inv[0] == pytest.approx(0.1) and inv[1] == pytest.approx(1.0)


def test_progress_sub_and_cancel():
    seen = []
    pr = ProgressReporter(lambda f, m: seen.append(round(f, 3)))
    pr.sub(0.5, 1.0)(0.5, "x")
    assert seen == [0.75]
    tok = CancelToken()
    tok.check()
    tok.cancel()
    with pytest.raises(Cancelled):
        tok.check()


def test_jobs_roundtrip():
    from vrphoto.pipeline.jobs import StereoJob, job_from_dict, job_to_dict

    j = StereoJob(inputs=["a.jpg"], layout="tb", strength=1.5)
    j.common.near_m = 0.8
    d = job_to_dict(j)
    j2 = job_from_dict(d)
    assert isinstance(j2, StereoJob) and j2.layout == "tb" and j2.common.near_m == 0.8


def test_settings_roundtrip(isolated_home):
    from vrphoto.core.config import Settings, data_root

    assert str(data_root()).startswith(str(isolated_home))
    s = Settings.load()
    s.device = "cpu"
    s.accept_license("da2-large")
    s.save()
    s2 = Settings.load()
    assert s2.device == "cpu" and s2.license_accepted("da2-large")
    assert s2.models_dir.is_dir()


def test_imaging_exif_fov(gradient_photo):
    from vrphoto.core.imaging import exif_hfov, load_image

    img = load_image(gradient_photo)
    assert img.shape == (480, 640, 3)
    fov = exif_hfov(gradient_photo)
    assert fov is not None and 60 < fov < 75  # 26mm equiv ~ 69 deg


def test_push_pull_fill():
    from vrphoto.core.fill import push_pull_fill

    img = np.zeros((64, 80, 3), np.uint8)
    img[:, :40] = (200, 10, 10)
    img[:, 40:] = (10, 10, 200)
    known = np.ones((64, 80), bool)
    known[20:44, 30:50] = False
    holey = img.copy()
    holey[~known] = 0
    out = push_pull_fill(holey, known)
    assert out.dtype == np.uint8
    assert np.array_equal(out[known], img[known])
    # filled pixels are a blend of the two colours, never black
    assert out[~known].min(axis=0).max() > 5 and out[~known][:, 0].mean() > 30
    # grayscale + float path
    g = push_pull_fill(np.ones((10, 10), np.float32), np.eye(10, dtype=bool))
    assert np.allclose(g, 1.0, atol=1e-5)
