"""vrphoto.pipeline package surface: lazy exports and cheap imports."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_lazy_exports():
    import vrphoto.pipeline as p

    assert set(p.__all__) <= set(dir(p))
    from vrphoto.pipeline.queue import JobQueue
    from vrphoto.pipeline.run import run_job

    assert p.run_job is run_job and p.JobQueue is JobQueue
    assert p.output_name("x", "vr180", "jpg") == "x_180_LR.jpg"
    with pytest.raises(AttributeError):
        _ = p.does_not_exist


def test_queue_import_is_light():
    """The GUI imports the queue at start-up: it must not pull in geometry/video/webxr/addons."""
    code = (
        "import sys, vrphoto.pipeline.queue, vrphoto.pipeline.run\n"
        "heavy = [m for m in ('vrphoto.geometry.warp', 'vrphoto.video.encode', 'vrphoto.webxr.server',"
        " 'vrphoto.addons.manager', 'onnxruntime', 'PySide6') if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == ""
