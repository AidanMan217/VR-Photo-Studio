"""End-to-end: the GUI with the real JobQueue and pipeline (no models: heuristic depth,
OpenCV inpainting, GrabCut cut-outs) for every mode that needs no add-on, plus "View in
headset" against the real viewer server."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gui

#: small sizes so every mode finishes in a few seconds on a CPU
FAST_OPTIONS: dict[str, dict] = {
    "parallax": {"duration_s": 2.0, "fps": 24.0},
    "vr180": {"eye_px": 1024},
    "pano": {"width": 2048},
    "scene": {"target_faces": 50_000},
    "video": {"max_long": 480},
}


@pytest.fixture
def real_window(qtbot, settings, heuristic_engine, fake_server_factory):
    pytest.importorskip("vrphoto.pipeline.queue")
    pytest.importorskip("vrphoto.pipeline.run")
    from vrphoto.gui.main_window import MainWindow
    from vrphoto.gui.state import GuiState

    settings.depth_model = "heuristic"
    settings.inpaint_model = "opencv"
    settings.segment_model = "grabcut"
    settings.max_process_px = 640
    w = MainWindow(
        settings, preview_engine=heuristic_engine, server_factory=fake_server_factory, state=GuiState()
    )  # queue=None: created lazily by the bridge (real JobQueue + run_job)
    qtbot.addWidget(w)
    w.show()
    yield w
    w.shutdown()


@pytest.fixture
def pano_photo(tmp_path) -> Path:
    from PIL import Image

    img = np.zeros((256, 512, 3), np.uint8)
    img[..., 1] = 120
    img[100:150, 200:300] = 250
    p = tmp_path / "pano360.jpg"
    Image.fromarray(img).save(p)
    return p


def _run(w, qtbot, kind: str, inputs: list[Path]):
    w.inputs.clear()
    w.add_inputs(inputs)
    w.modes.set_mode(kind)
    if kind in FAST_OPTIONS:
        w.modes.current_panel().form.set_values(FAST_OPTIONS[kind])
    ids = w.create_jobs()
    assert len(ids) == 1
    info = w.queue.jobs[ids[0]]
    qtbot.waitUntil(lambda: info.status not in ("queued", "running"), timeout=180_000)
    assert info.status == "finished", info.error
    return ids[0], info


@pytest.mark.parametrize("kind", ["depth", "stereo", "parallax", "vr180", "pano", "scene", "object", "video"])
def test_real_pipeline_job_from_gui(real_window, media, pano_photo, qtbot, settings, kind):
    if kind == "video" and "video" not in media:
        pytest.skip("OpenCV cannot write MP4 here")
    inputs = [media["video"]] if kind == "video" else [pano_photo] if kind == "pano" else [media["a"]]
    w = real_window
    job_id, info = _run(w, qtbot, kind, inputs)
    files = w.queue.output_files(job_id)
    assert files and all(f.is_file() for f in files)
    assert all(Path(settings.output_dir) in f.parents for f in files)
    assert w.queue.output_folder(job_id) == Path(settings.output_dir)
    row = w.queue.rows[job_id]
    if kind == "depth":
        assert info.scene_id is None and not row.view_button.isVisible()
    else:  # every other mode is added to the viewer library by default
        assert info.scene_id and (Path(settings.library_dir) / info.scene_id / "scene.json").is_file()
        assert row.view_button.isVisible()


def test_view_in_headset_with_real_server(qtbot, settings, heuristic_engine, media):
    """A finished scene is served by the real viewer server at the address the dialog shows."""
    pytest.importorskip("vrphoto.pipeline.run")
    pytest.importorskip("vrphoto.webxr.server")
    from vrphoto.gui.main_window import MainWindow
    from vrphoto.gui.state import GuiState

    settings.depth_model = "heuristic"
    settings.inpaint_model = "opencv"
    settings.https_port = 0  # free ports
    settings.http_port = 0
    w = MainWindow(settings, preview_engine=heuristic_engine, state=GuiState())
    qtbot.addWidget(w)
    w.show()
    try:
        job_id, info = _run(w, qtbot, "scene", [media["a"]])
        dlg = w.view_in_headset(job_id, info.scene_id)
        qtbot.waitUntil(lambda: w.server.running or bool(w.server.last_error), timeout=30_000)
        if not w.server.running:
            pytest.skip(f"viewer server could not start here: {w.server.last_error}")
        qtbot.waitUntil(lambda: info.scene_id in dlg.scene_label.text(), timeout=5000)
        assert [f.suffix for f in dlg.files] == [".glb"]
        local = w.server.local_url()
        with urllib.request.urlopen(local + "api/scenes", timeout=10) as r:
            assert info.scene_id in [s.get("id") for s in json.loads(r.read())]
        with urllib.request.urlopen(w.server.scene_url(info.scene_id, base=local), timeout=10) as r:
            assert r.status == 200
        with urllib.request.urlopen(f"{local}scenes/{info.scene_id}/scene.json", timeout=10) as r:
            assert json.loads(r.read())["mode"] == "photo3d"
    finally:
        w.shutdown()
