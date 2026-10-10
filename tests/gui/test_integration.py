"""End-to-end: the GUI with the real JobQueue and pipeline (no models: heuristic depth)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.gui


@pytest.fixture
def real_window(qtbot, settings, heuristic_engine, fake_server_factory):
    pytest.importorskip("vrphoto.pipeline.queue")
    pytest.importorskip("vrphoto.pipeline.run")
    from vrphoto.gui.main_window import MainWindow
    from vrphoto.gui.state import GuiState

    settings.depth_model = "heuristic"
    settings.inpaint_model = "opencv"
    w = MainWindow(
        settings, preview_engine=heuristic_engine, server_factory=fake_server_factory, state=GuiState()
    )  # queue=None: created lazily by the bridge (real JobQueue + run_job)
    qtbot.addWidget(w)
    w.show()
    yield w
    w.shutdown()


@pytest.mark.parametrize("kind", ["depth", "stereo"])
def test_real_pipeline_job_from_gui(real_window, media, qtbot, settings, kind):
    w = real_window
    w.add_inputs([media["a"]])
    w.modes.set_mode(kind)
    ids = w.create_jobs()
    assert len(ids) == 1
    info = w.queue.jobs[ids[0]]
    qtbot.waitUntil(lambda: info.status not in ("queued", "running"), timeout=120_000)
    assert info.status == "finished", info.error
    files = w.queue.output_files(ids[0])
    assert files and all(f.is_file() for f in files)
    assert all(Path(settings.output_dir) in f.parents for f in files)
    assert w.queue.output_folder(ids[0]) == Path(settings.output_dir)
