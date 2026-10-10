"""Main window: open/close, inputs (drag & drop API), modes, job submission, cancel,
queue widget, "View in headset", state persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gui_helpers import BlockingRunner, failing_runner
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent

from vrphoto.gui import options_meta as om
from vrphoto.pipeline.jobs import JOB_TYPES, job_from_dict, job_to_dict

pytestmark = pytest.mark.gui


def test_window_opens_and_closes(make_window, qtbot):
    w, q = make_window()
    assert w.isVisible()
    assert w.windowTitle() == "VR Photo Studio"
    assert set(w.modes.cards) == set(JOB_TYPES)
    assert w.modes.current_kind() == "stereo"
    assert not w.modes.create_button.isEnabled()  # no inputs yet
    w.close()
    assert not w.isVisible()
    assert w._shut_down


def test_add_files_and_folders(make_window, media):
    w, _ = make_window()
    n = w.add_inputs([media["a"], media["txt"]])
    assert n == 1  # text file skipped
    assert w.inputs.paths() == [media["a"]]
    n = w.add_inputs([media["dir"]])  # folder: recursive, duplicates skipped
    expected = {media["a"], media["b"], media["c"]} | ({media["video"]} if "video" in media else set())
    assert set(w.inputs.paths()) == expected
    assert n == len(expected) - 1
    assert w.add_inputs([media["a"]]) == 0
    assert w.inputs.current_path() == media["b"]  # first newly added item (a.jpg was already there)
    assert w.modes.create_button.isEnabled()


def test_drag_and_drop_adds_files(make_window, media, qtbot):
    w, _ = make_window()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(media["a"])), QUrl.fromLocalFile(str(media["b"]))])
    ev = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.inputs.list.dropEvent(ev)
    assert set(w.inputs.paths()) == {media["a"], media["b"]}
    # dropping on the window itself works too
    mime2 = QMimeData()
    mime2.setUrls([QUrl.fromLocalFile(str(media["c"]))])
    w.dropEvent(
        QDropEvent(
            QPointF(5, 5),
            Qt.DropAction.CopyAction,
            mime2,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert media["c"] in w.inputs.paths()


def test_create_uses_the_selection(make_window, media, qtbot):
    w, q = make_window()
    w.add_inputs([media["a"], media["b"]])
    assert w.inputs.selected_paths() == [media["a"], media["b"]]  # what was just added
    assert w.inputs.current_path() == media["a"]  # previewed
    assert w.preview.current_path() == media["a"]
    assert "2 jobs" in w.modes.inputs_label.text()
    w.add_inputs([media["c"]])
    assert w.inputs.selected_paths() == [media["c"]]
    assert "1 job" in w.modes.inputs_label.text()
    w.inputs.list.clearSelection()
    assert w.inputs.selected_paths() == [media["a"], media["b"], media["c"]]  # nothing selected = all
    assert "3 jobs" in w.modes.inputs_label.text()


def test_remove_and_clear(make_window, media):
    w, _ = make_window()
    w.add_inputs([media["a"], media["b"]])
    w.inputs.list.setCurrentRow(0)
    w.inputs.list.item(0).setSelected(True)
    w.inputs.remove_selected()
    assert w.inputs.paths() == [media["b"]]
    w.inputs.clear()
    assert w.inputs.count() == 0
    assert w.preview.current_path() is None


def test_thumbnails_load_in_background(make_window, media, qtbot):
    w, _ = make_window()
    w.add_inputs([media["a"]])
    item = w.inputs.list.item(0)
    first = item.icon().cacheKey()
    qtbot.waitUntil(lambda: item.icon().cacheKey() != first, timeout=5000)


@pytest.mark.parametrize("kind", sorted(JOB_TYPES))
def test_every_mode_builds_a_round_tripping_spec(make_window, media, kind):
    w, _ = make_window()
    w.modes.set_mode(kind)
    assert w.modes.current_kind() == kind
    assert w.modes.cards[kind].selected
    panel = w.modes.current_panel()
    assert panel.kind == kind
    mode = om.MODE_BY_KIND[kind]
    src = [Path("C:/clips/v.mp4")] if mode.accepts == frozenset({"video"}) else [media["a"], media["b"]]
    specs = w.modes.build_specs(src)
    assert len(specs) == (1 if mode.batch else len(src))
    for spec in specs:
        assert spec.kind == kind
        d = job_to_dict(spec)
        json.dumps(d)
        assert job_from_dict(d) == spec


def test_switching_modes_builds_all_panels(make_window):
    w, _ = make_window()
    for m in om.MODES:
        w.modes.set_mode(m.kind)
    assert set(w.modes.panels) == set(JOB_TYPES)
    assert w.modes.create_button.text() == f"Create {om.MODE_BY_KIND['depth'].title}"


def test_submit_job_updates_queue_widget(make_window, media, qtbot, settings):
    w, q = make_window()
    w.add_inputs([media["a"], media["b"]])
    w.inputs.select_all()
    w.modes.set_mode("stereo")
    progress: list[float] = []
    w.bridge.jobProgress.connect(lambda j, f, m: progress.append(f))
    with qtbot.waitSignals([w.bridge.jobFinished, w.bridge.jobFinished], timeout=10000):
        ids = w.create_jobs()
    assert len(ids) == 2 and len(q.submitted) == 2
    assert [s.inputs for s in q.submitted] == [[str(media["a"])], [str(media["b"])]]
    assert progress, "progress signals were emitted"
    qtbot.waitUntil(lambda: all(w.queue.jobs[i].status == "finished" for i in ids), timeout=5000)
    row = w.queue.rows[ids[0]]
    assert row.bar.value() == 1000
    assert row.status.text().startswith("Done")
    assert not row.cancel_button.isVisible()
    assert row.open_button.isVisible() and row.view_button.isVisible()
    assert w.queue.output_files(ids[0])[0].name == "a_stereo_LR.jpg"
    assert w.queue.output_folder(ids[0]) == Path(settings.output_dir)


def test_progress_and_state_signals_reach_rows(make_window, media, qtbot):
    runner = BlockingRunner()
    w, q = make_window(runner)
    w.add_inputs([media["a"]])
    ids = w.create_jobs()
    assert runner.started.wait(5)
    row = w.queue.rows[ids[0]]
    qtbot.waitUntil(lambda: row.info.status == "running" and row.bar.value() == 100, timeout=5000)
    assert row.cancel_button.isVisible()
    assert "Waiting" in row.status.text()  # the runner's message
    runner.release.set()
    qtbot.waitUntil(lambda: row.info.status == "finished", timeout=5000)


def test_cancel_job(make_window, media, qtbot):
    runner = BlockingRunner()
    w, q = make_window(runner)
    w.add_inputs([media["a"], media["b"]])
    w.inputs.select_all()
    ids = w.create_jobs()
    assert runner.started.wait(5)
    second = w.queue.rows[ids[1]]
    with qtbot.waitSignal(w.bridge.jobCancelled, timeout=5000):
        second.cancel_button.click()
    first = w.queue.rows[ids[0]]
    with qtbot.waitSignal(w.bridge.jobCancelled, timeout=5000):
        first.cancel_button.click()
    qtbot.waitUntil(
        lambda: first.info.status == "cancelled" and second.info.status == "cancelled", timeout=5000
    )
    assert not first.cancel_button.isVisible()
    w.queue.clear_finished()
    assert w.queue.list.count() == 0


def test_cancel_all(make_window, media, qtbot):
    runner = BlockingRunner()
    w, q = make_window(runner)
    w.add_inputs([media["a"], media["b"], media["c"]])
    w.inputs.select_all()
    ids = w.create_jobs()
    assert runner.started.wait(5)
    w.queue.cancel_all_button.click()
    qtbot.waitUntil(lambda: all(w.queue.jobs[i].status == "cancelled" for i in ids), timeout=5000)
    assert w.queue.active_count() == 0


def test_failed_job_shows_error(make_window, media, qtbot, prompts):
    w, q = make_window(failing_runner)
    w.add_inputs([media["a"]])
    with qtbot.waitSignal(w.bridge.jobFailed, timeout=5000):
        ids = w.create_jobs()
    row = w.queue.rows[ids[0]]
    qtbot.waitUntil(lambda: row.info.status == "failed", timeout=5000)
    assert "out of memory" in row.status.text()
    assert row.error_button.isVisible()
    row.error_button.click()
    assert any("out of memory" in e for e in prompts.errors)


def test_video_mode_skips_photos(make_window, media, prompts):
    w, q = make_window()
    w.add_inputs([media["a"]])
    w.modes.set_mode("video")
    assert not w.modes.create_button.isEnabled()
    assert w.create_jobs() == []
    assert prompts.infos and "videos" in prompts.infos[-1]
    assert q.submitted == []


def test_invalid_options_show_error(make_window, media, prompts):
    w, q = make_window()
    w.add_inputs([media["a"]])
    w.modes.set_mode("scene")
    w.modes.current_panel().common.set_values({"near_m": 30.0, "far_m": 10.0})
    assert w.create_jobs() == []
    assert any("Nearest" in e for e in prompts.errors)
    assert q.submitted == []


def test_noncommercial_model_requires_license(make_window, media, prompts, settings):
    w, q = make_window()
    w.add_inputs([media["a"]])
    w.modes.current_panel().common.set_values({"depth_model": "da2-large"})
    prompts.license_answer = False
    assert w.create_jobs() == []
    assert prompts.licenses and q.submitted == []
    prompts.license_answer = True
    assert len(w.create_jobs()) == 1
    assert settings.license_accepted("da2-large")
    n = len(prompts.licenses)
    w.create_jobs()
    assert len(prompts.licenses) == n  # not asked again


def test_capture_needs_addon(make_window, media, prompts, monkeypatch):
    w, q = make_window()
    w.add_inputs([media["a"], media["b"]])
    w.inputs.select_all()
    w.modes.set_mode("capture")
    prompts.answer = False
    assert w.create_jobs() == []
    assert any("add-on" in t.lower() for t in prompts.questions)
    assert "addons" not in w.dialogs
    prompts.answer = True
    assert w.create_jobs() == []
    assert "addons" in w.dialogs  # offered to open the Add-ons window
    w.dialogs["addons"].close()
    # with the add-on installed: one job with all inputs
    monkeypatch.setattr(w, "_check_addons", lambda kind, values: True)
    ids = w.create_jobs()
    assert len(ids) == 1
    assert q.submitted[-1].inputs == [str(media["a"]), str(media["b"])]


def test_view_in_headset_opens_quest_dialog(make_window, media, qtbot):
    w, q = make_window()
    w.add_inputs([media["a"]])
    with qtbot.waitSignal(w.bridge.jobFinished, timeout=5000):
        ids = w.create_jobs()
    row = w.queue.rows[ids[0]]
    qtbot.waitUntil(lambda: row.view_button.isVisible(), timeout=5000)
    row.view_button.click()
    dlg = w.dialogs["quest"]
    assert dlg.focus_scene == "a-scene"
    assert [f.name for f in dlg.files] == ["a_stereo_LR.jpg"]
    qtbot.waitUntil(lambda: w.server.running, timeout=5000)
    qtbot.waitUntil(lambda: "a-scene" in dlg.scene_label.text(), timeout=5000)
    assert "view.html?scene=a-scene" in dlg.current_url()
    # opening again re-uses the same dialog
    assert w.open_quest() is dlg
    dlg.close()
    qtbot.waitUntil(lambda: "quest" not in w.dialogs, timeout=5000)


def test_close_with_running_jobs_asks(make_window, media, qtbot, prompts):
    runner = BlockingRunner()
    w, q = make_window(runner)
    w.add_inputs([media["a"]])
    w.create_jobs()
    assert runner.started.wait(5)
    qtbot.waitUntil(lambda: w.queue.active_count() == 1, timeout=5000)
    prompts.answer = False
    w.close()
    assert w.isVisible() and not w._shut_down
    prompts.answer = True
    w.close()
    assert not w.isVisible() and w._shut_down
    runner.release.set()


def test_close_stops_server(make_window, qtbot, fake_server_factory):
    w, _ = make_window()
    w.server.start()
    qtbot.waitUntil(lambda: w.server.running, timeout=5000)
    srv = fake_server_factory.instances[-1]
    w.close()
    assert srv.stops == 1 and not srv.running


def test_state_is_saved_and_restored(make_window, media, qtbot, isolated_home):
    from vrphoto.gui.state import GuiState

    w, _ = make_window()
    w.modes.set_mode("vr180")
    w.modes.current_panel().form.set_values({"size_mult": 1.4, "fill": "black"})
    w.close()
    st = GuiState.load()
    assert st.last_mode == "vr180"
    assert st.options["vr180"]["size_mult"] == pytest.approx(1.4)
    assert st.geometry
    w2, _ = make_window(state=GuiState.load())
    assert w2.modes.current_kind() == "vr180"
    spec = w2.modes.build_specs([media["a"]])[0]
    assert spec.size_mult == pytest.approx(1.4) and spec.fill == "black"


def test_settings_change_updates_option_defaults(make_window, settings):
    w, _ = make_window()
    w.modes.set_mode("stereo")
    settings.depth_model = "moge2"
    w.apply_settings({"depth_model"})
    assert w.modes.current_panel().common_values()["depth_model"] == "moge2"
    w.modes.set_mode("scene")  # panels created later use the new default too
    assert w.modes.current_panel().common_values()["depth_model"] == "moge2"


def test_log_panel_shows_messages(make_window, qtbot):
    from vrphoto.core.log import get_logger

    w, _ = make_window()
    get_logger("vrphoto.test").warning("Hello from the test")
    qtbot.waitUntil(lambda: "Hello from the test" in w.log_panel.text(), timeout=5000)


def test_dialogs_open_from_window(make_window, qtbot, monkeypatch):
    w, _ = make_window()
    for opener, name in (
        (w.open_about, "about"),
        (w.open_log, "log"),
        (w.open_settings, "settings"),
        (w.open_models, "models"),
    ):
        dlg = opener()
        assert w.dialogs[name] is dlg and dlg.isVisible()
        assert opener() is dlg  # single instance
        dlg.close()
        qtbot.waitUntil(lambda n=name: n not in w.dialogs, timeout=5000)
