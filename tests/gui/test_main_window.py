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


def test_output_folder_change_does_not_restart_viewer(make_window, qtbot, fake_server_factory):
    """The viewer does not serve the output folder (SEC-1), so changing it keeps the viewer
    running; changing a port restarts it."""
    w, _ = make_window()
    w.server.start()
    qtbot.waitUntil(lambda: w.server.running and not w.server.busy, timeout=5000)
    first = fake_server_factory.instances[-1]
    w.apply_settings({"output_dir"})
    assert w.server.running and not w.server.busy and first.stops == 0
    w.apply_settings({"http_port"})
    qtbot.waitUntil(lambda: first.stops == 1 and w.server.running and not w.server.busy, timeout=5000)
    assert fake_server_factory.instances[-1] is not first


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


# --------------------------------------------------------------------------- review fixes
def test_preview_asks_for_noncommercial_license_once(make_window, media, prompts, settings, qtbot):
    """The depth preview must not use a non-commercial model before its license is accepted."""
    w, _ = make_window()
    w.add_inputs([media["a"]])
    w.modes.current_panel().common.set_values({"depth_model": "da2-large"})
    prompts.license_answer = False
    assert w._preview_license_ok("da2-large") is False
    assert prompts.licenses == ["Depth Anything V2 Large"]
    assert w._preview_license_ok("da2-large") is False  # declined: not asked again this session
    assert len(prompts.licenses) == 1
    assert w._preview_license_ok("auto") and w._preview_license_ok("heuristic")  # no prompt
    assert len(prompts.licenses) == 1
    # creating a job is an explicit action: it asks again, and accepting unlocks the preview
    prompts.license_answer = True
    assert len(w.create_jobs()) == 1
    assert settings.license_accepted("da2-large")
    assert w._preview_license_ok("da2-large") is True
    assert len(prompts.licenses) == 2


def test_preview_tab_respects_license_gate(make_window, media, prompts, qtbot):
    from vrphoto.gui.panels.preview import TAB_DEPTH

    w, _ = make_window()
    w.add_inputs([media["a"]])
    w.modes.current_panel().common.set_values({"depth_model": "da2-base"})
    prompts.license_answer = False
    w.preview.set_current_tab(TAB_DEPTH)
    assert prompts.licenses == ["Depth Anything V2 Base"]
    assert "non-commercial" in w.preview.views[TAB_DEPTH].message()
    assert "depth" not in w.preview._running


def test_output_name_is_not_remembered_across_sessions(make_window, isolated_home):
    from vrphoto.gui.state import GuiState

    w, _ = make_window()
    w.modes.current_panel().common.set_values({"name": "beach", "near_m": 2.0})
    w.close()
    st = GuiState.load()
    assert "name" not in st.options["stereo"]["common"]
    assert st.options["stereo"]["common"]["near_m"] == pytest.approx(2.0)


def test_options_are_saved_when_jobs_are_created(make_window, media, isolated_home):
    from vrphoto.gui.state import GuiState

    w, _ = make_window()
    w.add_inputs([media["a"]])
    w.modes.current_panel().form.set_values({"strength": 1.35})
    assert w.create_jobs()
    assert GuiState.load().options["stereo"]["strength"] == pytest.approx(1.35)  # before any exit


def test_closing_waits_for_cancelled_jobs_to_stop(make_window, media, qtbot, prompts):
    """On exit the running job is cancelled AND given time to clean up (subprocesses)."""
    import threading
    import time

    from vrphoto.pipeline.jobs import JobResult

    stopped = threading.Event()

    def runner(spec, ctx=None, *, progress=None, cancel=None):
        try:
            while True:
                cancel.check()
                time.sleep(0.01)
        finally:
            time.sleep(0.2)  # e.g. killing an add-on worker process
            stopped.set()
        return JobResult()

    w, q = make_window(runner)
    w.add_inputs([media["a"]])
    w.create_jobs()
    qtbot.waitUntil(lambda: w.queue.active_count() == 1 and q.jobs()[0].status == "running", timeout=5000)
    prompts.answer = True
    w.close()
    assert stopped.is_set()  # shutdown waited for the job's clean-up
    assert q.jobs()[0].status == "cancelled"


def test_bridge_shutdown_wait_is_bounded(qtbot):
    import time

    from gui_helpers import FakeJobQueue

    from vrphoto.gui.bridge import QueueBridge
    from vrphoto.pipeline.jobs import StereoJob

    release = __import__("threading").Event()

    def stubborn(spec, ctx=None, *, progress=None, cancel=None):
        release.wait(10)  # ignores cancellation

    q = FakeJobQueue(stubborn)
    b = QueueBridge(q)
    b.submit(StereoJob(inputs=["x.jpg"]))
    qtbot.waitUntil(lambda: q.jobs()[0].status == "running", timeout=5000)
    t0 = time.monotonic()
    b.shutdown(wait_s=0.3)
    assert time.monotonic() - t0 < 2.0
    release.set()
    q.shutdown(wait=True)


def test_addons_experimental_toggle_updates_option_choices(make_window, qtbot, monkeypatch, settings):
    from vrphoto.gui.dialogs import addons_dialog

    monkeypatch.setattr(addons_dialog, "detect_gpu", lambda: {"name": None, "vram_mb": None, "nvidia": False})
    w, _ = make_window()
    w.modes.set_mode("object")
    method = w.modes.current_panel().form.editors["method"]
    assert "sharp" not in method.values()
    dlg = w.open_addons()
    dlg.exp_check.setChecked(True)
    assert settings.show_experimental
    assert "sharp" in method.values()  # the main window's option panels follow
    dlg.close()


def test_big_folder_notice(make_window, media, monkeypatch, qtbot):
    from vrphoto.gui.panels import input_list

    monkeypatch.setattr(input_list, "MAX_FILES", 2)
    w, _ = make_window()
    n = w.add_inputs([media["dir"]])
    assert n == 2 and w.inputs.count() == 2 and w.inputs.last_truncated
    assert "Stopped after 2 files" in w.statusBar().currentMessage()
    assert w.add_inputs([media["a"]]) == 0
    assert "list is full" in w.statusBar().currentMessage()


def test_thumbnail_of_removed_item_is_ignored(make_window, media, qtbot):
    w, _ = make_window()
    w.add_inputs([media["a"], media["b"]])
    w.inputs.clear()
    w.add_inputs([media["b"]])
    item = w.inputs.list.item(0)
    first = item.icon().cacheKey()
    qtbot.waitUntil(lambda: item.icon().cacheKey() != first, timeout=5000)
    qtbot.waitUntil(lambda: not w.inputs.tasks.busy(), timeout=5000)
    assert w.inputs.count() == 1 and len(w.inputs._items) == 1


def test_broken_dialog_shows_one_friendly_error(make_window, prompts, monkeypatch):
    from vrphoto.gui.dialogs import about_dialog

    def broken(*a, **k):
        raise RuntimeError("missing notices module")

    monkeypatch.setattr(about_dialog, "AboutDialog", broken)
    w, _ = make_window()
    assert w.open_about() is None  # no exception escapes into the Qt event loop
    assert len(prompts.errors) == 1 and "missing notices module" in prompts.errors[0]
    assert "about" not in w.dialogs


def test_typed_number_is_used_by_ctrl_enter(make_window, media, monkeypatch):
    """A value typed into a spin box (not yet confirmed with Enter) is used by Create."""
    from vrphoto.gui import main_window

    w, q = make_window()
    w.add_inputs([media["a"]])
    spin = w.modes.current_panel().form.editors["strength"].spin
    spin.lineEdit().setText("1.70 ×")  # typed, keyboard tracking off: not committed yet
    assert spin.value() == pytest.approx(1.0)
    monkeypatch.setattr(main_window.QApplication, "focusWidget", staticmethod(lambda: spin))
    ids = w.create_jobs()
    assert ids and q.submitted[-1].strength == pytest.approx(1.7)


# --------------------------------------------------------------------------- SHARP 360 add-on check
class _FakeAddons:
    """``addons.manager`` stand-in: SHARP / TRELLIS manifests, a configurable installed set."""

    TITLES = {"sharp": "SHARP (photo to splat)", "trellis": "TRELLIS (photo to 3D object)"}

    def __init__(self) -> None:
        self.installed: set[str] = set()
        self.checked: list[str] = []
        self.broken = False

    def get_manifest(self, aid: str):
        from types import SimpleNamespace

        if aid not in self.TITLES:
            raise KeyError(aid)
        return SimpleNamespace(id=aid, title=self.TITLES[aid])

    def is_installed(self, aid: str) -> bool:
        self.checked.append(aid)
        if self.broken:
            raise OSError("cannot read the add-on folder")
        return aid in self.installed


@pytest.fixture
def addons(monkeypatch) -> _FakeAddons:
    from vrphoto.gui.dialogs import common as dc

    real = dc.try_backend
    mgr = _FakeAddons()
    monkeypatch.setattr(
        dc, "try_backend", lambda name: (mgr, None) if name == "addons.manager" else real(name)
    )
    return mgr


@pytest.fixture
def choices(monkeypatch):
    """Records ``util.ask_choice`` (Open Add-ons / Continue / Cancel) and answers ``.answer``."""
    from types import SimpleNamespace

    from vrphoto.gui import util

    rec = SimpleNamespace(calls=[], answer=1)

    def ask(parent, title, text, buttons, **kw):  # noqa: ANN001
        rec.calls.append((title, text, tuple(buttons), kw))
        return rec.answer

    monkeypatch.setattr(util, "ask_choice", ask)
    return rec


def _object_window(make_window, media, monkeypatch, **values):
    """Main window in Person / Object mode with ``values`` set; open_addons is recorded."""
    w, q = make_window()
    w.add_inputs([media["a"]])
    w.modes.set_mode("object")
    w.modes.current_panel().form.set_values(values)
    opened: list[bool] = []
    monkeypatch.setattr(w, "open_addons", lambda: opened.append(True))
    return w, q, opened


def test_missing_optional_back_source_can_continue(make_window, media, monkeypatch, prompts, addons, choices):
    """design §1.7: TRELLIS missing -> "Open Add-ons / Continue / Cancel"; Continue starts the
    job as asked (the job itself warns and uses the built-in back)."""
    from vrphoto.gui.main_window import OPTIONAL_ADDON_BUTTONS

    addons.installed = {"sharp"}
    w, q, opened = _object_window(make_window, media, monkeypatch, method="sharp360", back_source="trellis")
    choices.answer = 1
    ids = w.create_jobs()
    assert len(ids) == 1 and q.submitted[-1].back_source == "trellis"
    assert len(choices.calls) == 1 and not prompts.questions and not opened
    title, text, buttons, kw = choices.calls[0]
    assert buttons == OPTIONAL_ADDON_BUTTONS
    assert buttons == ("Open Add-ons", "Continue (the built-in back is used)", "Cancel")
    assert "TRELLIS" in text and "not installed" in text
    assert kw.get("escape") == 2  # Esc / closing the box cancels


def test_missing_optional_back_source_open_addons(make_window, media, monkeypatch, prompts, addons, choices):
    addons.installed = {"sharp"}
    w, q, opened = _object_window(make_window, media, monkeypatch, method="sharp360", back_source="trellis")
    choices.answer = 0
    assert w.create_jobs() == []
    assert opened == [True] and q.submitted == [] and not prompts.questions


@pytest.mark.parametrize("answer", [2, None])
def test_missing_optional_back_source_cancel(
    make_window, media, monkeypatch, prompts, addons, choices, answer
):
    addons.installed = {"sharp"}
    w, q, opened = _object_window(make_window, media, monkeypatch, method="sharp360", back_source="trellis")
    choices.answer = answer
    assert w.create_jobs() == []
    assert not opened and q.submitted == [] and len(choices.calls) == 1


def test_missing_sharp_still_shows_the_required_dialog(
    make_window, media, monkeypatch, prompts, addons, choices
):
    """SHARP is required: today's Yes/No dialog, and no optional question for a job that
    cannot run anyway."""
    w, q, opened = _object_window(make_window, media, monkeypatch, method="sharp360", back_source="trellis")
    prompts.answer = False
    assert w.create_jobs() == []
    assert any("Add-on needed" in t and "SHARP" in t for t in prompts.questions)
    assert choices.calls == [] and not opened and q.submitted == []
    prompts.answer = True
    assert w.create_jobs() == []
    assert opened == [True] and choices.calls == []


def test_installed_or_unneeded_back_sources_ask_nothing(
    make_window, media, monkeypatch, prompts, addons, choices
):
    addons.installed = {"sharp", "trellis"}
    w, q, opened = _object_window(make_window, media, monkeypatch, method="sharp360", back_source="trellis")
    assert len(w.create_jobs()) == 1
    addons.installed = {"sharp"}
    for source in ("auto", "shell", "images"):
        extra = {"back_images": str(media["b"])} if source == "images" else {}
        w.modes.current_panel().form.set_values({"back_source": source, **extra})
        assert len(w.create_jobs()) == 1, source
    assert choices.calls == [] and not prompts.questions and not opened
    assert [s.back_source for s in q.submitted] == ["trellis", "auto", "shell", "images"]


def test_remembered_back_source_is_ignored_for_other_methods(
    make_window, media, monkeypatch, prompts, addons, choices
):
    """design §5.5: Billboard + a remembered back_source=trellis asks for nothing."""
    w, q, opened = _object_window(make_window, media, monkeypatch, method="billboard", back_source="trellis")
    assert len(w.create_jobs()) == 1
    assert addons.checked == [] and choices.calls == [] and not prompts.questions
    # the TRELLIS method needs TRELLIS for real: the required dialog, not the optional one
    w.modes.current_panel().form.set_values({"method": "trellis"})
    prompts.answer = False
    assert w.create_jobs() == []
    assert any("TRELLIS" in t for t in prompts.questions) and choices.calls == []


def test_addon_check_that_fails_lets_the_job_report(
    make_window, media, monkeypatch, prompts, addons, choices
):
    """A manager that cannot check (or unknown ids) never blocks: the pipeline reports."""
    addons.broken = True
    w, q, opened = _object_window(make_window, media, monkeypatch, method="sharp360", back_source="trellis")
    assert len(w.create_jobs()) == 1
    assert choices.calls == [] and not prompts.questions


def _answer_box(action, *, max_polls: int = 300):
    """Run ``action(box)`` on the message box once it is shown (it is modal) and make sure the
    box closes whatever happens.  Returns a record: ``done`` (the box was seen), ``texts``
    (its button labels) and ``errors``.

    The checks belong in the test body, after the dialog returns: this callback runs inside
    ``box.exec()``, where a failing assertion would leave the box open and ``exec()`` blocked
    for ever - Python's signal handlers, and so pytest-timeout, cannot run inside Qt's C++
    event loop.  So an exception in ``action`` is recorded and the box closed (``done(0)``),
    and when no message box shows up within ``max_polls`` polls (10 ms apart) whatever modal
    widget is open is closed and a failure recorded."""
    from types import SimpleNamespace

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

    rec = SimpleNamespace(done=False, texts=[], errors=[], polls=0)

    def poll() -> None:
        box = QApplication.activeModalWidget()
        if isinstance(box, QMessageBox) and box.isVisible():
            rec.done = True
            rec.texts = [b.text() for b in box.buttons()]
            try:
                action(box)
            except BaseException as e:  # noqa: BLE001 - reported by the test body
                rec.errors.append(e)
            finally:
                if box.isVisible():  # the action failed or did not answer: never leave it open
                    box.done(0)
            return
        rec.polls += 1
        if rec.polls >= max_polls:
            rec.errors.append(AssertionError(f"no message box was shown (modal widget: {box!r})"))
            if isinstance(box, QDialog):
                box.done(0)
            elif box is not None:
                box.close()
            return
        QTimer.singleShot(10, poll)

    QTimer.singleShot(0, poll)
    return rec


@pytest.mark.parametrize("label,expected", [("Open", 0), ("Continue", 1), ("Cancel", 2)])
def test_ask_choice_returns_the_clicked_button(qtbot, prompts, label, expected):
    ask_choice = prompts.real_ask_choice  # the real dialog (the prompts fixture stubs util.ask_choice)
    buttons = ("Open Add-ons", "Continue (the built-in back is used)", "Cancel")

    def click(box) -> None:
        next(b for b in box.buttons() if b.text().startswith(label)).click()

    box = _answer_box(click)
    got = ask_choice(None, "Question", "Pick one", buttons, escape=2, informative="More")
    assert not box.errors, box.errors
    assert box.done and set(box.texts) == set(buttons), box.texts
    assert got == expected


def test_ask_choice_escape_and_close(qtbot, prompts):
    from PySide6.QtCore import Qt as QtNs
    from PySide6.QtTest import QTest

    ask_choice = prompts.real_ask_choice
    box = _answer_box(lambda b: QTest.keyClick(b, QtNs.Key.Key_Escape))
    got = ask_choice(None, "Q", "Pick", ("A", "B", "Cancel"), escape=2)
    assert not box.errors and box.done and got == 2, box.errors
    box = _answer_box(lambda b: b.done(0))  # closed without a button
    got = ask_choice(None, "Q", "Pick", ("A", "B"))
    assert not box.errors and box.done and got is None, box.errors


def test_answer_box_never_leaves_a_failing_dialog_open(qtbot, prompts):
    """A check that fails inside the dialog closes it (no hang) and is reported afterwards."""
    ask_choice = prompts.real_ask_choice

    def wrong(b) -> None:
        raise AssertionError("labels changed")

    box = _answer_box(wrong)
    assert ask_choice(None, "Q", "Pick", ("A", "B")) is None
    assert box.done and [str(e) for e in box.errors] == ["labels changed"]
