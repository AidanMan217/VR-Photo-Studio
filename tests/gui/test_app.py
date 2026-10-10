"""run_gui, exception hooks, theme, GUI state and small utilities."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.gui


# --------------------------------------------------------------------------- run_gui
def test_run_gui_starts_and_exits(qtbot, monkeypatch, isolated_home, tmp_path):
    from vrphoto.gui import app as gui_app

    monkeypatch.setattr(gui_app, "setup_logging", lambda *a, **k: None)
    from vrphoto.core.config import Settings

    s = Settings()
    s.output_dir = str(tmp_path / "out")
    s.theme = "dark"
    s.save()
    photo = tmp_path / "p.jpg"
    from gui_helpers import make_image

    make_image(photo)
    seen = {}

    def fake_exec(self):
        # Stands in for the event loop: really quitting the shared test QApplication would
        # break later tests, so inspect the running window and "exit" with code 0.
        win = gui_app._main_window
        seen["visible"] = win.isVisible()
        seen["inputs"] = [p.name for p in win.inputs.paths()]
        seen["hook"] = sys.excepthook
        seen["title"] = QApplication.instance().applicationName()
        seen["win"] = win
        return 0

    monkeypatch.setattr(QApplication, "exec", fake_exec)
    old_hook = sys.excepthook
    code = gui_app.run_gui(["vrphoto", str(photo), "--not-a-file"])
    assert code == 0
    assert seen["visible"] and seen["inputs"] == ["p.jpg"]
    assert seen["hook"] is not old_hook  # our hook was installed while running
    assert sys.excepthook is old_hook  # and restored afterwards
    assert seen["title"] == "VR Photo Studio"
    assert gui_app._main_window is None
    assert seen["win"]._shut_down  # cleaned up after the event loop ended
    assert QApplication.instance().property("vrphoto_theme") == "dark"


def test_run_gui_reports_startup_failure(qtbot, monkeypatch, isolated_home):
    from vrphoto.gui import app as gui_app
    from vrphoto.gui import main_window

    monkeypatch.setattr(gui_app, "setup_logging", lambda *a, **k: None)
    shown = []
    monkeypatch.setattr(gui_app.QMessageBox, "critical", lambda *a, **k: shown.append(a))

    def boom(*a, **k):
        raise RuntimeError("cannot build window")

    monkeypatch.setattr(main_window, "MainWindow", boom)
    assert gui_app.run_gui(["vrphoto"]) == 1
    assert shown and "cannot build window" in shown[0][2]


def test_lazy_package_attribute():
    import vrphoto.gui
    from vrphoto.gui.app import run_gui

    assert vrphoto.gui.run_gui is run_gui
    with pytest.raises(AttributeError):
        vrphoto.gui.does_not_exist  # noqa: B018


def test_file_args():
    from vrphoto.gui.app import _file_args

    here = Path(__file__)
    assert _file_args([str(here), "-x", "/definitely/missing.jpg"]) == [here]


# --------------------------------------------------------------------------- exception hooks
def test_error_reporter_shows_dialog_on_ui_thread(qtbot, monkeypatch):
    from vrphoto.gui.app import ErrorReporter

    rep = ErrorReporter()
    main = threading.get_ident()
    shown: list[tuple] = []
    monkeypatch.setattr(
        rep, "show_dialog", lambda info: shown.append((info[1], threading.get_ident() == main))
    )

    def worker():
        try:
            raise ValueError("bad thing")
        except ValueError:
            rep.report(sys.exc_info())

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    qtbot.waitUntil(lambda: bool(shown), timeout=5000)
    assert str(shown[0][0]) == "bad thing" and shown[0][1] is True
    assert rep.count == 1


def test_error_reporter_one_dialog_at_a_time(qtbot, monkeypatch):
    from vrphoto.gui.app import ErrorReporter

    rep = ErrorReporter()
    shown: list[str] = []

    def show(info):
        shown.append(str(info[1]))
        rep._show((ValueError, ValueError("nested"), None))  # re-entrant report while showing

    monkeypatch.setattr(rep, "show_dialog", show)
    rep._show((ValueError, ValueError("first"), None))
    assert shown == ["first"]


def test_install_excepthook_routes_and_restores(qtbot):
    from vrphoto.gui.app import ErrorReporter, install_excepthook

    class Rec(ErrorReporter):
        def __init__(self):
            super().__init__()
            self.got = []

        def report(self, exc_info):
            self.got.append(exc_info[1])

    rep = Rec()
    old_sys, old_thread = sys.excepthook, threading.excepthook
    restore = install_excepthook(rep)
    try:
        err = RuntimeError("from sys hook")
        sys.excepthook(RuntimeError, err, None)
        t = threading.Thread(target=lambda: (_ for _ in ()).throw(KeyError("in thread")))
        t.start()
        t.join()
        assert rep.got[0] is err
        assert isinstance(rep.got[1], KeyError)
    finally:
        restore()
    assert sys.excepthook is old_sys and threading.excepthook is old_thread


def test_friendly_error_text():
    from vrphoto.core.errors import InferenceError
    from vrphoto.gui.util import error_text

    assert error_text(InferenceError("GPU failed", hint="Use CPU")) == ("GPU failed", "Use CPU")
    msg, hint = error_text(MemoryError())
    assert "memory" in msg and "Max processing size" in hint
    assert error_text(ValueError("x"))[0] == "ValueError: x"
    assert error_text("plain")[0] == "plain"
    assert "Permission denied" in error_text(PermissionError("locked"))[0]


# --------------------------------------------------------------------------- theme
def test_themes_apply(qtbot):
    from vrphoto.gui.theme import DARK, LIGHT, apply_theme, colors, resolve_theme, stylesheet

    app = QApplication.instance()
    assert apply_theme(app, "light") == "light"
    assert app.styleSheet() == stylesheet(LIGHT)
    assert apply_theme(app, "dark") == "dark"
    assert "QPushButton#PrimaryButton" in app.styleSheet()
    assert resolve_theme("system") in ("dark", "light")
    assert colors("dark") is DARK
    assert "arrow_down" in stylesheet(DARK)  # bundled arrow images found via resource_path


def test_resources_bundled():
    from vrphoto.core.config import resource_path

    for name in (
        "icon.png",
        "arrow_up_light.png",
        "arrow_down_light.png",
        "arrow_up_dark.png",
        "arrow_down_dark.png",
    ):
        assert resource_path("gui", "resources", name).is_file(), name


def test_app_icon(qtbot):
    from vrphoto.gui.main_window import app_icon

    assert not app_icon().isNull()


# --------------------------------------------------------------------------- state
def test_gui_state_round_trip(isolated_home):
    from vrphoto.gui.state import GuiState

    st = GuiState(last_mode="scene", options={"scene": {"target_faces": 200000}}, splitters={"top": "AAA"})
    st.save()
    back = GuiState.load()
    assert back.last_mode == "scene" and back.options["scene"]["target_faces"] == 200000
    assert back.splitters == {"top": "AAA"}


def test_gui_state_tolerates_corrupt_file(isolated_home):
    from vrphoto.gui.state import GuiState

    GuiState.default_path().write_text("{not json", encoding="utf-8")
    assert GuiState.load().last_mode == "stereo"
    GuiState.default_path().write_text('{"last_mode": 5, "options": [], "unknown": 1}', encoding="utf-8")
    st = GuiState.load()
    assert st.last_mode == "stereo" and st.options == {}


# --------------------------------------------------------------------------- utilities
def test_np_to_qimage_formats(qtbot):
    from vrphoto.gui.util import np_to_pixmap, np_to_qimage

    rgb = np.zeros((10, 20, 3), np.uint8)
    rgb[..., 0] = 255
    q = np_to_qimage(rgb)
    assert (q.width(), q.height()) == (20, 10)
    assert q.pixelColor(5, 5).red() == 255 and q.pixelColor(5, 5).green() == 0
    assert np_to_qimage(np.zeros((4, 5), np.uint8)).width() == 5
    assert np_to_qimage(np.zeros((4, 5, 4), np.uint8)).hasAlphaChannel()
    assert np_to_qimage(np.full((3, 3, 3), 1.0, np.float32)).pixelColor(0, 0).red() == 255
    assert not np_to_pixmap(rgb[:, ::2]).isNull()  # non-contiguous input
    with pytest.raises(ValueError):
        np_to_qimage(np.zeros((2, 2, 2), np.uint8))


def test_collect_media(media):
    from vrphoto.gui.util import collect_media, is_media

    found = collect_media([media["dir"], media["a"], media["txt"]])
    names = [p.name for p in found]
    assert "notes.txt" not in names
    assert names.count("a.jpg") == 1
    assert "c.jpg" in names  # recursive
    assert is_media("X.HEIC") and not is_media("x.txt")
    assert collect_media([media["dir"] / "missing"]) == []


def test_thumbnails_and_sources(media):
    from vrphoto.gui.util import load_preview_source, load_thumbnail

    th = load_thumbnail(media["a"], 48)
    assert max(th.shape[:2]) <= 48 and th.shape[2] == 3
    src = load_preview_source(media["b"], 100)
    assert max(src.shape[:2]) == 100
    if "video" in media:
        assert max(load_thumbnail(media["video"], 32).shape[:2]) <= 32


def test_human_formatting_and_output_folder(tmp_path):
    from vrphoto.gui.util import human_duration, human_size_mb, output_folder_of

    assert human_size_mb(1500) == "1.5 GB" and human_size_mb(99.4) == "99 MB" and human_size_mb(0) == "?"
    assert human_duration(5) == "5 s" and human_duration(125) == "2 min 05 s" and "h" in human_duration(7300)
    f = tmp_path / "x.jpg"
    f.write_bytes(b"1")
    assert output_folder_of({"a": str(tmp_path / "missing.jpg"), "b": str(f)}) == tmp_path
    assert output_folder_of({}, "C:/fallback") == Path("C:/fallback")
    assert output_folder_of(None) is None


def test_file_dialog_filter_lists_extensions():
    from vrphoto.gui.util import file_dialog_filter

    f = file_dialog_filter()
    assert "*.jpg" in f and "*.mp4" in f and "*.heic" in f


# --------------------------------------------------------------------------- server controller
def test_server_controller_lifecycle(qtbot, settings, fake_server_factory):
    from vrphoto.gui.server_ctl import ServerController

    ctl = ServerController(settings, server_factory=fake_server_factory)
    assert not ctl.running and ctl.urls() == {} and ctl.scene_url("x") == ""
    with qtbot.waitSignal(ctl.started, timeout=5000) as blocker:
        ctl.start()
    assert blocker.args[0]["https"] == "https://192.168.1.5:8443/"
    assert ctl.running and ctl.https_url().startswith("https://")
    assert ctl.scene_url("my scene") == "https://192.168.1.5:8443/view.html?scene=my%20scene"
    with qtbot.waitSignal(ctl.started, timeout=2000):
        ctl.start()  # already running: re-announces the URLs
    assert len(fake_server_factory.instances) == 1
    ctl.restart()
    qtbot.waitUntil(lambda: ctl.running, timeout=5000)
    assert len(fake_server_factory.instances) == 2 and fake_server_factory.instances[0].stops == 1
    with qtbot.waitSignal(ctl.stopped, timeout=5000):
        ctl.stop()
    assert not ctl.running
    ctl.shutdown()


def test_server_controller_stop_during_start(qtbot, settings, fake_server_factory):
    """Quick Start -> Stop clicks are reconciled: the server ends up stopped."""
    from vrphoto.gui.server_ctl import ServerController

    ctl = ServerController(settings, server_factory=fake_server_factory)
    ctl.start()
    assert ctl.busy
    ctl.stop()
    qtbot.waitUntil(lambda: not ctl.busy and not ctl.running, timeout=5000)
    srv = fake_server_factory.instances[-1]
    assert srv.starts == 1 and srv.stops == 1
    # restart requested while starting -> a fresh server with the new settings
    ctl.start()
    ctl.restart()
    qtbot.waitUntil(
        lambda: ctl.running and not ctl.busy and len(fake_server_factory.instances) == 3, timeout=5000
    )
    assert fake_server_factory.instances[1].stops == 1
    ctl.shutdown()
    assert not ctl.running and fake_server_factory.instances[2].stops == 1


def test_server_controller_failure_is_not_retried(qtbot, settings):
    from vrphoto.core.errors import VRPhotoError
    from vrphoto.gui.server_ctl import ServerController

    attempts = []

    class Broken:
        running = False

        def start(self):
            attempts.append(1)
            raise VRPhotoError("busy port")

        def stop(self):
            pass

    ctl = ServerController(settings, server_factory=lambda s: Broken())
    with qtbot.waitSignal(ctl.failed, timeout=5000) as blocker:
        ctl.start()
    assert "busy port" in blocker.args[0]
    qtbot.wait(50)
    assert attempts == [1] and not ctl.busy and not ctl.running
    ctl.shutdown()
