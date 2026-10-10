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


def test_error_reporter_does_not_repeat_the_same_error(qtbot, monkeypatch):
    """A recurring error (e.g. raised in a timer) is shown once, not in an endless loop."""
    from vrphoto.gui.app import ErrorReporter

    rep = ErrorReporter(repeat_after_s=60)
    shown: list[str] = []
    monkeypatch.setattr(rep, "show_dialog", lambda info: shown.append(str(info[1])))
    for _ in range(3):
        rep._show((ValueError, ValueError("timer broke"), None))
    rep._show((ValueError, ValueError("something else"), None))
    assert shown == ["timer broke", "something else"]
    rep.repeat_after_s = 0
    rep._show((ValueError, ValueError("timer broke"), None))
    assert shown[-1] == "timer broke" and len(shown) == 3


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
    assert "check_light.png" in stylesheet(DARK) and "QCheckBox::indicator:checked" in stylesheet(LIGHT)


def test_resources_bundled():
    from vrphoto.core.config import resource_path

    for name in (
        "icon.png",
        "arrow_up_light.png",
        "arrow_down_light.png",
        "arrow_up_dark.png",
        "arrow_down_dark.png",
        "check_light.png",
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
    assert human_duration(0.5) == "0.5 s" and human_duration(0.01) == "less than 0.1 s"  # never "0 s"
    assert human_duration(42.4) == "42 s"
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


# --------------------------------------------------------------------------- review fixes
def test_stylesheet_image_urls_are_quoted(monkeypatch, tmp_path):
    """Install paths with spaces/parentheses (e.g. 'Program Files (x86)') keep working."""
    from vrphoto.gui import theme

    res = tmp_path / "VR Photo Studio (x86)" / "vrphoto" / "gui" / "resources"
    res.mkdir(parents=True)
    for n in ("arrow_up_light.png", "arrow_down_light.png"):
        (res / n).write_bytes(b"png")
    monkeypatch.setattr(
        "vrphoto.core.config.resource_path", lambda *parts: res.parent.parent.joinpath(*parts)
    )
    qss = theme.stylesheet(theme.DARK)
    assert f'url("{(res / "arrow_up_light.png").as_posix()}")' in qss


def test_follow_system_theme_reapplies_only_for_system(qtbot):
    from vrphoto.gui.theme import apply_theme, follow_system_theme

    app = QApplication.instance()
    chosen = {"theme": "dark"}
    stop = follow_system_theme(app, lambda: chosen["theme"])
    apply_theme(app, "light")
    from PySide6.QtGui import QGuiApplication

    QGuiApplication.styleHints().colorSchemeChanged.emit(QGuiApplication.styleHints().colorScheme())
    assert app.property("vrphoto_theme") == "light"  # user chose a fixed theme: untouched
    chosen["theme"] = "system"
    QGuiApplication.styleHints().colorSchemeChanged.emit(QGuiApplication.styleHints().colorScheme())
    from vrphoto.gui.theme import resolve_theme

    assert app.property("vrphoto_theme") == resolve_theme("system")
    stop()
    apply_theme(app, "light")
    chosen["theme"] = "system"
    QGuiApplication.styleHints().colorSchemeChanged.emit(QGuiApplication.styleHints().colorScheme())
    assert app.property("vrphoto_theme") == "light"  # no longer following
    apply_theme(app, "dark")


def test_collect_media_stops_at_the_limit(media):
    from vrphoto.gui.util import collect_media_ex

    found, truncated = collect_media_ex([media["dir"]], max_files=2)
    assert len(found) == 2 and truncated
    found, truncated = collect_media_ex([media["dir"]], max_files=100)
    assert len(found) >= 3 and not truncated
    found, truncated = collect_media_ex([media["a"], media["a"]], max_files=1)
    assert found == [media["a"]] and not truncated  # duplicates do not count


# --------------------------------------------------------------------------- single instance
def _run_client(qtbot, code: str, *args: str) -> str:
    """Run ``code`` in another Python process (a second "start" of the app) while this
    process keeps serving Qt events; returns its stdout."""
    import os
    import subprocess

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.Popen(
        [sys.executable, "-c", code, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        qtbot.waitUntil(lambda: proc.poll() is not None, timeout=30000)
        out, err = proc.communicate(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    return out


SEND = (
    "import sys, json\n"
    "from PySide6.QtCore import QCoreApplication\n"
    "from vrphoto.gui.single_instance import SingleInstance\n"
    "app = QCoreApplication([])\n"
    "print(SingleInstance(sys.argv[1]).send_to_running(json.loads(sys.argv[2]), timeout_ms=5000))\n"
)


def test_single_instance_hands_files_to_the_running_one(qtbot, tmp_path):
    import json

    from vrphoto.gui.single_instance import SingleInstance

    key = str(tmp_path / "home")
    first = SingleInstance(key)
    assert first.listen() and first.listening
    got: list[list[str]] = []
    first.activated.connect(got.append)
    files = ["C:/photos/a b.jpg", "x.png"]
    assert _run_client(qtbot, SEND, key, json.dumps(files)).strip() == "True"
    qtbot.waitUntil(lambda: bool(got), timeout=5000)
    assert got == [files]
    first.close()
    assert not first.listening
    assert _run_client(qtbot, SEND, key, "[]").strip() == "False"  # nobody listening: start normally


def test_single_instance_names_are_per_data_folder(tmp_path):
    from vrphoto.gui.single_instance import server_name

    assert server_name(str(tmp_path / "a")) != server_name(str(tmp_path / "b"))
    assert server_name(str(tmp_path / "a")) == server_name(str(tmp_path / "a"))
    assert server_name(str(tmp_path / "a")).startswith("VRPhotoStudio-")


def test_single_instance_ignores_garbage(qtbot, tmp_path):
    from vrphoto.gui.single_instance import SingleInstance

    key = str(tmp_path / "home")
    first = SingleInstance(key)
    assert first.listen()
    got: list[list[str]] = []
    first.activated.connect(got.append)
    junk = (
        "import sys\n"
        "from PySide6.QtCore import QCoreApplication\n"
        "from PySide6.QtNetwork import QLocalSocket\n"
        "app = QCoreApplication([])\n"
        "s = QLocalSocket(); s.connectToServer(sys.argv[1]); assert s.waitForConnected(3000)\n"
        "s.write(b'not json\\n'); s.flush()\n"
        "s.waitForReadyRead(5000); print(bytes(s.readAll().data()).decode().strip())\n"
    )
    assert _run_client(qtbot, junk, first.name).strip() == "OK"
    qtbot.waitUntil(lambda: bool(got), timeout=5000)
    assert got == [[]]  # just "show yourself"
    first.close()


def test_second_start_hands_over_and_exits(qtbot, monkeypatch, isolated_home, tmp_path):
    """run_gui exits at once when another instance (here: a subprocess) is running."""
    import os
    import subprocess

    from vrphoto.gui import app as gui_app

    script = tmp_path / "first.py"
    script.write_text(
        "import sys\n"
        "from PySide6.QtCore import QCoreApplication, QTimer\n"
        "from vrphoto.gui.single_instance import SingleInstance\n"
        "app = QCoreApplication([])\n"
        "g = SingleInstance()\n"
        "assert g.listen()\n"
        "g.activated.connect(lambda files: (print('GOT', files, flush=True), app.quit()))\n"
        "print('READY', flush=True)\n"
        "QTimer.singleShot(30000, app.quit)\n"
        "app.exec()\n"
    )
    env = dict(os.environ, VRPHOTO_HOME=str(isolated_home), QT_QPA_PLATFORM="offscreen")
    proc = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, text=True, env=env, cwd=str(tmp_path)
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "READY"
        monkeypatch.setattr(gui_app, "setup_logging", lambda *a, **k: None)
        monkeypatch.setattr(gui_app.QMessageBox, "critical", lambda *a, **k: None)  # never block
        built = []
        from vrphoto.gui import main_window

        monkeypatch.setattr(main_window, "MainWindow", lambda *a, **k: built.append(1))
        photo = tmp_path / "p.jpg"
        photo.write_bytes(b"x")
        assert gui_app.run_gui(["vrphoto", str(photo)]) == 0
        assert built == []  # no second window
        out, _ = proc.communicate(timeout=20)
        assert "GOT" in out and "p.jpg" in out
    finally:
        if proc.poll() is None:
            proc.kill()


def test_bring_to_front_adds_files_and_guard_closes_on_shutdown(make_window, media, tmp_path):
    from vrphoto.gui.single_instance import SingleInstance

    w, _ = make_window()
    guard = SingleInstance(str(tmp_path / "guard"))
    assert guard.listen()
    w.instance_guard = guard
    w.showMinimized()
    w.bring_to_front([str(media["a"])])
    assert w.inputs.paths() == [media["a"]] and not w.isMinimized()
    w.shutdown()
    assert not guard.listening
    w.bring_to_front([str(media["b"])])  # ignored after shutdown
    assert w.inputs.paths() == [media["a"]]
