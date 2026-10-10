"""GUI entry point: :func:`run_gui`.

Creates the QApplication (high-DPI aware, app name/icon, theme), sets up logging, loads
the settings, shows the :class:`MainWindow` and installs exception hooks that log every
unhandled error and show it in a friendly dialog instead of silently dying (important
in the windowed exe, which has no console).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from .. import APP_ID, APP_NAME, __version__
from ..core.log import get_logger, setup_logging

log = get_logger(__name__)

_main_window: Any = None  # keeps the window alive for the lifetime of run_gui


# --------------------------------------------------------------------------- exception reporting
ExcInfo = tuple[type[BaseException], BaseException, types.TracebackType | None]


def _friendly(exc: BaseException) -> tuple[str, str | None]:
    from .util import error_text

    return error_text(exc)


class ErrorReporter(QObject):
    """Shows unhandled exceptions in a dialog (on the Qt thread), rate-limited."""

    _report = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._showing = False
        self.count = 0
        self.last: BaseException | None = None
        self._report.connect(self._show, Qt.ConnectionType.QueuedConnection)

    def report(self, exc_info: ExcInfo) -> None:
        """Log ``exc_info`` and queue a dialog; safe to call from any thread."""
        etype, exc, tb = exc_info
        log.error("Unhandled error: %s", exc, exc_info=(etype, exc, tb))
        self.count += 1
        self.last = exc
        try:
            self._report.emit(exc_info)
        except RuntimeError:  # reporter deleted during shutdown
            pass

    def show_dialog(self, exc_info: ExcInfo) -> None:
        """Display the error (tests monkeypatch this)."""
        etype, exc, tb = exc_info
        import traceback

        msg, hint = _friendly(exc)
        box = QMessageBox(QApplication.activeWindow())
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(f"{APP_NAME} - something went wrong")
        box.setText(msg)
        info = hint or (
            "The app can usually continue. If this keeps happening, please report it and include the log "
            "(Help > Log)."
        )
        box.setInformativeText(info)
        box.setDetailedText("".join(traceback.format_exception(etype, exc, tb)))
        box.exec()

    @Slot(object)
    def _show(self, exc_info: ExcInfo) -> None:
        if self._showing:  # one dialog at a time; the rest is in the log
            return
        self._showing = True
        try:
            self.show_dialog(exc_info)
        except Exception:  # never recurse into the hook
            log.exception("Could not show the error dialog")
        finally:
            self._showing = False


def install_excepthook(reporter: ErrorReporter) -> Callable[[], None]:
    """Route ``sys.excepthook`` and ``threading.excepthook`` to ``reporter``.

    Returns a function that restores the previous hooks.
    """
    old_sys = sys.excepthook
    old_thread = threading.excepthook

    def sys_hook(etype: type[BaseException], exc: BaseException, tb: types.TracebackType | None) -> None:
        if issubclass(etype, KeyboardInterrupt):
            old_sys(etype, exc, tb)
            return
        reporter.report((etype, exc, tb))

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit or args.exc_value is None:
            return
        reporter.report((args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = sys_hook
    threading.excepthook = thread_hook

    def restore() -> None:
        sys.excepthook = old_sys
        threading.excepthook = old_thread

    return restore


# --------------------------------------------------------------------------- application setup
def _set_windows_app_id() -> None:
    """Give the process its own taskbar identity so Windows shows our icon (not python's)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"{APP_ID}.App")  # type: ignore[attr-defined]
    except Exception:
        log.debug("SetCurrentProcessExplicitAppUserModelID failed", exc_info=True)


def create_app(argv: Sequence[str] | None = None) -> QApplication:
    """Create (or return the existing) QApplication, configured for VR Photo Studio."""
    app = QApplication.instance()
    if app is None:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        if (
            sys.platform.startswith("linux")
            and not os.environ.get("QT_QPA_PLATFORM")
            and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        ):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"  # headless Linux: avoid an abort
        app = QApplication(list(argv) if argv else [sys.argv[0] if sys.argv else APP_NAME])
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_ID)
    app.setApplicationVersion(__version__)
    try:
        app.setDesktopFileName("vrphoto")
    except Exception:  # pragma: no cover - not available on all platforms
        pass
    from .main_window import app_icon

    app.setWindowIcon(app_icon())
    return app  # type: ignore[return-value]


def _file_args(argv: Sequence[str]) -> list[Path]:
    """Existing files/folders passed on the command line (e.g. dropped on the exe)."""
    out = []
    for a in argv:
        if a.startswith("-"):
            continue
        p = Path(a)
        if p.exists():
            out.append(p)
    return out


def run_gui(argv: Sequence[str] | None = None) -> int:
    """Start the desktop app; returns the exit code.

    ``argv`` is the full argument vector (``argv[0]`` = program); further arguments that
    are existing files/folders are added to the input list.
    """
    global _main_window
    argv = list(sys.argv if argv is None else argv)
    log_path = setup_logging(logging.INFO)
    _set_windows_app_id()
    app = create_app(argv[:1])
    log.info("Starting %s %s (log: %s)", APP_NAME, __version__, log_path)

    from ..core.config import Settings, set_settings

    try:
        settings = Settings.load()
    except Exception:  # corrupt / unreadable settings -> defaults
        log.exception("Could not load settings; using defaults")
        settings = Settings()
    set_settings(settings)

    from .theme import apply_theme

    apply_theme(app, getattr(settings, "theme", "system"))
    reporter = ErrorReporter(app)
    restore = install_excepthook(reporter)
    try:
        from .main_window import MainWindow

        win = MainWindow(settings)
        _main_window = win
        app.aboutToQuit.connect(win.shutdown)
        files = _file_args(argv[1:])
        if files:
            win.add_inputs(files)
        win.show()
        code = int(app.exec())
        log.info("%s exited with code %d", APP_NAME, code)
        return code
    except Exception as e:
        log.exception("The app failed to start")
        try:
            msg, hint = _friendly(e)
            QMessageBox.critical(None, f"{APP_NAME} could not start", msg + (f"\n\n{hint}" if hint else ""))
        except Exception:
            pass
        return 1
    finally:
        restore()
        win = _main_window
        _main_window = None
        if win is not None:
            try:
                win.shutdown()
                win.hide()
                win.deleteLater()
            except Exception:
                log.debug("shutdown after exec failed", exc_info=True)
