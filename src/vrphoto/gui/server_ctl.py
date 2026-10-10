"""Owns the WebXR :class:`~vrphoto.webxr.server.ViewerServer` for the GUI.

Starting the server can take a moment (certificate generation, LAN address detection),
so :meth:`ServerController.start` runs in a background thread and reports through
signals.  The controller outlives the "Send to Quest" dialog so the viewer keeps running
while the user puts the headset on; the main window stops it on exit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from ..core.log import get_logger
from .util import error_text
from .workers import TaskRunner

log = get_logger(__name__)


class ServerController(QObject):
    """Start/stop the viewer server without blocking the UI.

    Signals:
        started(dict): URLs (``{"https": ..., "local": ...}``).
        stopped(): the server stopped.
        failed(str): friendly error message (with hint).
        busyChanged(bool): a start/stop is in progress.
    """

    started = Signal(dict)
    stopped = Signal()
    failed = Signal(str)
    busyChanged = Signal(bool)

    def __init__(self, settings: Any, parent: QObject | None = None, *, server_factory: Any = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.server_factory = server_factory
        self.server: Any = None
        self._starting: Any = None  # server started by the worker, not yet handed to the UI thread
        self.tasks = TaskRunner(self, max_threads=1)
        self._busy = False
        self._want = False  # desired state: running?
        self._restart_pending = False
        self.last_error: str | None = None

    # ---------------------------------------------------------------- state
    @property
    def running(self) -> bool:
        return bool(self.server is not None and getattr(self.server, "running", False))

    @property
    def busy(self) -> bool:
        return self._busy

    def urls(self) -> dict[str, str]:
        if not self.running:
            return {}
        try:
            return dict(self.server.urls())
        except Exception:
            log.debug("urls() failed", exc_info=True)
            return {}

    def https_url(self) -> str:
        return self.urls().get("https", "")

    def local_url(self) -> str:
        return self.urls().get("local", "")

    def http_problem(self) -> str:
        """Why the local http address (USB cable / 'Open on this PC') is missing, else "".

        The viewer server treats its localhost-only http listener as optional: when that port
        is busy the Wi-Fi (https) viewer still starts, on another free http port or without
        the ``"local"`` URL.
        """
        if not self.running or self.local_url():
            return ""
        port = int(getattr(self.settings, "http_port", 0) or 0)
        if not port:
            return ""
        return (
            f"The USB (http) port {port} could not be opened - another program is probably using it - "
            "so the USB cable connection is not available (Wi-Fi works normally). Choose another "
            "'USB (http) port' in Settings."
        )

    def http_note(self) -> str:
        """Note when the configured USB (http) port was busy and another one is used, else ""."""
        if not self.running or getattr(self.server, "http_error", None) is None:
            return ""
        from urllib.parse import urlparse

        try:
            actual = urlparse(self.local_url()).port
        except ValueError:
            actual = None
        wanted = int(getattr(self.settings, "http_port", 0) or 0)
        if not actual or not wanted or actual == wanted:
            return ""
        return (
            f"The USB (http) port {wanted} is used by another program, so port {actual} is used "
            "instead this time. To keep a fixed address, choose another 'USB (http) port' in Settings."
        )

    def scene_url(self, scene_id: str | None, base: str | None = None) -> str:
        """URL of the viewer page for ``scene_id`` (gallery when None)."""
        base = base if base is not None else self.https_url()
        if not base:
            return ""
        if not scene_id:
            return base
        from urllib.parse import quote

        return base.rstrip("/") + "/view.html?scene=" + quote(scene_id)

    def library(self) -> Path:
        return Path(self.settings.library_dir)

    def _make(self) -> Any:
        if self.server_factory is not None:
            return self.server_factory(self.settings)
        from ..webxr.server import ViewerServer

        # Only the viewer app and the scene library are served to the network.  The output
        # folder is never shared (no ``files_dir``): it can be any folder the user picked
        # (Desktop, Documents, a whole drive...) and the viewer does not need it.
        return ViewerServer(
            self.settings.library_dir,
            https_port=int(self.settings.https_port),
            http_port=int(self.settings.http_port),
        )

    def _set_busy(self, on: bool) -> None:
        if on != self._busy:
            self._busy = on
            self.busyChanged.emit(on)

    # ---------------------------------------------------------------- control
    # The UI asks for a desired state (``_want``); at most one start/stop runs at a time and
    # when it finishes the controller reconciles, so quick Start/Stop/Restart clicks are
    # never lost.
    def start(self) -> None:
        """Start in the background (if running, just re-announce the URLs)."""
        self._want = True
        if self.running and not self._busy:
            self.started.emit(self.urls())
            return
        self._reconcile()

    def stop(self, *, wait: bool = False) -> None:
        """Stop the server (in the background unless ``wait``)."""
        self._want = False
        if wait:
            self._restart_pending = False
            srv, self.server = self.server, None
            if srv is not None:
                self._stop_now(srv)
                self.stopped.emit()
            return
        self._reconcile()

    def restart(self) -> None:
        """Apply new settings (ports, folders): restart if running (or starting).

        Both the stop and the new start run in the background (stopping waits for the
        server threads, which must never freeze the window).
        """
        if self._busy:
            if self._want:
                self._restart_pending = True
            return
        if not self.running:
            return
        self._want = True
        self._begin_stop()  # -> _on_stopped -> _reconcile starts it again (still wanted)

    def _reconcile(self) -> None:
        if self._busy:
            return  # re-checked when the current operation finishes
        if self._want and not self.running:
            self._begin_start()
        elif not self._want and self.server is not None:
            self._begin_stop()

    def _begin_start(self) -> None:
        self._set_busy(True)

        def work() -> Any:
            srv = self._make()
            srv.start()
            self._starting = srv
            return srv

        self.tasks.start(
            work,
            pass_progress=False,
            pass_cancel=False,
            name="viewer-start",
            on_result=self._on_started,
            on_error=self._on_failed,
        )

    def _begin_stop(self) -> None:
        srv, self.server = self.server, None
        self._set_busy(True)
        self.tasks.start(
            self._stop_now,
            srv,
            pass_progress=False,
            pass_cancel=False,
            name="viewer-stop",
            on_finished=self._on_stopped,
        )

    @staticmethod
    def _stop_now(srv: Any) -> None:
        try:
            srv.stop()
        except Exception:
            log.debug("Server stop failed", exc_info=True)

    def _on_started(self, srv: Any) -> None:
        self._starting = None
        self.server = srv
        self.last_error = None
        self._set_busy(False)
        urls = self.urls()
        log.info("Live viewer running: %s", ", ".join(urls.values()))
        problem = self.http_problem() or self.http_note()
        if problem:
            log.warning("%s", problem)
        self.started.emit(urls)
        if self._restart_pending:
            self._restart_pending = False
            self.restart()
        else:
            self._reconcile()

    def _on_failed(self, exc: BaseException) -> None:
        msg, hint = error_text(exc)
        text = msg + (f"\n{hint}" if hint else "")
        self.last_error = text
        self.server = None
        self._want = False  # do not retry in a loop; the user can click Start again
        self._restart_pending = False
        self._set_busy(False)
        log.error("Could not start the live viewer: %s", msg)
        self.failed.emit(text)

    def _on_stopped(self) -> None:
        self._set_busy(False)
        log.info("Live viewer stopped")
        self.stopped.emit()
        self._reconcile()

    def shutdown(self) -> None:
        """Stop everything (blocking, for application exit)."""
        self._want = False
        self._restart_pending = False
        self.tasks.shutdown(5000)
        for srv in (self.server, self._starting):
            if srv is not None:
                self._stop_now(srv)
        self.server = None
        self._starting = None
