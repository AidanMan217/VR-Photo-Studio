"""Single-instance guard: a second start of the app hands its files to the running window.

The frozen app takes a few seconds to start, so users often double-click its icon twice.
Two instances would compete for the GPU (jobs failing with out-of-memory) and for the
viewer ports.  The first instance listens on a per-user local socket (a named pipe on
Windows); a later start connects, sends the files it was given, waits for an "OK" and
exits.  If the running instance does not answer (hung, or busy shutting down) the new
one simply starts normally.
"""

from __future__ import annotations

import contextlib
import getpass
import hashlib
import json
import os
import sys
import time
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from .. import APP_ID
from ..core.log import get_logger

log = get_logger(__name__)

#: Set to "1" to allow several instances (e.g. for testing two data folders side by side).
ENV_MULTI = "VRPHOTO_MULTI_INSTANCE"


def server_name(key: str | None = None) -> str:
    """Local socket name: unique per user and data folder (``$VRPHOTO_HOME``)."""
    if key is None:
        from ..core.config import data_root

        key = str(data_root())
    try:
        user = getpass.getuser()
    except Exception:  # no user name in this environment
        user = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    digest = hashlib.sha1(f"{user}|{os.path.normcase(key)}".encode()).hexdigest()[:16]
    return f"{APP_ID}-{digest}"


def _allow_foreground() -> None:
    """Windows: let the running instance bring its window to the front (foreground lock)."""
    if sys.platform != "win32":
        return
    with contextlib.suppress(Exception):
        import ctypes

        ctypes.windll.user32.AllowSetForegroundWindow(-1)  # type: ignore[attr-defined]  # ASFW_ANY


class SingleInstance(QObject):
    """Per-user single-instance guard.

    Signals:
        activated(list): another start asked this instance to show itself; the list holds
            the file paths it was given (may be empty).
    """

    activated = Signal(list)

    def __init__(self, key: str | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.name = server_name(key)
        self._server: QLocalServer | None = None
        self._buffers: dict[int, bytearray] = {}

    @property
    def listening(self) -> bool:
        return self._server is not None and self._server.isListening()

    # ------------------------------------------------------------------ second instance
    def send_to_running(self, files: list[str], *, timeout_ms: int = 5000) -> bool:
        """Hand ``files`` to a running instance.  True if it confirmed (then exit).

        Returns False within half a second when no instance is running.  ``timeout_ms`` is
        how long a running instance may take to answer (it may still be starting up).
        """
        sock = QLocalSocket()
        sock.connectToServer(self.name)
        if not sock.waitForConnected(min(500, timeout_ms)):
            return False
        try:
            _allow_foreground()
            sock.write((json.dumps({"files": [str(f) for f in files]}) + "\n").encode("utf-8"))
            sock.flush()  # may already write everything (then waitForBytesWritten has nothing to wait for)
            if sock.bytesToWrite() > 0 and not sock.waitForBytesWritten(timeout_ms):
                return False
            deadline = time.monotonic() + timeout_ms / 1000.0
            reply = bytes(sock.readAll().data())  # the answer may already be buffered
            while b"\n" not in reply:
                left_ms = int((deadline - time.monotonic()) * 1000)
                if left_ms <= 0 or not sock.waitForReadyRead(left_ms):
                    reply += bytes(sock.readAll().data())
                    break
                reply += bytes(sock.readAll().data())
            ok = reply.startswith(b"OK")
            if not ok:
                log.warning("The running VR Photo Studio did not answer; starting another one")
            return ok
        finally:
            sock.abort()

    # ------------------------------------------------------------------ first instance
    def listen(self) -> bool:
        """Start accepting hand-overs.  False if the name cannot be used (no guard then)."""
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        if not server.listen(self.name):
            # a stale socket of a crashed instance (Unix), or an instance that did not answer
            QLocalServer.removeServer(self.name)
            if not server.listen(self.name):
                log.warning("Single-instance guard unavailable: %s", server.errorString())
                server.deleteLater()
                return False
        server.newConnection.connect(self._on_connection)
        self._server = server
        return True

    def close(self) -> None:
        """Stop accepting hand-overs (call when the app starts shutting down)."""
        if self._server is not None:
            self._server.close()
            self._server.deleteLater()
            self._server = None

    def _on_connection(self) -> None:
        server = self._server
        while server is not None and server.hasPendingConnections():
            sock = server.nextPendingConnection()
            if sock is None:
                break
            key = id(sock)
            self._buffers[key] = bytearray()
            sock.readyRead.connect(lambda s=sock: self._read(s))
            sock.disconnected.connect(lambda k=key, s=sock: (self._buffers.pop(k, None), s.deleteLater()))
            if sock.bytesAvailable():
                self._read(sock)

    def _read(self, sock: Any) -> None:
        buf = self._buffers.get(id(sock))
        if buf is None:
            return
        buf += bytes(sock.readAll().data())
        if len(buf) > 1_000_000:  # not one of ours
            sock.abort()
            return
        if b"\n" not in buf:
            return
        line = bytes(buf.split(b"\n", 1)[0])
        self._buffers.pop(id(sock), None)
        try:
            msg = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            msg = {}
        files = [f for f in (msg.get("files") or []) if isinstance(f, str)] if isinstance(msg, dict) else []
        sock.write(b"OK\n")
        sock.flush()
        sock.disconnectFromServer()  # waits until the reply is written
        log.info("Another start of the app was handed over (%d file(s))", len(files))
        self.activated.emit(files)
