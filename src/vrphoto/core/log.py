"""Logging setup: rotating file log in ``<data_root>/logs`` plus console when available.

Safe in a PyInstaller windowed app where ``sys.stdout``/``sys.stderr`` are ``None``.

The GUI writes ``vrphoto.log``; the command line tool passes ``filename=CLI_LOG_FILE`` so
the two never share one file.  The file handler also tolerates a log that another process
keeps open: on Windows such a file cannot be renamed, so rotation is postponed (with
backoff) instead of failing - and dropping the record - on every single log line.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

_configured = False
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_FILE = "vrphoto.log"
CLI_LOG_FILE = "cli.log"
MAX_BYTES = 5_000_000
BACKUP_COUNT = 3


def _now() -> float:
    return time.monotonic()


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """``RotatingFileHandler`` that survives a log file locked by another process.

    * A failed rotation (Windows ``PermissionError`` [WinError 32] while another VR Photo
      Studio process has the file open) keeps writing to the current file and retries the
      rotation only after a growing pause (``RETRY_FIRST_S`` doubling up to ``RETRY_MAX_S``).
    * The live file is moved aside *before* the backups are shifted, so a failed attempt
      leaves every file where it was (no lost backup generations).
    * Any other logging failure prints at most one traceback per ``ERROR_QUIET_S``, so a
      broken disk can never flood the console.
    """

    RETRY_FIRST_S = 30.0
    RETRY_MAX_S = 900.0
    ERROR_QUIET_S = 600.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._retry_at = 0.0
        self._failures = 0
        self._quiet_until = 0.0

    def shouldRollover(self, record: logging.LogRecord) -> bool:  # noqa: N802 - stdlib name
        if self._retry_at and _now() < self._retry_at:
            return False
        return bool(super().shouldRollover(record))

    def doRollover(self) -> None:  # noqa: N802 - stdlib name
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]
        try:
            if self.backupCount > 0:
                self._rotate_files()
        except OSError:
            self._failures += 1
            pause = min(self.RETRY_MAX_S, self.RETRY_FIRST_S * 2 ** (self._failures - 1))
            self._retry_at = _now() + pause
        else:
            self._failures = 0
            self._retry_at = 0.0
        # Keep logging into whichever file is now at baseFilename (the old one if the
        # rotation was postponed).  With delay=True FileHandler.emit reopens it lazily.
        if not self.delay:
            self.stream = self._open()

    def _rotate_files(self) -> None:
        base = self.baseFilename
        hold = base + ".rotating"
        if os.path.exists(hold):  # left behind by a crash in the middle of a rotation
            os.remove(hold)
        # The step that fails while another process has the log open: do it first.
        os.rename(base, hold)
        try:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename(f"{base}.{i}")
                dfn = self.rotation_filename(f"{base}.{i + 1}")
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = self.rotation_filename(base + ".1")
            if os.path.exists(dfn):
                os.remove(dfn)
            os.rename(hold, dfn)
        except OSError:
            try:  # put the live log back so nothing is lost
                if not os.path.exists(base):
                    os.rename(hold, base)
            except OSError:
                pass
            raise

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802 - stdlib name
        now = _now()
        if now < self._quiet_until:
            return
        self._quiet_until = now + self.ERROR_QUIET_S
        super().handleError(record)


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | None = None,
    console: bool = True,
    *,
    filename: str = LOG_FILE,
) -> Path | None:
    """Configure root logging once.  Returns the log file path (or None if unwritable).

    ``filename`` names the file inside ``log_dir`` (default ``<data_root>/logs``); the
    command line tool uses :data:`CLI_LOG_FILE` so it never shares the GUI's log file.
    """
    global _configured
    root = logging.getLogger()
    if _configured:
        return getattr(setup_logging, "_path", None)
    root.setLevel(level)
    fmt = logging.Formatter(LOG_FORMAT)
    path: Path | None = None
    try:
        if log_dir is None:
            from .config import data_root

            log_dir = data_root() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / (filename or LOG_FILE)
        fh = SafeRotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        path = None
    if console and sys.stderr is not None:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    _configured = True
    setup_logging._path = path  # type: ignore[attr-defined]
    return path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("vrphoto") else f"vrphoto.{name}")
