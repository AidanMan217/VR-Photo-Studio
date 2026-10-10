"""Logging setup: rotating file log in ``<data_root>/logs`` plus console when available.

Safe in a PyInstaller windowed app where ``sys.stdout``/``sys.stderr`` are ``None``.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_configured = False
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO, log_dir: Path | None = None, console: bool = True) -> Path | None:
    """Configure root logging once.  Returns the log file path (or None if unwritable)."""
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
        path = log_dir / "vrphoto.log"
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
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
