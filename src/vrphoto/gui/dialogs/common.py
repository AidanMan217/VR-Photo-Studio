"""Shared dialog helpers: lazy backend imports, license acceptance prompt, small widgets."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QMessageBox, QWidget

from ...core.log import get_logger

log = get_logger(__name__)


def load_backend(name: str) -> ModuleType:
    """Import ``vrphoto.<name>`` lazily (raises the original exception on failure)."""
    return importlib.import_module(f"vrphoto.{name}")


def try_backend(name: str) -> tuple[ModuleType | None, str | None]:
    """``(module, None)`` or ``(None, friendly error)`` - never raises."""
    try:
        return load_backend(name), None
    except Exception as e:  # ImportError, or a crash inside the module's import
        log.warning("Backend %s is not available: %s", name, e)
        return None, f"{type(e).__name__}: {e}"


def unavailable_label(what: str, error: str) -> QLabel:
    """A friendly "not available" message used when a backend failed to import."""
    lab = QLabel(
        f"<b>{what} is not available right now.</b><br>{error}<br><br>"
        "Try restarting VR Photo Studio. If it keeps happening, reinstall the app or report the problem."
    )
    lab.setObjectName("Error")
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def badge(text: str, kind: str) -> QLabel:
    """Small coloured label: kind "nc" (red), "exp" (amber), "ok" (green)."""
    lab = QLabel(text)
    lab.setProperty("badge", kind)
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lab


def hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def confirm_license(
    parent: QWidget | None, name: str, license_name: str, *, url: str = "", notices: list[str] | None = None
) -> bool:
    """Ask the user to accept a (non-commercial / restricted) license.  True = accepted.

    Tests monkeypatch this function.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle(f"License for {name}")
    box.setTextFormat(Qt.TextFormat.RichText)
    lines = [f"<b>{name}</b> is licensed under <b>{license_name or 'its own license'}</b>."]
    for n in notices or []:
        lines.append(f"• {n}")
    if not notices:
        lines.append(
            "• Non-commercial use only: fine for your own photos and fun, not for paid work or products."
        )
    if url:
        lines.append(f'Read the full license: <a href="{url}">{url}</a>')
    box.setText("<br>".join(lines))
    box.setInformativeText("Do you accept these terms?")
    accept = box.addButton("I accept", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(accept)
    box.exec()
    return box.clickedButton() is accept


def tail_text(path: Path | str, max_bytes: int = 64_000, max_lines: int = 400) -> str:
    """Last part of a (log) text file; empty string if missing."""
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # first line is probably cut
    return "\n".join(lines[-max_lines:])


def save_settings(settings: Any) -> bool:
    """Persist settings; logs (never raises) on failure."""
    try:
        settings.save()
        return True
    except Exception as e:
        log.warning("Could not save settings: %s", e)
        return False
