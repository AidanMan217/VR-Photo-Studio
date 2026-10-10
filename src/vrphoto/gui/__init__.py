"""PySide6 desktop GUI of VR Photo Studio.

Entry point: :func:`vrphoto.gui.app.run_gui` (also available as ``vrphoto.gui.run_gui``;
imported lazily so ``import vrphoto.gui`` does not load Qt).
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_gui"]


def __getattr__(name: str) -> Any:
    if name == "run_gui":
        from .app import run_gui

        return run_gui
    raise AttributeError(f"module 'vrphoto.gui' has no attribute {name!r}")
