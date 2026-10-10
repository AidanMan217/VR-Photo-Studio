"""Persistent GUI state (last mode, last options per mode, window layout).

Stored as JSON in ``<data_root>/gui_state.json`` next to ``settings.json`` (the
:class:`~vrphoto.core.config.Settings` dataclass has no field for free-form GUI state).
Unknown keys are ignored and a corrupt file falls back to defaults, like Settings.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..core.config import data_root
from ..core.log import get_logger

log = get_logger(__name__)

STATE_FILE = "gui_state.json"


@dataclass
class GuiState:
    last_mode: str = "stereo"
    #: ``kind -> job options dict`` (``job_to_dict`` without ``inputs``/``kind``)
    options: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: ``kind -> preset name`` last chosen (informational)
    presets: dict[str, str] = field(default_factory=dict)
    geometry: str = ""  # base64 QMainWindow.saveGeometry()
    window_state: str = ""  # base64 QMainWindow.saveState()
    splitters: dict[str, str] = field(default_factory=dict)  # name -> base64 saveState()
    last_open_dir: str = ""
    advanced_open: bool = False

    @staticmethod
    def default_path() -> Path:
        return data_root() / STATE_FILE

    @classmethod
    def load(cls, path: Path | None = None) -> GuiState:
        path = path or cls.default_path()
        st = cls()
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    names = {f.name: f for f in fields(cls)}
                    for k, v in data.items():
                        if k in names and isinstance(v, type(getattr(st, k))):
                            setattr(st, k, v)
        except (OSError, ValueError) as e:
            log.warning("Ignoring unreadable GUI state %s: %s", path, e)
        return st

    def save(self, path: Path | None = None) -> None:
        path = path or self.default_path()
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            log.warning("Could not save GUI state to %s: %s", path, e)
