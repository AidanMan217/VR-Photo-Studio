"""Application settings and well-known directories.

The data root is ``$VRPHOTO_HOME`` if set, else ``%LOCALAPPDATA%\\VRPhotoStudio`` on
Windows, else ``~/.local/share/vrphoto``.  Settings persist as JSON in
``<root>/settings.json``.  Unknown keys are ignored and missing keys get defaults so
old settings files keep working.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def data_root() -> Path:
    env = os.environ.get("VRPHOTO_HOME")
    if env:
        root = Path(env)
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        root = Path(base) / "VRPhotoStudio"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "VRPhotoStudio"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "vrphoto"
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_library_dir() -> Path:
    """Default output folder (user-visible): ~/Pictures/VR Photo Studio or ~/VR Photo Studio."""
    pics = Path.home() / "Pictures"
    base = pics if pics.is_dir() else Path.home()
    return base / "VR Photo Studio"


@dataclass
class Settings:
    # devices / performance
    device: str = "auto"  # "auto" | "cpu" | "<adapter index>"
    use_fp16: bool = False
    max_process_px: int = 4096  # long side cap for processing
    # models
    depth_model: str = "auto"  # see depth.get_estimator
    inpaint_model: str = "auto"
    segment_model: str = "auto"
    hf_token: str = ""
    accepted_licenses: list[str] = field(default_factory=list)  # model/add-on ids with NC/region ack
    # outputs
    output_dir: str = ""  # empty -> default_library_dir()
    quest_model: str = "quest3"  # "quest3" | "quest2" | "pcvr"
    video_encoder: str = "auto"  # "auto" | "nvenc" | "x265" | "x264" | ...
    # server
    https_port: int = 8443
    http_port: int = 8080
    # ui
    theme: str = "system"
    last_open_dir: str = ""
    show_experimental: bool = False
    gui_state: dict = field(default_factory=dict)  # window geometry, last options per mode, ...
    depth_cache_mb: int = 1024

    # ---------------------------------------------------------------- derived paths
    @property
    def root(self) -> Path:
        return data_root()

    @property
    def models_dir(self) -> Path:
        return _mk(self.root / "models")

    @property
    def addons_dir(self) -> Path:
        return _mk(self.root / "addons")

    @property
    def cache_dir(self) -> Path:
        return _mk(self.root / "cache")

    @property
    def certs_dir(self) -> Path:
        return _mk(self.root / "certs")

    @property
    def logs_dir(self) -> Path:
        return _mk(self.root / "logs")

    @property
    def library_dir(self) -> Path:
        """Where exported viewer scenes live (served to the Quest)."""
        return _mk(self.root / "library")

    @property
    def outputs_dir(self) -> Path:
        return _mk(Path(self.output_dir) if self.output_dir else default_library_dir())

    # ---------------------------------------------------------------- persistence
    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or (data_root() / "settings.json")
        s = cls()
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                names = {f.name for f in dataclasses.fields(cls)}
                for k, v in data.items():
                    if k in names:
                        setattr(s, k, v)
            except (OSError, ValueError):
                pass  # corrupt settings -> defaults
        return s

    def save(self, path: Path | None = None) -> None:
        path = path or (data_root() / "settings.json")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def license_accepted(self, ident: str) -> bool:
        return ident in self.accepted_licenses

    def accept_license(self, ident: str) -> None:
        if ident not in self.accepted_licenses:
            self.accepted_licenses.append(ident)


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


_lock = threading.Lock()
_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton (lazy-loaded)."""
    global _settings
    with _lock:
        if _settings is None:
            _settings = Settings.load()
        return _settings


def set_settings(s: Settings | None) -> None:
    """Replace the singleton (tests / settings dialog)."""
    global _settings
    with _lock:
        _settings = s


def resource_path(*parts: str) -> Path:
    """Path of a bundled package resource (works from source and from PyInstaller)."""
    base = Path(getattr(sys, "_MEIPASS", "")) / "vrphoto" if getattr(sys, "frozen", False) else None
    if base is None or not base.exists():
        base = Path(__file__).resolve().parent.parent
    return base.joinpath(*parts)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))
