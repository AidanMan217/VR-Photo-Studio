"""vrphoto.webxr - delivering VR content to the Quest.

* ``scene``  - export/list viewer scenes in the library (``scene.json`` + assets)
* ``server`` - :class:`ViewerServer`: HTTPS (LAN) + HTTP (localhost/adb) web server
* ``cert``   - self-signed certificate for the HTTPS server
* ``net``    - LAN addresses, QR codes, Windows Firewall
* ``quest``  - adb (USB debugging) and MTP transfers to the headset

Names are resolved lazily, so ``import vrphoto.webxr`` stays cheap and has no side effects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY: dict[str, str] = {
    # server / cert / net / quest
    "ViewerServer": "server",
    "ServerError": "server",
    "ensure_cert": "cert",
    "lan_ips": "net",
    "primary_ip": "net",
    "qr_png": "net",
    "firewall_status": "net",
    "add_firewall_rule": "net",
    "find_adb": "quest",
    "install_platform_tools": "quest",
    "adb_devices": "quest",
    "adb_reverse": "quest",
    "adb_push": "quest",
    "open_url_on_quest": "quest",
    "mtp_copy": "quest",
    # scene (contract names)
    "export_scene": "scene",
    "list_scenes": "scene",
    "delete_scene": "scene",
    "validate_manifest": "scene",
    "new_scene_id": "scene",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'vrphoto.webxr' has no attribute {name!r}")
    value = getattr(import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
