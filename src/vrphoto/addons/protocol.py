"""JSON-lines protocol between the app and add-on worker processes.

The app starts ``<venv python> <worker.py>`` and writes ONE request line to stdin::

    {"type": "request", "protocol": 1, "task": "image_to_3d", "inputs": {...},
     "params": {...}, "out_dir": "C:/.../out"}

The worker writes JSON lines to its *protocol stream* (its original stdout; workers
redirect fd 1 to stderr so stray prints from libraries cannot corrupt the stream)::

    {"type": "hello", "protocol": 1, "worker": "trellis", "version": "..."}
    {"type": "progress", "frac": 0.42, "msg": "Sampling structure"}
    {"type": "log", "level": "info", "msg": "..."}
    {"type": "result", "outputs": {"glb": "C:/.../out/model.glb"}}
    {"type": "error", "message": "CUDA out of memory", "trace": "..."}

Exactly one ``result`` or ``error`` message ends a run.  Lines that are not valid JSON
objects with a known ``type`` are ignored by the app (logged at debug level).
A worker invoked with ``--check`` must import its heavy deps, print a single
``{"type":"result","outputs":{}}`` (or error) and exit; used to verify installs.
"""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 1
MSG_TYPES = ("hello", "progress", "log", "result", "error")


def make_request(task: str, inputs: dict[str, Any], params: dict[str, Any], out_dir: str) -> str:
    return json.dumps(
        {
            "type": "request",
            "protocol": PROTOCOL_VERSION,
            "task": task,
            "inputs": inputs,
            "params": params,
            "out_dir": out_dir,
        }
    )


def encode(msg: dict[str, Any]) -> str:
    return json.dumps(msg, separators=(",", ":"))


def decode(line: str | bytes) -> dict[str, Any] | None:
    """Parse one protocol line; returns None for junk."""
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        msg = json.loads(line)
    except ValueError:
        return None
    if not isinstance(msg, dict) or msg.get("type") not in MSG_TYPES:
        return None
    return msg
