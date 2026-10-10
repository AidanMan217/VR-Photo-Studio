"""vrphoto.pipeline - job specs, the job runner, the job queue and output naming.

* ``jobs``    - job dataclasses (one per output mode) + JSON round trip
* ``run``     - :func:`run_job` (spec -> output files)
* ``context`` - :class:`RunContext` (settings, cached models, depth cache)
* ``queue``   - :class:`JobQueue` (background worker threads, events)
* ``naming``  - Quest player filename tokens, Windows-safe names
* ``cache``   - on-disk depth cache

Names are resolved lazily so ``import vrphoto.pipeline`` stays cheap.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY: dict[str, str] = {
    "run_job": "run",
    "RunContext": "context",
    "JobQueue": "queue",
    "JobEvent": "queue",
    "JobRecord": "queue",
    "DepthCache": "cache",
    "output_name": "naming",
    "unique_path": "naming",
    "JOB_TYPES": "jobs",
    "JobResult": "jobs",
    "job_from_dict": "jobs",
    "job_to_dict": "jobs",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'vrphoto.pipeline' has no attribute {name!r}")
    value = getattr(import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
