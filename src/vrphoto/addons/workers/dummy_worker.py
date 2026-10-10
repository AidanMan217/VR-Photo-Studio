"""Dummy add-on worker used by the test-suite and the self-test (stdlib only).

Tasks:

* ``echo``  - returns ``{"echo": {"inputs": ..., "params": ...}, "file": <json written to out_dir>}``.
  params: ``steps`` (int, progress messages), ``delay`` (s per step), ``fail`` (str: raise a
  user-facing error with that message), ``oom`` (bool: raise a CUDA-OOM style error),
  ``junk`` (bool: print garbage to stdout from Python *and* from the raw fd 1).
* ``sleep`` - sleeps ``seconds`` (default 30) while reporting progress (for cancel tests);
  ``pid_file`` (optional) receives the worker's process id.
* ``crash`` - exits abruptly without sending a result (``code`` param, default 3).

``--check`` reports the interpreter; set ``DUMMY_CHECK_FAIL=1`` to make it fail.
``--prefetch`` writes ``prefetch.txt`` into ``VRPHOTO_ADDON_DIR`` (``DUMMY_PREFETCH_FAIL=1``
makes it fail).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

from _common import Progress, WorkerError, addon_dir, param, run

VERSION = "1.0"


def handle(task: str, inputs: dict, params: dict, out_dir: Path, progress: Progress) -> dict:
    if task == "echo":
        steps = param(params, "steps", 3, int, lo=0, hi=1000)
        delay = param(params, "delay", 0.0, float, lo=0.0, hi=10.0)
        if params.get("junk"):
            print("hello from a chatty library")  # goes to stderr thanks to _common
            os.write(1, b"raw fd-1 junk {not json}\n")
            sys.stdout.write('{"type": "result", "outputs": {"fake": true}}\n')  # must NOT count
            sys.stdout.flush()
        for i in range(steps):
            progress((i + 1) / max(steps, 1), f"step {i + 1}/{steps}")
            if delay:
                time.sleep(delay)
        if params.get("oom"):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 12.00 GiB total capacity)")
        if params.get("fail"):
            raise WorkerError(str(params["fail"]), hint="dummy hint")
        payload = {"inputs": inputs, "params": params}
        path = out_dir / "echo.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"echo": payload, "file": str(path)}
    if task == "sleep":
        seconds = param(params, "seconds", 30.0, float, lo=0.0, hi=3600.0)
        if params.get("pid_file"):  # lets tests find (and watch) the worker process
            Path(str(params["pid_file"])).write_text(str(os.getpid()), encoding="utf-8")
        t0 = time.monotonic()
        while (elapsed := time.monotonic() - t0) < seconds:
            progress(elapsed / seconds if seconds else 1.0, "sleeping")
            time.sleep(0.05)
        return {"slept": seconds}
    if task == "crash":
        sys.stderr.write("about to crash\n")
        sys.stderr.flush()
        os._exit(int(params.get("code", 3)))
    raise WorkerError(f"Unknown task '{task}'.")


def check(require_gpu: bool) -> dict:
    if os.environ.get("DUMMY_CHECK_FAIL") == "1":
        raise WorkerError("dummy check failed")
    return {"python": platform.python_version(), "executable": sys.executable, "gpu_required": require_gpu}


def prefetch(progress: Progress) -> dict:
    if os.environ.get("DUMMY_PREFETCH_FAIL") == "1":
        raise WorkerError("dummy prefetch failed")
    progress(0.5, "prefetching")
    path = addon_dir() / "prefetch.txt"
    path.write_text("ok", encoding="utf-8")
    return {"file": str(path)}


if __name__ == "__main__":
    sys.exit(run(handle, name="dummy", version=VERSION, check=check, prefetch=prefetch))
