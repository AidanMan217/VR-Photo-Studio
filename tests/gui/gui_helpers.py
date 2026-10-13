"""Shared GUI test helpers (imported by conftest.py and the test modules)."""

from __future__ import annotations

import itertools
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------- fake queue (contract section 10)
@dataclass
class FakeJobRecord:
    job_id: str
    spec: Any
    status: str = "queued"
    frac: float = 0.0
    msg: str = ""
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None


@dataclass
class FakeJobEvent:
    job_id: str
    type: str
    frac: float = 0.0
    msg: str = ""
    result: Any = None
    error: str | None = None


class FakeJobQueue:
    """Implements the ``pipeline.queue.JobQueue`` contract with one worker thread."""

    _ids = itertools.count(1)

    def __init__(self, runner: Callable[..., Any], *, workers: int = 1, ctx: Any = None) -> None:
        self.runner = runner
        self.ctx = ctx
        self._lock = threading.RLock()
        self._records: dict[str, FakeJobRecord] = {}
        self._order: list[str] = []
        self._pending: list[str] = []
        self._tokens: dict[str, Any] = {}
        self._subs: list[Callable[[Any], None]] = []
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="fake-queue", daemon=True)
        self._thread.start()
        self.submitted: list[Any] = []

    def _emit(self, ev: FakeJobEvent) -> None:
        for cb in list(self._subs):
            cb(ev)

    def submit(self, spec: Any) -> str:
        from vrphoto.core.types import CancelToken

        job_id = f"job-{next(self._ids)}"
        with self._lock:
            self._records[job_id] = FakeJobRecord(job_id, spec)
            self._order.append(job_id)
            self._tokens[job_id] = CancelToken()
            self.submitted.append(spec)
        self._emit(FakeJobEvent(job_id, "queued"))
        with self._lock:
            if self._records[job_id].status == "queued":
                self._pending.append(job_id)
                self._cv.notify_all()
        return job_id

    def _loop(self) -> None:
        from vrphoto.core.errors import Cancelled
        from vrphoto.core.types import ProgressReporter

        while True:
            with self._lock:
                while not self._pending and not self._stop:
                    self._cv.wait(0.1)
                if self._stop:
                    return
                job_id = self._pending.pop(0)
                rec = self._records[job_id]
                token = self._tokens[job_id]
                rec.status = "running"
                rec.started = time.time()
            self._emit(FakeJobEvent(job_id, "started"))

            def prog(frac: float, msg: str, _id: str = job_id, _rec: FakeJobRecord = rec) -> None:
                _rec.frac, _rec.msg = frac, msg
                self._emit(FakeJobEvent(_id, "progress", frac=frac, msg=msg))

            try:
                token.check()
                result = self.runner(rec.spec, self.ctx, progress=ProgressReporter(prog), cancel=token)
            except Cancelled:
                rec.status, rec.finished = "cancelled", time.time()
                self._emit(FakeJobEvent(job_id, "cancelled"))
            except Exception as e:  # noqa: BLE001
                rec.status, rec.error, rec.finished = "failed", str(e), time.time()
                self._emit(FakeJobEvent(job_id, "failed", error=str(e)))
            else:
                rec.status, rec.result, rec.finished, rec.frac = "done", result, time.time(), 1.0
                self._emit(FakeJobEvent(job_id, "finished", frac=1.0, result=result))

    def cancel(self, job_id: str) -> None:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            self._tokens[job_id].cancel()
            if job_id in self._pending:
                self._pending.remove(job_id)
                rec.status, rec.finished = "cancelled", time.time()
                queued_cancel = True
            else:
                queued_cancel = False
        if queued_cancel:
            self._emit(FakeJobEvent(job_id, "cancelled"))

    def cancel_all(self) -> None:
        for job_id in list(self._order):
            self.cancel(job_id)

    def subscribe(self, cb: Callable[[Any], None]) -> Callable[[], None]:
        self._subs.append(cb)

        def unsubscribe() -> None:
            if cb in self._subs:
                self._subs.remove(cb)

        return unsubscribe

    def jobs(self) -> list[FakeJobRecord]:
        with self._lock:
            return [self._records[j] for j in self._order]

    def get(self, job_id: str) -> FakeJobRecord:
        return self._records[job_id]

    def clear_finished(self) -> None:
        with self._lock:
            self._order = [j for j in self._order if self._records[j].status in ("queued", "running")]

    def shutdown(self, wait: bool = True) -> None:
        self.cancel_all()
        with self._lock:
            self._stop = True
            self._cv.notify_all()
        if wait:
            self._thread.join(timeout=5)


# --------------------------------------------------------------------------- runners
def fake_runner(spec: Any, ctx: Any = None, *, progress: Any = None, cancel: Any = None) -> Any:
    """Quick successful job: 5 progress steps, writes one output file."""
    from vrphoto.pipeline.jobs import JobResult

    for i in range(5):
        if cancel is not None:
            cancel.check()
        if progress is not None:
            progress(i / 5, f"Step {i + 1} of 5")
        time.sleep(0.005)
    from vrphoto.core.config import get_settings

    out_dir = Path(spec.common.output_dir) if spec.common.output_dir else get_settings().outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(spec.inputs[0]).stem}_{spec.kind}_LR.jpg"
    out.write_bytes(b"\xff\xd8fake")
    return JobResult(
        outputs={"image": str(out)}, scene_id=f"{Path(spec.inputs[0]).stem}-scene", elapsed_s=0.03
    )


class BlockingRunner:
    """Runner that waits (checking cancel) until released; for cancel tests."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, spec: Any, ctx: Any = None, *, progress: Any = None, cancel: Any = None) -> Any:
        from vrphoto.pipeline.jobs import JobResult

        self.started.set()
        if progress is not None:
            progress(0.1, "Waiting")
        while not self.release.is_set():
            if cancel is not None:
                cancel.check()
            time.sleep(0.01)
        return JobResult()


def failing_runner(spec: Any, ctx: Any = None, *, progress: Any = None, cancel: Any = None) -> Any:
    from vrphoto.core.errors import InferenceError

    raise InferenceError("The GPU ran out of memory", hint="Lower 'Max processing size'.")


class Prompts:
    """Records message boxes; answers questions with ``self.answer``."""

    def __init__(self) -> None:
        self.answer = True
        self.license_answer = True
        self.choice_answer: int | None = None  # ask_choice: the button index, None = closed
        self.choices: list[str] = []
        self.real_ask_choice = None
        self.questions: list[str] = []
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.licenses: list[str] = []

    def ask(self, parent, title, text, **kw) -> bool:  # noqa: ANN001
        self.questions.append(f"{title}: {text}")
        return self.answer

    def choose(self, parent, title, text, buttons, **kw) -> int | None:  # noqa: ANN001
        self.choices.append(f"{title}: {text}")
        return self.choice_answer

    def error(self, parent, title, exc, **kw) -> None:  # noqa: ANN001
        self.errors.append(f"{title}: {exc}")

    def info(self, parent, title, text) -> None:  # noqa: ANN001
        self.infos.append(f"{title}: {text}")

    def license(self, parent, name, license_name, **kw) -> bool:  # noqa: ANN001
        self.licenses.append(name)
        return self.license_answer


class FakeServer:
    """Stands in for webxr.server.ViewerServer."""

    instances: list[FakeServer] = []

    def __init__(self, settings: Any = None) -> None:
        self.running = False
        self.https_port = 8443
        self.http_port = 8080
        self.starts = 0
        self.stops = 0
        FakeServer.instances.append(self)

    def start(self) -> None:
        self.starts += 1
        self.running = True

    def stop(self) -> None:
        self.stops += 1
        self.running = False

    def urls(self) -> dict[str, str]:
        return {
            "https": "https://192.168.1.5:8443/",
            "https_2": "https://10.0.0.2:8443/",
            "local": "http://localhost:8080/",
        }


def make_image(path: Path, w: int = 320, h: int = 240, seed: int = 0) -> Path:
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 // w, yy * 255 // h, np.full_like(xx, 128)], -1).astype(np.uint8)
    img[h // 3 : h - 10, w // 3 : 2 * w // 3] = rng.integers(100, 255, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def make_video(path: Path, frames: int = 6, w: int = 160, h: int = 120) -> Path | None:
    """Tiny MP4 via OpenCV (None if this OpenCV cannot write MP4)."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    if not vw.isOpened():
        return None
    for i in range(frames):
        frame = np.full((h, w, 3), (i * 30) % 255, np.uint8)
        cv2.rectangle(frame, (20 + i * 5, 20), (80 + i * 5, 90), (0, 0, 255), -1)
        vw.write(frame)
    vw.release()
    return path if path.is_file() and path.stat().st_size > 0 else None


def wait_until(qtbot, cond: Callable[[], bool], timeout: int = 5000) -> None:
    qtbot.waitUntil(cond, timeout=timeout)
