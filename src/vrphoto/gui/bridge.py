"""Qt bridge around :class:`vrphoto.pipeline.queue.JobQueue`.

The queue (no Qt dependency, see docs/CONTRACTS.md section 10) calls its subscribers from
worker threads.  :class:`QueueBridge` subscribes once, relays every :class:`JobEvent`
through a *queued* signal connection onto the Qt thread and re-emits it as typed
signals that widgets can connect to safely.  Progress events are coalesced per job so a
fast job cannot flood the event loop.

The real queue is created lazily (``pipeline.queue`` may be slow to import or missing in
a broken install); if that fails :attr:`QueueBridge.error` explains why and
:meth:`submit` raises a friendly :class:`~vrphoto.core.errors.VRPhotoError`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal, Slot

from ..core.errors import VRPhotoError
from ..core.log import get_logger

log = get_logger(__name__)

TERMINAL = ("finished", "failed", "cancelled")
#: lifecycle order; events that would move a job backwards are stale and dropped (the queue
#: may deliver "queued" from the submitting thread after the worker already reported "started")
_RANK = {"queued": 0, "running": 1, "finished": 2, "failed": 2, "cancelled": 2}
_EVENT_STATUS = {
    "queued": "queued",
    "started": "running",
    "progress": "running",
    "finished": "finished",
    "failed": "failed",
    "cancelled": "cancelled",
}


class QueueBridge(QObject):
    """Owns a JobQueue and exposes its events as Qt signals (always on the UI thread).

    Signals:
        jobEvent(object): every event (after coalescing), the raw ``JobEvent``.
        jobQueued(str), jobStarted(str): job id.
        jobProgress(str, float, str): job id, fraction 0..1, message.
        jobFinished(str, object): job id, ``JobResult``.
        jobFailed(str, str): job id, error message.
        jobCancelled(str): job id.
        countsChanged(int, int): (running, queued) after each lifecycle event.
    """

    jobEvent = Signal(object)
    jobQueued = Signal(str)
    jobStarted = Signal(str)
    jobProgress = Signal(str, float, str)
    jobFinished = Signal(str, object)
    jobFailed = Signal(str, str)
    jobCancelled = Signal(str)
    countsChanged = Signal(int, int)

    _relay = Signal(object)  # emitted from worker threads -> _dispatch on the UI thread

    def __init__(
        self,
        queue: Any | None = None,
        *,
        runner: Callable[..., Any] | None = None,
        workers: int = 1,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._queue = queue
        self._runner = runner
        self._workers = max(1, int(workers))
        self._unsubscribe: Callable[[], None] | None = None
        self._lock = threading.Lock()
        self._pending_progress: dict[str, Any] = {}
        self._flush_scheduled = False
        self._status: dict[str, str] = {}
        self.error: str | None = None
        self._closed = False
        self._relay.connect(self._dispatch, Qt.ConnectionType.QueuedConnection)
        if queue is not None:
            self._attach(queue)

    # ------------------------------------------------------------------ queue lifecycle
    def _attach(self, queue: Any) -> None:
        self._queue = queue
        self._unsubscribe = queue.subscribe(self._on_event_thread)

    def ensure_queue(self) -> Any:
        """Return the JobQueue, creating it on first use.  Raises VRPhotoError if unavailable."""
        if self._queue is not None:
            return self._queue
        if self._closed:
            raise VRPhotoError("The job queue has been shut down.")
        try:
            from ..pipeline.queue import JobQueue

            q = (
                JobQueue(self._runner, workers=self._workers)
                if self._runner
                else JobQueue(workers=self._workers)
            )
        except Exception as e:  # missing module, import error in a dependency...
            self.error = f"The processing engine could not be started: {e}"
            log.exception("Cannot create the job queue")
            raise VRPhotoError(
                self.error, hint="Try restarting the app. If it keeps happening, reinstall VR Photo Studio."
            ) from e
        self.error = None
        self._attach(q)
        return q

    @property
    def available(self) -> bool:
        try:
            self.ensure_queue()
        except VRPhotoError:
            return False
        return True

    # ------------------------------------------------------------------ API (UI thread)
    def submit(self, spec: Any) -> str:
        """Queue a job; returns its id.  Raises VRPhotoError if the queue is unavailable."""
        q = self.ensure_queue()
        job_id = q.submit(spec)
        self._status.setdefault(job_id, "queued")
        return job_id

    def cancel(self, job_id: str) -> None:
        if self._queue is not None:
            self._queue.cancel(job_id)

    def cancel_all(self) -> None:
        if self._queue is not None:
            self._queue.cancel_all()

    def jobs(self) -> list[Any]:
        return list(self._queue.jobs()) if self._queue is not None else []

    def get(self, job_id: str) -> Any:
        if self._queue is None:
            raise KeyError(job_id)
        return self._queue.get(job_id)

    def clear_finished(self) -> None:
        if self._queue is not None:
            self._queue.clear_finished()
        self._status = {k: v for k, v in self._status.items() if v not in TERMINAL}

    def counts(self) -> tuple[int, int]:
        """``(running, queued)`` as last reported by events (or the queue records)."""
        if self._queue is not None:
            try:
                recs = self._queue.jobs()
                running = sum(1 for r in recs if getattr(r, "status", "") == "running")
                queued = sum(1 for r in recs if getattr(r, "status", "") == "queued")
                return running, queued
            except Exception:  # pragma: no cover - defensive
                pass
        running = sum(1 for s in self._status.values() if s == "running")
        queued = sum(1 for s in self._status.values() if s == "queued")
        return running, queued

    def active_count(self) -> int:
        r, q = self.counts()
        return r + q

    def shutdown(self, wait: bool = False, *, wait_s: float = 0.0) -> None:
        """Cancel all jobs, stop listening and shut the queue down.  Idempotent.

        Args:
            wait: join the queue's worker threads (may block for as long as a job ignores
                cancellation - avoid on the UI thread).
            wait_s: otherwise, wait at most this long for cancelled jobs to stop, so they
                can terminate their subprocesses (add-on workers, ffmpeg) before the app
                exits.  Daemon worker threads would simply be killed at exit.
        """
        if self._closed:
            return
        self._closed = True
        q = self._queue
        if q is None:
            return
        try:
            q.cancel_all()
        except Exception:  # pragma: no cover - defensive
            log.debug("cancel_all failed during shutdown", exc_info=True)
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:  # pragma: no cover - defensive
                pass
            self._unsubscribe = None
        if not wait and wait_s > 0:
            self._wait_idle(q, wait_s)
        try:
            q.shutdown(wait=wait)
        except Exception:
            log.debug("Queue shutdown failed", exc_info=True)

    @staticmethod
    def _wait_idle(q: Any, timeout: float) -> bool:
        """Wait until no job of ``q`` is queued or running (at most ``timeout`` s)."""

        def busy() -> bool:
            try:
                return any(getattr(r, "status", "") in ("queued", "running") for r in q.jobs())
            except Exception:  # pragma: no cover - defensive
                return False

        if not busy():
            return True
        log.info("Waiting up to %.0f s for running jobs to stop", timeout)
        deadline = time.monotonic() + timeout
        wait_fn = getattr(q, "wait", None)  # pipeline.queue.JobQueue extension
        if callable(wait_fn):
            try:
                if wait_fn(None, timeout=timeout):
                    return True
            except Exception:
                log.debug("JobQueue.wait failed", exc_info=True)
        while busy() and time.monotonic() < deadline:
            time.sleep(0.05)
        idle = not busy()
        if not idle:
            log.warning("A job did not stop within %.0f s; exiting anyway", timeout)
        return idle

    # ------------------------------------------------------------------ worker-thread side
    def _on_event_thread(self, ev: Any) -> None:
        """JobQueue subscriber; may run on any thread.  Never touches widgets."""
        if getattr(ev, "type", None) == "progress":
            with self._lock:
                self._pending_progress[ev.job_id] = ev
                if self._flush_scheduled:
                    return
                self._flush_scheduled = True
            self._relay.emit(None)  # None = flush pending progress
        else:
            self._relay.emit(ev)

    # ------------------------------------------------------------------ UI-thread side
    def _take_progress(self) -> list[Any]:
        with self._lock:
            evs = list(self._pending_progress.values())
            self._pending_progress.clear()
            self._flush_scheduled = False
        return evs

    @Slot(object)
    def _dispatch(self, ev: Any) -> None:
        if ev is None:
            for p in self._take_progress():
                if self._status.get(p.job_id) in TERMINAL:
                    continue  # a late progress event after the job ended
                self._status[p.job_id] = "running"
                self.jobEvent.emit(p)
                self.jobProgress.emit(p.job_id, float(p.frac), str(p.msg or ""))
            return
        job_id = ev.job_id
        t = ev.type
        new_status = _EVENT_STATUS.get(t)
        old_status = self._status.get(job_id)
        if new_status is not None and old_status is not None and _RANK[new_status] < _RANK[old_status]:
            log.debug("Dropping stale %s event for job %s (already %s)", t, job_id, old_status)
            return
        if old_status in TERMINAL and t in TERMINAL:
            log.debug("Dropping duplicate terminal event %s for job %s", t, job_id)
            return
        if t in TERMINAL:
            with self._lock:  # progress superseded by the terminal event
                self._pending_progress.pop(job_id, None)
        self.jobEvent.emit(ev)
        if t == "queued":
            self._status[job_id] = "queued"
            self.jobQueued.emit(job_id)
        elif t == "started":
            self._status[job_id] = "running"
            self.jobStarted.emit(job_id)
        elif t == "finished":
            self._status[job_id] = "finished"
            self.jobFinished.emit(job_id, ev.result)
        elif t == "failed":
            self._status[job_id] = "failed"
            self.jobFailed.emit(job_id, str(ev.error or "Unknown error"))
        elif t == "cancelled":
            self._status[job_id] = "cancelled"
            self.jobCancelled.emit(job_id)
        else:
            log.debug("Ignoring unknown job event type %r", t)
            return
        self.countsChanged.emit(*self.counts())
