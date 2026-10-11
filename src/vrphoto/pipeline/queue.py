"""Background job queue (no Qt dependency).

:class:`JobQueue` runs job specs on a pool of ``workers`` threads in FIFO order.  Each job
gets its own :class:`~vrphoto.core.types.CancelToken`; :meth:`JobQueue.cancel` stops a
queued job immediately and asks a running one to stop at its next check.  Lifecycle
events (:class:`JobEvent`) are delivered to subscribers from the thread where they happen
(the submitting / cancelling thread for ``queued`` and queued-job ``cancelled`` events,
the worker thread for everything else); the GUI marshals them to the Qt thread.

Per job the event order is always ``queued`` -> (``started`` -> ``progress``* ->
``finished`` | ``failed`` | ``cancelled``) or ``queued`` -> ``cancelled``.  A runner
exception becomes a ``failed`` event whose ``error`` is the friendly message (a
:class:`~vrphoto.core.errors.VRPhotoError`'s text plus its hint); the traceback goes to
the log (at DEBUG level for a ``VRPhotoError``, an expected error, so the Log panel shows
one readable line).  Any exception raised after the job was asked to cancel counts as
``cancelled``.
"""

from __future__ import annotations

import dataclasses
import itertools
import queue as _queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import UI_LOG, Cancelled, VRPhotoError
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter
from .jobs import JobBase, JobResult

log = get_logger(__name__)

STATUSES = ("queued", "running", "done", "failed", "cancelled")
TERMINAL = frozenset({"done", "failed", "cancelled"})
EVENT_TYPES = ("queued", "started", "progress", "finished", "failed", "cancelled")
#: Minimum interval between two progress events of one job (a changed message is always sent).
PROGRESS_INTERVAL_S = 0.05


@dataclass
class JobRecord:
    """State of one submitted job (a snapshot when returned by :meth:`JobQueue.jobs`)."""

    job_id: str
    spec: JobBase
    status: str = "queued"  # queued | running | done | failed | cancelled
    frac: float = 0.0
    msg: str = ""
    result: JobResult | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    @property
    def done(self) -> bool:
        """True once the job reached a terminal status."""
        return self.status in TERMINAL


@dataclass
class JobEvent:
    """A lifecycle event of one job."""

    job_id: str
    type: str  # queued | started | progress | finished | failed | cancelled
    frac: float = 0.0
    msg: str = ""
    result: JobResult | None = None
    error: str | None = None


def error_message(exc: BaseException) -> str:
    """Friendly one-line message for an exception (VRPhotoError text + hint)."""
    if isinstance(exc, VRPhotoError):
        msg = str(exc) or type(exc).__name__
        hint = getattr(exc, "hint", None)
        return f"{msg} {hint}" if hint and hint not in msg else msg
    if isinstance(exc, MemoryError):
        return "Not enough memory to finish this job. Lower 'Max processing size' or close other programs."
    if isinstance(exc, OSError):  # e.g. disk full / folder not writable outside the job's own checks
        from .run import _friendly

        friendly = _friendly(exc)
        if friendly is not None:
            return error_message(friendly)
    detail = str(exc).strip()
    return (
        f"Unexpected error ({type(exc).__name__}{': ' + detail if detail else ''}). "
        f"Details are in the log ({UI_LOG})."
    )


def _default_runner() -> Callable[..., JobResult]:
    from .run import run_job

    return run_job


class JobQueue:
    """FIFO job queue on ``workers`` background threads.

    Args:
        runner: ``runner(spec, ctx, *, progress, cancel) -> JobResult``; default
            :func:`vrphoto.pipeline.run.run_job`.
        workers: number of jobs that may run at the same time (default 1).
        ctx: shared :class:`~vrphoto.pipeline.context.RunContext`; created on first use
            when omitted (models and the depth cache are then shared by all jobs).
    """

    def __init__(
        self,
        runner: Callable[..., JobResult] | None = None,
        *,
        workers: int = 1,
        ctx: Any | None = None,
    ) -> None:
        self._runner = runner if runner is not None else _default_runner()
        self._ctx = ctx
        self._ctx_lock = threading.Lock()
        self._lock = threading.RLock()
        self._records: dict[str, JobRecord] = {}
        self._tokens: dict[str, CancelToken] = {}
        self._subs: dict[int, Callable[[JobEvent], None]] = {}
        self._sub_ids = itertools.count(1)
        self._pending: _queue.Queue[str | None] = _queue.Queue()
        self._closed = False
        self._idle = threading.Condition(self._lock)
        self._settled: set[str] = set()
        n = max(1, int(workers))
        self._threads = [
            threading.Thread(target=self._worker, name=f"vrphoto-job-{i + 1}", daemon=True) for i in range(n)
        ]
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------------ properties
    @property
    def workers(self) -> int:
        return len(self._threads)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def ctx(self) -> Any:
        """The shared run context (created on first access)."""
        with self._ctx_lock:
            if self._ctx is None:
                from .context import RunContext

                self._ctx = RunContext()
            return self._ctx

    # ------------------------------------------------------------------ subscribers
    def subscribe(self, cb: Callable[[JobEvent], None]) -> Callable[[], None]:
        """Call ``cb(event)`` for every event; returns a function that unsubscribes."""
        if not callable(cb):
            raise TypeError("subscriber must be callable")
        with self._lock:
            sid = next(self._sub_ids)
            self._subs[sid] = cb

        def unsubscribe() -> None:
            with self._lock:
                self._subs.pop(sid, None)

        return unsubscribe

    def _emit(self, ev: JobEvent) -> None:
        with self._lock:
            subs = list(self._subs.values())
        for cb in subs:
            try:
                cb(ev)
            except Exception:
                log.exception("Job event subscriber failed on %s/%s", ev.job_id, ev.type)

    # ------------------------------------------------------------------ API
    def submit(self, spec: JobBase) -> str:
        """Queue a job; returns its id.  Raises RuntimeError after :meth:`shutdown`."""
        if not isinstance(spec, JobBase):
            raise TypeError(f"submit() needs a job spec, got {type(spec).__name__}")
        job_id = uuid.uuid4().hex[:12]
        rec = JobRecord(job_id=job_id, spec=spec)
        with self._lock:
            if self._closed:
                raise RuntimeError("The job queue has been shut down.")
            self._records[job_id] = rec
            self._tokens[job_id] = CancelToken()
        log.info("Queued %s job %s (%d input(s))", spec.kind, job_id, len(spec.inputs))
        self._emit(JobEvent(job_id, "queued", msg="Waiting"))
        self._pending.put(job_id)
        return job_id

    def cancel(self, job_id: str) -> None:
        """Cancel a queued job (immediately) or ask a running job to stop.  Unknown or
        finished jobs are ignored."""
        ev: JobEvent | None = None
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None or rec.status in TERMINAL:
                return
            tok = self._tokens.get(job_id)
            if tok is not None:
                tok.cancel()
            if rec.status == "queued":
                rec.status = "cancelled"
                rec.msg = "Cancelled"
                rec.finished = time.time()
                ev = JobEvent(job_id, "cancelled", frac=rec.frac, msg="Cancelled")
            else:
                rec.msg = "Cancelling..."
        if ev is not None:
            log.info("Cancelled queued job %s", job_id)
            self._settle(ev)
        else:
            log.info("Cancelling running job %s", job_id)

    def cancel_all(self) -> None:
        """Cancel every queued and running job."""
        with self._lock:
            ids = [j for j, r in self._records.items() if r.status not in TERMINAL]
        for j in ids:
            self.cancel(j)

    def jobs(self) -> list[JobRecord]:
        """Snapshots of all known jobs in submission order."""
        with self._lock:
            return [dataclasses.replace(r) for r in self._records.values()]

    def get(self, job_id: str) -> JobRecord:
        """Snapshot of one job (raises KeyError for unknown ids)."""
        with self._lock:
            return dataclasses.replace(self._records[job_id])

    def clear_finished(self) -> None:
        """Forget finished, failed and cancelled jobs."""
        with self._lock:
            for j in [j for j, r in self._records.items() if r.status in TERMINAL and j in self._settled]:
                self._records.pop(j, None)
                self._tokens.pop(j, None)
                self._settled.discard(j)

    def counts(self) -> tuple[int, int]:
        """``(running, queued)``."""
        with self._lock:
            running = sum(1 for r in self._records.values() if r.status == "running")
            queued = sum(1 for r in self._records.values() if r.status == "queued")
        return running, queued

    def wait(self, job_id: str | None = None, timeout: float | None = None) -> bool:
        """Block until ``job_id`` (or every job) is finished; returns False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._idle:
            while True:
                if job_id is not None:
                    done = job_id not in self._records or job_id in self._settled
                else:
                    done = all(j in self._settled for j in self._records)
                if done:
                    return True
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._idle.wait(0.25 if remaining is None else min(0.25, remaining))

    def shutdown(self, wait: bool = True, *, cancel_pending: bool = True) -> None:
        """Stop accepting jobs.  ``cancel_pending`` (default) cancels queued and running
        jobs; otherwise queued jobs still run.  ``wait`` joins the worker threads."""
        with self._lock:
            if self._closed:
                already = True
            else:
                already = False
                self._closed = True
        if cancel_pending:
            self.cancel_all()
        if not already:
            for _ in self._threads:
                self._pending.put(None)
        if wait:
            for t in self._threads:
                if t is not threading.current_thread():
                    t.join()

    def __enter__(self) -> JobQueue:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown(wait=True)

    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        while True:
            job_id = self._pending.get()
            if job_id is None:
                return
            try:
                self._run_one(job_id)
            except Exception:  # pragma: no cover - never let a worker thread die
                log.exception("Job worker crashed while handling %s", job_id)

    def _run_one(self, job_id: str) -> None:
        with self._lock:
            rec = self._records.get(job_id)
            tok = self._tokens.get(job_id)
            if rec is None or tok is None or rec.status != "queued":
                return  # cancelled (or cleared) while waiting
            rec.status = "running"
            rec.started = time.time()
            rec.msg = "Starting"
        log.info("Started %s job %s", rec.spec.kind, job_id)
        self._emit(JobEvent(job_id, "started", msg="Starting"))

        last = [0.0, None]  # monotonic time, message of the last progress event

        def on_progress(frac: float, msg: str = "") -> None:
            f = min(1.0, max(0.0, float(frac)))
            if tok.cancelled:
                msg = "Cancelling..."  # keep showing that the job is stopping
            with self._lock:
                rec.frac = f
                if msg:
                    rec.msg = msg
            now = time.monotonic()
            if msg == last[1] and now - float(last[0] or 0.0) < PROGRESS_INTERVAL_S and f < 1.0:
                return
            last[0], last[1] = now, msg
            self._emit(JobEvent(job_id, "progress", frac=f, msg=msg))

        ev: JobEvent
        try:
            result = self._runner(rec.spec, self.ctx, progress=ProgressReporter(on_progress), cancel=tok)
        except BaseException as e:  # noqa: BLE001 - every failure becomes an event; workers never die
            if isinstance(e, Cancelled) or tok.cancelled:
                # after the user pressed Cancel, whatever error the interrupted work raises
                # (a killed add-on worker, a broken ffmpeg pipe...) just means "stopped"
                with self._lock:
                    rec.status, rec.msg, rec.finished = "cancelled", "Cancelled", time.time()
                if isinstance(e, Cancelled):
                    log.info("Job %s cancelled", job_id)
                else:
                    log.info("Job %s cancelled (it stopped with: %s)", job_id, error_message(e), exc_info=e)
                ev = JobEvent(job_id, "cancelled", frac=rec.frac, msg="Cancelled")
            else:
                msg = error_message(e)
                if isinstance(e, VRPhotoError):
                    # an expected, user-facing error (bad input, missing add-on...): one readable
                    # line in the Log panel; the traceback only helps debugging
                    log.error("Job %s failed: %s", job_id, msg)
                    log.debug("Traceback of failed job %s", job_id, exc_info=e)
                else:
                    log.error("Job %s failed with an unexpected error", job_id, exc_info=e)
                with self._lock:
                    rec.status, rec.error, rec.msg, rec.finished = "failed", msg, "Failed", time.time()
                ev = JobEvent(job_id, "failed", frac=rec.frac, msg="Failed", error=msg)
        else:
            if not isinstance(result, JobResult):
                result = JobResult(outputs=dict(getattr(result, "outputs", None) or {}))
            with self._lock:
                rec.status, rec.result, rec.frac, rec.msg = "done", result, 1.0, "Done"
                rec.finished = time.time()
            log.info("Job %s finished in %.1f s", job_id, (rec.finished or 0.0) - (rec.started or 0.0))
            ev = JobEvent(job_id, "finished", frac=1.0, msg="Done", result=result)
        self._settle(ev)

    def _settle(self, ev: JobEvent) -> None:
        """Deliver a terminal event, then mark the job settled (for :meth:`wait`)."""
        self._emit(ev)
        with self._lock:
            self._settled.add(ev.job_id)
            self._idle.notify_all()


__all__ = ["JobQueue", "JobRecord", "JobEvent", "error_message", "STATUSES", "EVENT_TYPES", "TERMINAL"]
