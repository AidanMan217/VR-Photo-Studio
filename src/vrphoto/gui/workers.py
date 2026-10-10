"""Background work for the GUI: run any function in a thread pool with progress,
cancellation and results delivered back on the Qt (UI) thread.

Usage::

    self.tasks = TaskRunner(self)                       # one per window/dialog
    w = self.tasks.start(ensure_model, "lama",
                         on_progress=self._on_progress,  # (frac, msg) on the UI thread
                         on_result=self._on_done,        # (result) on the UI thread
                         on_error=self._on_error)        # (exception) on the UI thread
    w.cancel()                                          # cooperative (CancelToken)

The callable receives ``progress=`` (a :class:`ProgressReporter`) and ``cancel=`` (a
:class:`CancelToken`) keyword arguments unless ``pass_progress`` / ``pass_cancel`` are
False.  A :class:`~vrphoto.core.errors.Cancelled` exception is reported through
``on_cancelled`` instead of ``on_error``.  ``on_finished`` always runs last.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThreadPool, Signal, Slot

from ..core.errors import Cancelled
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter

log = get_logger(__name__)

_ids = itertools.count(1)
PROGRESS_INTERVAL_S = 0.04
#: How long closing a dialog waits for its background tasks.  Tasks still running after
#: that (e.g. a firewall check or an adb call that cannot be interrupted) are cancelled and
#: detached (see :meth:`TaskRunner.shutdown`) instead of freezing the window.
DIALOG_CLOSE_WAIT_MS = 300
_ORPHAN_POOLS: set[QThreadPool] = set()


def _prune_orphans() -> None:
    """Forget detached thread pools whose last task has ended."""
    for pool in list(_ORPHAN_POOLS):
        if pool.activeThreadCount() == 0:
            _ORPHAN_POOLS.discard(pool)


class _Signals(QObject):
    """Signals emitted from the worker thread; connected to the TaskRunner (UI thread)."""

    progress = Signal(int, float, str)
    result = Signal(int, object)
    error = Signal(int, object)
    cancelled = Signal(int)
    finished = Signal(int)


class Worker:
    """One unit of background work (create through :meth:`TaskRunner.start`).

    A plain Python object whose :meth:`run` is handed to ``QThreadPool.start`` as a
    callable, so Qt never holds a pointer to an object Python may garbage-collect.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        pass_progress: bool = True,
        pass_cancel: bool = True,
        name: str = "",
    ) -> None:
        self.id = next(_ids)
        self.name = name or getattr(fn, "__name__", "task")
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.pass_progress = pass_progress
        self.pass_cancel = pass_cancel
        self.cancel_token = CancelToken()
        self.signals = _Signals()
        self.started = threading.Event()
        self.done = threading.Event()
        self.callbacks: dict[str, Callable[..., Any] | None] = {}
        self._last_report = (0.0, -1.0, "")  # (time, frac, msg) of the last emitted progress

    # ---------------------------------------------------------------- control
    def cancel(self) -> None:
        """Request cooperative cancellation (the function must call ``cancel.check()``)."""
        self.cancel_token.cancel()

    @property
    def cancelled(self) -> bool:
        return self.cancel_token.cancelled

    # ---------------------------------------------------------------- thread body
    def _report(self, frac: float, msg: str) -> None:
        """Progress callback (worker thread); throttled to ~25 updates/s unless the message changes."""
        now = time.monotonic()
        t, f, m = self._last_report
        msg = str(msg)
        if msg == m and frac < 1.0 and now - t < PROGRESS_INTERVAL_S and abs(frac - f) < 0.05:
            return
        self._last_report = (now, float(frac), msg)
        self.signals.progress.emit(self.id, float(frac), msg)

    def run(self) -> None:
        """Thread body: call the function and emit the outcome."""
        self.started.set()
        kwargs = dict(self.kwargs)
        if self.pass_progress:
            kwargs["progress"] = ProgressReporter(self._report)
        if self.pass_cancel:
            kwargs["cancel"] = self.cancel_token
        try:
            if self.cancel_token.cancelled:
                raise Cancelled("Operation cancelled")
            result = self.fn(*self.args, **kwargs)
        except Cancelled:
            self.signals.cancelled.emit(self.id)
        except BaseException as e:  # noqa: BLE001 - reported to the UI
            if self.cancel_token.cancelled:
                self.signals.cancelled.emit(self.id)
            else:
                log.debug("Background task %s failed", self.name, exc_info=True)
                self.signals.error.emit(self.id, e)
        else:
            if self.cancel_token.cancelled:
                self.signals.cancelled.emit(self.id)
            else:
                self.signals.result.emit(self.id, result)
        finally:
            self.done.set()
            self.signals.finished.emit(self.id)


class TaskRunner(QObject):
    """Owns a thread pool and the workers started on it; dispatches callbacks on the UI thread."""

    def __init__(self, parent: QObject | None = None, *, max_threads: int = 2) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max(1, max_threads))
        self._workers: dict[int, Worker] = {}
        self._closed = False

    # ---------------------------------------------------------------- API
    def start(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_progress: Callable[[float, str], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        pass_progress: bool = True,
        pass_cancel: bool = True,
        name: str = "",
        **kwargs: Any,
    ) -> Worker:
        """Run ``fn(*args, **kwargs)`` in the pool; returns the :class:`Worker` handle."""
        _prune_orphans()
        w = Worker(fn, args, kwargs, pass_progress=pass_progress, pass_cancel=pass_cancel, name=name)
        w.callbacks = {
            "result": on_result,
            "error": on_error,
            "progress": on_progress,
            "cancelled": on_cancelled,
            "finished": on_finished,
        }
        w.signals.progress.connect(self._on_progress)
        w.signals.result.connect(self._on_result)
        w.signals.error.connect(self._on_error)
        w.signals.cancelled.connect(self._on_cancelled)
        w.signals.finished.connect(self._on_finished)
        self._workers[w.id] = w
        if self._closed:
            w.cancel()
        self.pool.start(w.run)
        return w

    def active(self) -> list[Worker]:
        """Workers that have not finished yet."""
        return [w for w in self._workers.values() if not w.done.is_set()]

    def busy(self) -> bool:
        return bool(self.active())

    def cancel_all(self) -> None:
        for w in list(self._workers.values()):
            w.cancel()

    def wait(self, msecs: int = 5000) -> bool:
        """Block until all workers are done (use only at shutdown / in tests)."""
        return bool(self.pool.waitForDone(msecs))

    def shutdown(self, msecs: int = 3000) -> None:
        """Cancel everything, wait up to ``msecs`` and stop dispatching callbacks.

        If a task is still running afterwards (it ignores cancellation for a while), the
        thread pool is detached from this object and kept alive until the task ends, so
        deleting the owning dialog never blocks the UI waiting for it.
        """
        self._closed = True
        self.cancel_all()
        self.pool.clear()  # tasks that never started will not run at all
        for w in self._workers.values():
            if not w.started.is_set():
                w.done.set()
        finished = self.wait(msecs)
        for w in list(self._workers.values()):
            w.callbacks = {}
        if not finished:
            log.info("Background task(s) still stopping: %s", ", ".join(w.name for w in self.active()))
            self.pool.setParent(None)
            _ORPHAN_POOLS.add(self.pool)
            _prune_orphans()

    # ---------------------------------------------------------------- dispatch (UI thread)
    def _call(self, wid: int, key: str, *args: Any) -> None:
        w = self._workers.get(wid)
        if w is None:
            return
        cb = w.callbacks.get(key)
        if cb is None:
            return
        try:
            cb(*args)
        except Exception:  # never let a UI callback kill the dispatcher
            log.exception("Error in %s callback of background task %s", key, w.name)

    @Slot(int, float, str)
    def _on_progress(self, wid: int, frac: float, msg: str) -> None:
        w = self._workers.get(wid)
        if w is not None and not w.cancelled:
            self._call(wid, "progress", frac, msg)

    @Slot(int, object)
    def _on_result(self, wid: int, value: object) -> None:
        self._call(wid, "result", value)

    @Slot(int, object)
    def _on_error(self, wid: int, exc: object) -> None:
        self._call(wid, "error", exc)

    @Slot(int)
    def _on_cancelled(self, wid: int) -> None:
        self._call(wid, "cancelled")

    @Slot(int)
    def _on_finished(self, wid: int) -> None:
        self._call(wid, "finished")
        self._workers.pop(wid, None)
