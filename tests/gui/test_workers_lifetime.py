"""Lifetime of the per-task signals QObject (regression: rare segfault when a worker's
signals object was destroyed by the pool thread / the garbage collector)."""

from __future__ import annotations

import gc
import threading

import shiboken6
from PySide6.QtCore import QCoreApplication

from vrphoto.gui.workers import TaskRunner


def test_signals_are_owned_by_the_runner_and_deleted_on_the_ui_thread(qtbot):
    runner = TaskRunner()
    done = []
    w = runner.start(lambda: 42, pass_progress=False, pass_cancel=False, on_result=done.append)
    sig = w.signals
    assert sig.parent() is runner  # Qt owns the C++ object, not Python
    qtbot.waitUntil(lambda: done == [42] and not runner._workers, timeout=5000)
    # deleteLater runs on the next event-loop pass of the UI thread
    qtbot.waitUntil(lambda: not shiboken6.isValid(sig), timeout=5000)
    runner.shutdown(1000)


def test_dropping_python_references_never_deletes_live_signals(qtbot):
    runner = TaskRunner()
    release = threading.Event()
    results = []

    def slow():
        release.wait(5)
        return "ok"

    w = runner.start(slow, pass_progress=False, pass_cancel=False, on_result=results.append)
    sig = w.signals
    del w
    gc.collect()  # the garbage collector must not touch the Qt-owned signals object
    assert shiboken6.isValid(sig)
    release.set()
    qtbot.waitUntil(lambda: results == ["ok"], timeout=5000)
    runner.shutdown(1000)


def test_running_task_survives_its_runner_being_deleted(qtbot):
    runner = TaskRunner()
    release = threading.Event()
    w = runner.start(lambda: release.wait(5), pass_progress=False, pass_cancel=False)
    qtbot.waitUntil(lambda: w.started.is_set(), timeout=5000)
    runner.shutdown(50)  # task still running -> signals re-homed to the application
    sig = w.signals
    assert sig.parent() is QCoreApplication.instance()
    runner.deleteLater()
    qtbot.wait(50)
    release.set()  # the worker now emits finished on a re-homed, still valid object
    qtbot.waitUntil(lambda: w.done.is_set(), timeout=5000)
    qtbot.waitUntil(lambda: not shiboken6.isValid(sig), timeout=5000)


def test_many_short_tasks_with_gc_pressure(qtbot):
    runner = TaskRunner(max_threads=4)
    finished = []
    for i in range(200):
        runner.start(lambda i=i: i * 2, pass_progress=False, pass_cancel=False,
                     on_finished=lambda: finished.append(1))
        if i % 20 == 0:
            gc.collect()
    qtbot.waitUntil(lambda: len(finished) == 200, timeout=20000)
    gc.collect()
    runner.shutdown(2000)
