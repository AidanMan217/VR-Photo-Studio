"""TaskRunner log levels: harmless shutdown details stay out of the GUI Log panel."""

from __future__ import annotations

import logging
import threading

from vrphoto.gui import workers
from vrphoto.gui.workers import TaskRunner

LOGGER = "vrphoto.gui.workers"


def test_still_stopping_message_is_debug_only(qtbot, caplog):
    tr = TaskRunner()
    release = threading.Event()
    w = tr.start(lambda: release.wait(10), pass_progress=False, pass_cancel=False, name="addon-status")
    assert w.started.wait(5)  # running (a queued task would just be dropped by shutdown)
    try:
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            tr.shutdown(msecs=50)
        assert tr.pool in workers._ORPHAN_POOLS  # really detached while still running
        records = [r for r in caplog.records if r.name == LOGGER]
        stopping = [r for r in records if "still stopping" in r.getMessage()]
        assert len(stopping) == 1
        assert stopping[0].levelno == logging.DEBUG
        assert "addon-status" in stopping[0].getMessage()
        assert not [r for r in records if r.levelno >= logging.INFO]
    finally:
        release.set()
        assert tr.pool.waitForDone(5000)
        workers._prune_orphans()


def test_clean_shutdown_logs_nothing(qtbot, caplog):
    tr = TaskRunner()
    done = []
    tr.start(lambda: 1, pass_progress=False, pass_cancel=False, on_result=done.append)
    qtbot.waitUntil(lambda: bool(done), timeout=5000)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        tr.shutdown()
    assert not [r for r in caplog.records if r.name == LOGGER and "still stopping" in r.getMessage()]
