"""QueueBridge (JobQueue events -> Qt signals on the UI thread) and TaskRunner."""

from __future__ import annotations

import threading
import time

import pytest
from gui_helpers import BlockingRunner, FakeJobQueue, failing_runner, fake_runner

from vrphoto.core.errors import VRPhotoError
from vrphoto.gui.bridge import QueueBridge
from vrphoto.gui.workers import TaskRunner
from vrphoto.pipeline.jobs import StereoJob

pytestmark = pytest.mark.gui


@pytest.fixture
def bridge_factory(qtbot):
    made = []

    def make(runner=fake_runner):
        q = FakeJobQueue(runner)
        b = QueueBridge(q)
        made.append((b, q))
        return b, q

    yield make
    for b, q in made:
        b.shutdown()
        q.shutdown(wait=True)


def _spec(tmp_path):
    s = StereoJob(inputs=[str(tmp_path / "a.jpg")])
    s.common.output_dir = str(tmp_path / "out")
    return s


def test_events_arrive_on_ui_thread_in_order(qtbot, bridge_factory, tmp_path):
    b, q = bridge_factory()
    main = threading.get_ident()
    seen: list[tuple[str, bool]] = []
    b.jobQueued.connect(lambda j: seen.append(("queued", threading.get_ident() == main)))
    b.jobStarted.connect(lambda j: seen.append(("started", threading.get_ident() == main)))
    b.jobProgress.connect(lambda j, f, m: seen.append(("progress", threading.get_ident() == main)))
    results = []
    b.jobFinished.connect(lambda j, r: results.append((j, r)))
    with qtbot.waitSignal(b.jobFinished, timeout=5000):
        job_id = b.submit(_spec(tmp_path))
    assert results[0][0] == job_id
    assert results[0][1].scene_id == "a-scene"
    kinds = [k for k, _ in seen]
    assert kinds[0] == "queued" and kinds[1] == "started" and "progress" in kinds
    assert all(on_main for _, on_main in seen)
    assert b.counts() == (0, 0)


def test_failed_job_emits_failed(qtbot, bridge_factory, tmp_path):
    b, q = bridge_factory(failing_runner)
    with qtbot.waitSignal(b.jobFailed, timeout=5000) as blocker:
        b.submit(_spec(tmp_path))
    assert "out of memory" in blocker.args[1]


def test_cancel_running_and_queued(qtbot, bridge_factory, tmp_path):
    runner = BlockingRunner()
    b, q = bridge_factory(runner)
    cancelled: list[str] = []
    b.jobCancelled.connect(cancelled.append)
    first = b.submit(_spec(tmp_path))
    second = b.submit(_spec(tmp_path))
    assert runner.started.wait(5)
    qtbot.waitUntil(lambda: b.counts() == (1, 1), timeout=5000)
    b.cancel(second)
    qtbot.waitUntil(lambda: second in cancelled, timeout=5000)
    b.cancel(first)
    qtbot.waitUntil(lambda: first in cancelled, timeout=5000)
    assert b.active_count() == 0


def test_progress_is_coalesced(qtbot, bridge_factory, tmp_path):
    def chatty(spec, ctx=None, *, progress=None, cancel=None):
        from vrphoto.pipeline.jobs import JobResult

        for i in range(2000):
            progress(i / 2000, "tick")
        progress(1.0, "last")
        return JobResult()

    b, q = bridge_factory(chatty)
    got: list[float] = []
    b.jobProgress.connect(lambda j, f, m: got.append(f))
    with qtbot.waitSignal(b.jobFinished, timeout=10000):
        b.submit(_spec(tmp_path))
    assert 0 < len(got) < 2001
    assert got == sorted(got)


def test_lazy_queue_unavailable_raises_friendly_error(qtbot, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "vrphoto.pipeline.queue", None)  # import fails
    b = QueueBridge()
    with pytest.raises(VRPhotoError, match="processing engine"):
        b.submit(StereoJob(inputs=["x.jpg"]))
    assert b.error and not b.available
    b.shutdown()


def test_lazy_queue_uses_pipeline_queue(qtbot, monkeypatch, tmp_path):
    import sys
    import types

    mod = types.ModuleType("vrphoto.pipeline.queue")
    mod.JobQueue = FakeJobQueue  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vrphoto.pipeline.queue", mod)
    b = QueueBridge(runner=fake_runner)
    with qtbot.waitSignal(b.jobFinished, timeout=5000):
        b.submit(_spec(tmp_path))
    q = b.ensure_queue()
    assert isinstance(q, FakeJobQueue)
    b.shutdown(wait=True)


def test_real_job_queue_with_fake_runner(qtbot, tmp_path):
    """Integration with the real pipeline.queue.JobQueue (skipped until it exists)."""
    queue_mod = pytest.importorskip("vrphoto.pipeline.queue")
    q = queue_mod.JobQueue(fake_runner, workers=1)
    b = QueueBridge(q)
    try:
        with qtbot.waitSignal(b.jobFinished, timeout=10000) as blocker:
            job_id = b.submit(_spec(tmp_path))
        assert blocker.args[0] == job_id
        assert b.get(job_id).status in ("done", "finished")
    finally:
        b.shutdown(wait=True)


# --------------------------------------------------------------------------- TaskRunner
def test_task_runner_result_progress_on_ui_thread(qtbot):
    tr = TaskRunner()
    main = threading.get_ident()
    calls: list[tuple] = []

    def work(x, *, progress, cancel):
        progress(0.5, "half")
        return x * 2

    tr.start(
        work,
        21,
        on_progress=lambda f, m: calls.append(("p", f, m, threading.get_ident() == main)),
        on_result=lambda r: calls.append(("r", r, threading.get_ident() == main)),
        on_finished=lambda: calls.append(("f",)),
    )
    qtbot.waitUntil(lambda: ("f",) in calls, timeout=5000)
    assert ("r", 42, True) in calls
    assert ("p", 0.5, "half", True) in calls
    assert calls[-1] == ("f",)
    assert not tr.busy()
    tr.shutdown()


def test_task_runner_error_and_cancel(qtbot):
    tr = TaskRunner()
    errors, cancelled = [], []

    def bad(*, progress, cancel):
        raise ValueError("boom")

    tr.start(bad, on_error=errors.append)
    qtbot.waitUntil(lambda: bool(errors), timeout=5000)
    assert isinstance(errors[0], ValueError)

    started = threading.Event()

    def slow(*, progress, cancel):
        started.set()
        while True:
            cancel.check()
            time.sleep(0.01)

    w = tr.start(slow, on_cancelled=lambda: cancelled.append(True), on_error=errors.append)
    assert started.wait(5)
    w.cancel()
    qtbot.waitUntil(lambda: bool(cancelled), timeout=5000)
    assert len(errors) == 1
    tr.shutdown()


def test_task_runner_without_progress_kwargs(qtbot):
    tr = TaskRunner()
    out = []
    tr.start(lambda a, b: a + b, 1, 2, pass_progress=False, pass_cancel=False, on_result=out.append)
    qtbot.waitUntil(lambda: bool(out), timeout=5000)
    assert out == [3]
    tr.shutdown()


def test_task_runner_shutdown_detaches_stuck_task(qtbot):
    from vrphoto.gui import workers

    tr = TaskRunner()
    release = threading.Event()
    tr.start(lambda: release.wait(10), pass_progress=False, pass_cancel=False)
    time.sleep(0.05)
    tr.shutdown(msecs=50)
    assert tr.pool in workers._ORPHAN_POOLS
    release.set()
    assert tr.pool.waitForDone(5000)
    workers._prune_orphans()
    assert tr.pool not in workers._ORPHAN_POOLS


def test_tasks_started_after_shutdown_are_cancelled(qtbot):
    tr = TaskRunner()
    tr.shutdown()
    seen: list[str] = []
    ran: list[bool] = []

    def work(*, progress, cancel):
        ran.append(True)

    w = tr.start(
        work, on_cancelled=lambda: seen.append("cancelled"), on_result=lambda r: seen.append("result")
    )
    qtbot.waitUntil(lambda: w.done.is_set(), timeout=5000)
    qtbot.waitUntil(lambda: bool(seen), timeout=5000)
    assert seen == ["cancelled"] and not ran


class _ManualQueue:
    """Queue stub whose events are pushed by the test (any order, any thread)."""

    def __init__(self) -> None:
        self.cb = None

    def subscribe(self, cb):
        self.cb = cb
        return lambda: None

    def submit(self, spec):
        return "j1"

    def jobs(self):
        return []

    def cancel_all(self):
        pass

    def shutdown(self, wait=True):
        pass


def test_out_of_order_and_duplicate_events_are_dropped(qtbot):
    from gui_helpers import FakeJobEvent

    q = _ManualQueue()
    b = QueueBridge(q)
    seen: list[str] = []
    b.jobEvent.connect(lambda ev: seen.append(ev.type))
    b.submit(StereoJob(inputs=["x.jpg"]))
    t = threading.Thread(
        target=lambda: [
            q.cb(FakeJobEvent("j1", "started")),
            q.cb(FakeJobEvent("j1", "progress", frac=0.5, msg="half")),
        ]
    )
    t.start()
    t.join()
    q.cb(FakeJobEvent("j1", "queued"))  # late "queued" from the submitting thread
    q.cb(FakeJobEvent("j1", "finished", frac=1.0))
    q.cb(FakeJobEvent("j1", "cancelled"))  # duplicate terminal event
    q.cb(FakeJobEvent("j1", "progress", frac=0.9))  # late progress
    qtbot.waitUntil(lambda: "finished" in seen, timeout=5000)
    qtbot.wait(50)
    assert seen == ["started", "progress", "finished"]
    assert b._status["j1"] == "finished"
    b.shutdown()
