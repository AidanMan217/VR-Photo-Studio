"""JobQueue lifecycle with fake runners (no real processing)."""

from __future__ import annotations

import threading
import time

import pytest

from vrphoto.core.errors import Cancelled, InputError
from vrphoto.core.types import CancelToken, ProgressReporter
from vrphoto.pipeline.jobs import JobResult, StereoJob
from vrphoto.pipeline.queue import JobEvent, JobQueue, JobRecord, error_message

TIMEOUT = 10.0


class Recorder:
    """Thread-safe event collector."""

    def __init__(self) -> None:
        self.events: list[JobEvent] = []
        self.lock = threading.Lock()

    def __call__(self, ev: JobEvent) -> None:
        with self.lock:
            self.events.append(ev)

    def types(self, job_id: str) -> list[str]:
        with self.lock:
            return [e.type for e in self.events if e.job_id == job_id]

    def lifecycle(self, job_id: str) -> list[str]:
        return [t for t in self.types(job_id) if t != "progress"]


def spec(name: str = "a.jpg") -> StereoJob:
    return StereoJob(inputs=[name])


def ok_runner(sp, ctx, *, progress, cancel):
    progress(0.25, "quarter")
    progress(0.75, "three quarters")
    return JobResult(outputs={"out": sp.inputs[0] + ".done"})


@pytest.fixture
def q():
    queues: list[JobQueue] = []

    def make(runner=ok_runner, **kw) -> JobQueue:
        jq = JobQueue(runner, ctx=object(), **kw)
        queues.append(jq)
        return jq

    yield make
    for jq in queues:
        jq.shutdown(wait=True)


def test_submit_progress_finish(q):
    jq = q()
    rec = Recorder()
    jq.subscribe(rec)
    jid = jq.submit(spec())
    assert jq.wait(jid, TIMEOUT)
    assert rec.lifecycle(jid) == ["queued", "started", "finished"]
    progress = [e for e in rec.events if e.type == "progress"]
    assert [p.msg for p in progress] == ["quarter", "three quarters"]
    assert [p.frac for p in progress] == [0.25, 0.75]
    final = rec.events[-1]
    assert final.type == "finished" and final.result.outputs == {"out": "a.jpg.done"} and final.frac == 1.0
    r = jq.get(jid)
    assert isinstance(r, JobRecord)
    assert r.status == "done" and r.done and r.frac == 1.0 and r.error is None
    assert r.result.outputs["out"] == "a.jpg.done"
    assert r.created <= r.started <= r.finished


def test_runner_receives_ctx_reporter_and_token():
    seen = {}

    def runner(sp, ctx, *, progress, cancel):
        seen.update(sp=sp, ctx=ctx, progress=progress, cancel=cancel)
        return JobResult()

    marker = object()
    jq = JobQueue(runner, ctx=marker)
    try:
        s = spec()
        jq.wait(jq.submit(s), TIMEOUT)
    finally:
        jq.shutdown()
    assert seen["sp"] is s and seen["ctx"] is marker
    assert isinstance(seen["progress"], ProgressReporter)
    assert isinstance(seen["cancel"], CancelToken)


def test_vrphoto_error_becomes_failed_event_with_hint(q):
    def runner(sp, ctx, *, progress, cancel):
        raise InputError("Bad photo", hint="Use a JPEG")

    jq = q(runner)
    rec = Recorder()
    jq.subscribe(rec)
    jid = jq.submit(spec())
    jq.wait(jid, TIMEOUT)
    assert rec.lifecycle(jid) == ["queued", "started", "failed"]
    ev = rec.events[-1]
    assert ev.error == "Bad photo Use a JPEG"
    r = jq.get(jid)
    assert r.status == "failed" and r.error == "Bad photo Use a JPEG" and r.result is None


def test_unexpected_error_is_reported_and_logged(q, caplog):
    def runner(sp, ctx, *, progress, cancel):
        raise ZeroDivisionError("boom")

    jq = q(runner)
    jid = jq.submit(spec())
    with caplog.at_level("ERROR"):
        jq.wait(jid, TIMEOUT)
    r = jq.get(jid)
    assert r.status == "failed"
    assert "Unexpected error" in r.error and "ZeroDivisionError" in r.error and "boom" in r.error
    assert any(rec.exc_info for rec in caplog.records if "failed" in rec.getMessage())


def test_error_message_helper():
    assert error_message(InputError("A", hint="B")) == "A B"
    assert error_message(InputError("Same hint", hint="hint")) == "Same hint"
    assert "memory" in error_message(MemoryError()).lower()
    assert error_message(ValueError("x")).startswith("Unexpected error (ValueError: x)")


def test_fifo_order_single_worker(q):
    order = []

    def runner(sp, ctx, *, progress, cancel):
        order.append(sp.inputs[0])
        time.sleep(0.01)
        return JobResult()

    jq = q(runner)
    ids = [jq.submit(spec(f"{i}.jpg")) for i in range(5)]
    assert jq.wait(None, TIMEOUT)
    assert order == [f"{i}.jpg" for i in range(5)]
    assert [r.job_id for r in jq.jobs()] == ids


def test_multiple_workers_run_concurrently(q):
    barrier = threading.Barrier(2, timeout=5)

    def runner(sp, ctx, *, progress, cancel):
        barrier.wait()  # deadlocks (-> BrokenBarrierError) unless two jobs run at once
        return JobResult()

    jq = q(runner, workers=2)
    assert jq.workers == 2
    a, b = jq.submit(spec("a")), jq.submit(spec("b"))
    assert jq.wait(None, TIMEOUT)
    assert jq.get(a).status == jq.get(b).status == "done"


def test_cancel_queued_job_never_runs(q):
    gate = threading.Event()
    ran = []

    def runner(sp, ctx, *, progress, cancel):
        ran.append(sp.inputs[0])
        gate.wait(5)
        return JobResult()

    jq = q(runner)
    rec = Recorder()
    jq.subscribe(rec)
    first = jq.submit(spec("first"))
    second = jq.submit(spec("second"))
    jq.cancel(second)
    assert rec.lifecycle(second) == ["queued", "cancelled"]
    assert jq.get(second).status == "cancelled"
    gate.set()
    assert jq.wait(None, TIMEOUT)
    assert ran == ["first"]
    assert jq.get(first).status == "done"
    assert rec.lifecycle(second) == ["queued", "cancelled"]  # no late "started"


def test_cancel_running_job(q):
    started = threading.Event()

    def runner(sp, ctx, *, progress, cancel):
        started.set()
        for _ in range(500):
            cancel.check()
            progress(0.1, "working")
            time.sleep(0.01)
        return JobResult()

    jq = q(runner)
    rec = Recorder()
    jq.subscribe(rec)
    jid = jq.submit(spec())
    assert started.wait(5)
    jq.cancel(jid)
    assert jq.wait(jid, TIMEOUT)
    assert rec.lifecycle(jid) == ["queued", "started", "cancelled"]
    assert jq.get(jid).status == "cancelled"


def test_non_cancelled_error_after_cancel_counts_as_cancelled(q):
    """E.g. ffmpeg killed by the cancel raises an OSError: that is still a cancellation."""
    started = threading.Event()

    def runner(sp, ctx, *, progress, cancel):
        started.set()
        while not cancel.cancelled:
            time.sleep(0.01)
        raise BrokenPipeError("pipe closed")

    jq = q(runner)
    jid = jq.submit(spec())
    started.wait(5)
    jq.cancel(jid)
    jq.wait(jid, TIMEOUT)
    assert jq.get(jid).status == "cancelled"


def test_cancel_all_and_unknown_ids(q):
    gate = threading.Event()

    def runner(sp, ctx, *, progress, cancel):
        while not gate.is_set():
            cancel.check()
            time.sleep(0.01)
        return JobResult()

    jq = q(runner)
    ids = [jq.submit(spec(str(i))) for i in range(3)]
    time.sleep(0.05)
    jq.cancel("does-not-exist")  # ignored
    jq.cancel_all()
    assert jq.wait(None, TIMEOUT)
    assert {jq.get(i).status for i in ids} == {"cancelled"}
    jq.cancel(ids[0])  # finished jobs: ignored


def test_runner_raising_cancelled_directly(q):
    def runner(sp, ctx, *, progress, cancel):
        raise Cancelled("stop")

    jq = q(runner)
    jid = jq.submit(spec())
    jq.wait(jid, TIMEOUT)
    assert jq.get(jid).status == "cancelled"


def test_subscribers_unsubscribe_and_bad_subscriber(q):
    jq = q()
    rec1, rec2 = Recorder(), Recorder()

    def bad(ev):
        raise RuntimeError("subscriber bug")

    jq.subscribe(bad)
    unsub = jq.subscribe(rec1)
    jq.subscribe(rec2)
    unsub()
    jid = jq.submit(spec())
    jq.wait(jid, TIMEOUT)
    assert rec1.events == []
    assert rec2.lifecycle(jid) == ["queued", "started", "finished"]
    assert jq.get(jid).status == "done"  # the failing subscriber did not break the job
    with pytest.raises(TypeError):
        jq.subscribe("not callable")  # type: ignore[arg-type]


def test_progress_throttling_keeps_message_changes(q):
    def runner(sp, ctx, *, progress, cancel):
        for i in range(200):
            progress(i / 400, "same message")
        progress(0.6, "new message")
        progress(1.0, "same final")
        return JobResult()

    jq = q(runner)
    rec = Recorder()
    jq.subscribe(rec)
    jid = jq.submit(spec())
    jq.wait(jid, TIMEOUT)
    msgs = [e.msg for e in rec.events if e.type == "progress"]
    assert msgs.count("same message") < 200  # throttled
    assert "new message" in msgs and "same final" in msgs
    assert jq.get(jid).frac == 1.0


def test_progress_values_are_clamped(q):
    def runner(sp, ctx, *, progress, cancel):
        progress(-1, "low")
        progress(5, "high")
        return JobResult()

    jq = q(runner)
    rec = Recorder()
    jq.subscribe(rec)
    jq.wait(jq.submit(spec()), TIMEOUT)
    fr = [e.frac for e in rec.events if e.type == "progress"]
    assert fr == [0.0, 1.0]


def test_jobs_are_snapshots_and_clear_finished(q):
    gate = threading.Event()

    def runner(sp, ctx, *, progress, cancel):
        if sp.inputs[0] == "slow":
            gate.wait(5)
        return JobResult()

    jq = q(runner)
    done_id = jq.submit(spec("fast"))
    jq.wait(done_id, TIMEOUT)
    slow = jq.submit(spec("slow"))
    snap = jq.jobs()
    snap[0].status = "tampered"
    assert jq.get(done_id).status == "done"
    jq.clear_finished()
    assert [r.job_id for r in jq.jobs()] == [slow]
    with pytest.raises(KeyError):
        jq.get(done_id)
    gate.set()
    jq.wait(slow, TIMEOUT)
    assert jq.counts() == (0, 0)


def test_counts_and_wait_timeout(q):
    gate = threading.Event()

    def runner(sp, ctx, *, progress, cancel):
        gate.wait(5)
        return JobResult()

    jq = q(runner)
    a = jq.submit(spec("a"))
    jq.submit(spec("b"))
    deadline = time.time() + 5
    while jq.counts() != (1, 1) and time.time() < deadline:
        time.sleep(0.01)
    assert jq.counts() == (1, 1)
    assert jq.wait(a, timeout=0.05) is False
    gate.set()
    assert jq.wait(None, TIMEOUT)


def test_non_jobresult_return_is_wrapped(q):
    class Obj:
        outputs = {"x": "y"}

    jq = q(lambda sp, ctx, *, progress, cancel: Obj())
    jid = jq.submit(spec())
    jq.wait(jid, TIMEOUT)
    assert jq.get(jid).result.outputs == {"x": "y"}


def test_submit_validation_and_shutdown():
    jq = JobQueue(ok_runner, ctx=object())
    with pytest.raises(TypeError):
        jq.submit({"kind": "stereo"})  # type: ignore[arg-type]
    gate = threading.Event()
    started = threading.Event()

    def slow(sp, ctx, *, progress, cancel):
        started.set()
        while not gate.is_set():
            cancel.check()
            time.sleep(0.01)
        return JobResult()

    jq2 = JobQueue(slow, ctx=object())
    running = jq2.submit(spec("r"))
    queued = jq2.submit(spec("q"))
    started.wait(5)
    jq2.shutdown(wait=True)  # cancels pending work and joins the workers
    assert jq2.closed
    assert jq2.get(running).status == "cancelled"
    assert jq2.get(queued).status == "cancelled"
    with pytest.raises(RuntimeError):
        jq2.submit(spec())
    jq2.shutdown()  # idempotent
    with jq:  # context manager shuts down
        jq.wait(jq.submit(spec()), TIMEOUT)
    assert jq.closed


def test_shutdown_without_cancel_lets_queued_jobs_finish():
    ran = []

    def runner(sp, ctx, *, progress, cancel):
        time.sleep(0.02)
        ran.append(sp.inputs[0])
        return JobResult()

    jq = JobQueue(runner, ctx=object())
    for i in range(3):
        jq.submit(spec(str(i)))
    jq.shutdown(wait=True, cancel_pending=False)
    assert ran == ["0", "1", "2"]


def test_default_runner_and_lazy_context(make_job, photo_png):
    """JobQueue() runs real jobs with run_job and a shared RunContext."""
    jq = JobQueue()
    rec = Recorder()
    jq.subscribe(rec)
    try:
        jid = jq.submit(make_job("depth", [photo_png]))
        assert jq.wait(jid, 60)
        r = jq.get(jid)
        assert r.status == "done", r.error
        assert any(k.startswith("Depth map") for k in r.result.outputs)
        from vrphoto.pipeline.context import RunContext

        assert isinstance(jq.ctx, RunContext)
        assert rec.lifecycle(jid) == ["queued", "started", "finished"]
    finally:
        jq.shutdown()


def test_vrphoto_error_after_cancel_counts_as_cancelled(q):
    """E.g. an add-on worker killed by the cancel reports an AddonError: still a cancellation."""
    from vrphoto.core.errors import AddonError

    started = threading.Event()

    def runner(sp, ctx, *, progress, cancel):
        started.set()
        while not cancel.cancelled:
            time.sleep(0.01)
        raise AddonError("The worker stopped unexpectedly (exit code 1).")

    rec = Recorder()
    jq = q(runner)
    jq.subscribe(rec)
    jid = jq.submit(spec())
    started.wait(5)
    jq.cancel(jid)
    jq.wait(jid, TIMEOUT)
    r = jq.get(jid)
    assert r.status == "cancelled" and r.error is None
    assert rec.lifecycle(jid)[-1] == "cancelled"


def test_progress_after_cancel_keeps_saying_cancelling(q):
    started = threading.Event()
    go = threading.Event()

    def runner(sp, ctx, *, progress, cancel):
        started.set()
        go.wait(5)
        progress(0.5, "Rendering frame 12/180")
        cancel.check()
        return JobResult()

    rec = Recorder()
    jq = q(runner)
    jq.subscribe(rec)
    jid = jq.submit(spec())
    started.wait(5)
    jq.cancel(jid)
    assert jq.get(jid).msg == "Cancelling..."
    go.set()
    jq.wait(jid, TIMEOUT)
    with rec.lock:
        msgs = [e.msg for e in rec.events if e.job_id == jid and e.type == "progress"]
    assert msgs == ["Cancelling..."]
    assert jq.get(jid).status == "cancelled"
