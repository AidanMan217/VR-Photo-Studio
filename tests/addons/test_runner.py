"""run_process / run_worker: streaming, protocol parsing, errors, cancel, timeout."""

from __future__ import annotations

import sys
import threading
import time

import pytest
from _addon_helpers import raw_worker, wait_dead

from vrphoto.addons.manifest import workers_dir
from vrphoto.addons.runner import LogSink, WorkerRun, error_hint, run_process, run_worker
from vrphoto.core.errors import AddonError, Cancelled
from vrphoto.core.types import CancelToken, ProgressReporter

DUMMY = workers_dir() / "dummy_worker.py"


class Recorder:
    def __init__(self):
        self.events: list[tuple[float, str]] = []
        self.lock = threading.Lock()

    def __call__(self, frac, msg):
        with self.lock:
            self.events.append((frac, msg))

    @property
    def reporter(self):
        return ProgressReporter(self)


def _cancel_after(token: CancelToken, seconds: float) -> threading.Timer:
    t = threading.Timer(seconds, token.cancel)
    t.start()
    return t


# ------------------------------------------------------------------ run_process
def test_run_process_lines_and_log(tmp_path):
    lines: list[str] = []
    log = tmp_path / "x.log"
    code = "import sys; print('one'); sys.stdout.write('a\\rb\\r\\nc\\n'); print('err', file=sys.stderr)"
    res = run_process([sys.executable, "-c", code], log_sink=log, on_line=lines.append)
    assert res.ok and res.returncode == 0
    assert lines[:4] == ["one", "a", "b", "c"]
    assert "err" in lines
    text = log.read_text(encoding="utf-8")
    assert "one" in text and "exit code 0" in text


def test_run_process_nonzero_exit_is_returned():
    res = run_process([sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"])
    assert res.returncode == 3 and not res.ok and res.tail == ["bad"]
    assert "bad" in res.tail_text()


def test_run_process_stdin():
    res = run_process([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"], stdin_data="hey")
    assert res.tail == ["HEY"]


def test_run_process_missing_program():
    with pytest.raises(AddonError, match="program not found"):
        run_process(["definitely-not-a-program-xyz"], title="Tool")


def test_run_process_cancel_is_fast():
    token = CancelToken()
    _cancel_after(token, 0.3)
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        run_process([sys.executable, "-c", "import time; time.sleep(30)"], cancel=token)
    assert time.monotonic() - t0 < 3


def test_run_process_timeout():
    t0 = time.monotonic()
    with pytest.raises(AddonError, match="took longer"):
        run_process([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5, title="Sleeper")
    assert time.monotonic() - t0 < 4


def test_run_process_env_is_clean_by_default(monkeypatch):
    monkeypatch.setenv("PYTHONSTARTUP", "/nonexistent.py")
    res = run_process([sys.executable, "-c", "import os; print(os.environ.get('PYTHONSTARTUP', 'none'))"])
    assert res.tail[-1] == "none"


def test_log_sink_truncates_big_file(tmp_path):
    p = tmp_path / "big.log"
    p.write_text("x" * 2000, encoding="utf-8")
    with LogSink(p, max_bytes=1000) as sink:
        sink.write("fresh")
    assert p.read_text(encoding="utf-8").strip().endswith("fresh")
    assert p.stat().st_size < 200


# ------------------------------------------------------------------ run_worker (dummy, _common)
def test_worker_echo_with_progress(tmp_path):
    rec = Recorder()
    st = WorkerRun()
    out = run_worker(sys.executable, DUMMY, "echo", {"image": tmp_path / "a.png", "n": 1}, {"steps": 4},
                     tmp_path / "out", progress=rec.reporter, log_sink=tmp_path / "run.log", state=st)
    assert out["echo"]["inputs"]["image"] == str(tmp_path / "a.png")  # Path made JSON-able
    assert out["echo"]["params"] == {"steps": 4}
    assert (tmp_path / "out" / "echo.json").is_file()
    fracs = [f for f, _ in rec.events]
    assert len(fracs) == 4 and fracs == sorted(fracs) and fracs[-1] == pytest.approx(1.0)
    assert rec.events[0][1] == "step 1/4"
    assert st.hello and st.hello["worker"] == "dummy"


def test_worker_library_prints_do_not_break_protocol(tmp_path):
    st = WorkerRun()
    out = run_worker(sys.executable, DUMMY, "echo", {}, {"junk": True, "steps": 1}, tmp_path,
                     log_sink=tmp_path / "run.log", state=st)
    assert "fake" not in out and "echo" in out  # the fake result printed by a "library" was ignored
    log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "hello from a chatty library" in log and "raw fd-1 junk" in log


def test_worker_error_propagates_with_hint(tmp_path):
    with pytest.raises(AddonError) as ei:
        run_worker(sys.executable, DUMMY, "echo", {}, {"fail": "Model exploded", "steps": 0}, tmp_path,
                   title="Dummy")
    assert str(ei.value) == "Dummy: Model exploded"
    assert ei.value.hint == "dummy hint"


def test_worker_oom_gets_gpu_hint(tmp_path):
    with pytest.raises(AddonError) as ei:
        run_worker(sys.executable, DUMMY, "echo", {}, {"oom": True, "steps": 0}, tmp_path)
    assert "out of memory" in str(ei.value).lower()
    assert "close other GPU-heavy apps" in ei.value.hint


def test_worker_crash_without_result(tmp_path):
    with pytest.raises(AddonError, match=r"exit code 3.*about to crash"):
        run_worker(sys.executable, DUMMY, "crash", {}, {"code": 3}, tmp_path)


def test_worker_unknown_task(tmp_path):
    with pytest.raises(AddonError, match="Unknown task"):
        run_worker(sys.executable, DUMMY, "nope", {}, {}, tmp_path)


def test_worker_cancel_kills_within_2s(tmp_path):
    token = CancelToken()
    rec = Recorder()
    _cancel_after(token, 0.8)
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        run_worker(sys.executable, DUMMY, "sleep", {}, {"seconds": 60}, tmp_path, cancel=token,
                   progress=rec.reporter)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.8 + 2.0
    assert rec.events, "progress should have been reported before the cancel"


def test_worker_timeout(tmp_path):
    with pytest.raises(AddonError, match="took longer"):
        run_worker(sys.executable, DUMMY, "sleep", {}, {"seconds": 60}, tmp_path, timeout=0.7)


def test_worker_check_and_prefetch_modes(tmp_path, monkeypatch):
    info = run_worker(sys.executable, DUMMY, "", {}, {}, tmp_path, args=["--check", "--no-gpu"], send_request=False)
    assert info["gpu_required"] is False and info["python"]
    from vrphoto.addons.env import clean_env

    env = clean_env({"VRPHOTO_ADDON_DIR": str(tmp_path), "DUMMY_CHECK_FAIL": "1"})
    with pytest.raises(AddonError, match="dummy check failed"):
        run_worker(sys.executable, DUMMY, "", {}, {}, tmp_path, args=["--check"], send_request=False, env=env)
    env = clean_env({"VRPHOTO_ADDON_DIR": str(tmp_path)})
    out = run_worker(sys.executable, DUMMY, "", {}, {}, tmp_path, args=["--prefetch"], send_request=False, env=env)
    assert (tmp_path / "prefetch.txt").read_text(encoding="utf-8") == "ok"
    assert out["file"].endswith("prefetch.txt")


# ------------------------------------------------------------------ run_worker (raw scripts)
def test_raw_worker_junk_lines_ignored(tmp_path):
    w = raw_worker(tmp_path, "w.py", """
        import sys
        sys.stdin.readline()
        print("Loading weights... 100%")
        print("{not json")
        print('{"type": "bogus", "x": 1}')
        print('[1, 2, 3]')
        print('{"type": "progress", "frac": 0.5, "msg": "half"}')
        print('{"type": "log", "level": "warning", "msg": "careful"}')
        print('{"type": "result", "outputs": {"ok": 1}}')
        print('{"type": "error", "message": "late error must be ignored"}')
    """)
    rec = Recorder()
    st = WorkerRun()
    out = run_worker(sys.executable, w, "t", {}, {}, tmp_path, progress=rec.reporter, state=st,
                     log_sink=tmp_path / "l.log")
    assert out == {"ok": 1}
    assert rec.events == [(0.5, "half")]
    assert st.junk_lines == 4
    assert "careful" in (tmp_path / "l.log").read_text(encoding="utf-8")


def test_raw_worker_request_line(tmp_path):
    w = raw_worker(tmp_path, "w.py", """
        import json, sys
        req = json.loads(sys.stdin.readline())
        rest = sys.stdin.read()
        print(json.dumps({"type": "result", "outputs": {"req": req, "rest": rest}}))
    """)
    out = run_worker(sys.executable, w, "image_to_3d", {"image": "a.png"}, {"seed": 3}, tmp_path / "o")
    req = out["req"]
    assert req["type"] == "request" and req["protocol"] == 1 and req["task"] == "image_to_3d"
    assert req["inputs"] == {"image": "a.png"} and req["params"] == {"seed": 3}
    assert req["out_dir"] == str(tmp_path / "o")
    assert out["rest"] == ""  # stdin closed after the request


def test_raw_worker_stderr_tail_in_message(tmp_path):
    w = raw_worker(tmp_path, "w.py", """
        import sys
        sys.stderr.write("Traceback...\\nModuleNotFoundError: No module named 'torch'\\n")
        sys.exit(1)
    """)
    with pytest.raises(AddonError) as ei:
        run_worker(sys.executable, w, "t", {}, {}, tmp_path)
    assert "No module named 'torch'" in str(ei.value)
    assert "Reinstall" in ei.value.hint


def test_error_hint_table():
    assert "GPU ran out of memory" in error_hint("torch.OutOfMemoryError: CUDA out of memory")
    assert "NVIDIA GPU" in error_hint("RuntimeError: Found no NVIDIA driver on your system")
    assert "Hugging Face" in error_hint("huggingface_hub.errors.GatedRepoError: 401")
    assert "disk is full" in error_hint("OSError: [Errno 28] No space left on device")
    assert error_hint("all good") is None
    assert error_hint(None, "") is None


def test_run_process_does_not_hang_on_orphaned_pipe(tmp_path):
    """A grandchild that inherits stdout must not keep run_process waiting forever."""
    pidfile = tmp_path / "gc.pid"
    code = (
        "import subprocess, sys, pathlib\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(c.pid))\n"
        "print('parent done')\n"
    )
    t0 = time.monotonic()
    res = run_process([sys.executable, "-c", code])
    assert res.ok and "parent done" in res.tail
    assert time.monotonic() - t0 < 10
    if sys.platform.startswith("linux"):
        assert wait_dead(int(pidfile.read_text()))


# ------------------------------------------------------------------ review regressions
def test_worker_lingering_after_result_is_killed(tmp_path, monkeypatch):
    """A worker whose interpreter does not exit after the result (CUDA teardown hang, stray
    non-daemon thread) must not block the job forever."""
    from vrphoto.addons import runner

    monkeypatch.setattr(runner, "_EXIT_GRACE_S", 0.5)
    w = raw_worker(tmp_path, "w.py", """
        import os, sys, time
        sys.stdin.readline()
        with open(sys.argv[1], "w") as fh:
            fh.write(str(os.getpid()))
        print('{"type": "result", "outputs": {"done": true}}', flush=True)
        time.sleep(60)
    """)
    pid_file = tmp_path / "pid"
    t0 = time.monotonic()
    out = run_worker(sys.executable, w, "t", {}, {}, tmp_path, args=[str(pid_file)], log_sink=tmp_path / "l.log")
    assert out == {"done": True}
    assert time.monotonic() - t0 < 10
    assert wait_dead(int(pid_file.read_text()))
    assert "did not exit" in (tmp_path / "l.log").read_text(encoding="utf-8")


def test_worker_error_then_hang_still_raises(tmp_path, monkeypatch):
    from vrphoto.addons import runner

    monkeypatch.setattr(runner, "_EXIT_GRACE_S", 0.3)
    w = raw_worker(tmp_path, "w.py", """
        import sys, time
        sys.stdin.readline()
        print('{"type": "error", "message": "boom"}', flush=True)
        time.sleep(60)
    """)
    with pytest.raises(AddonError, match="boom"):
        run_worker(sys.executable, w, "t", {}, {}, tmp_path, timeout=30)


def test_worker_gets_parent_pid(tmp_path):
    import os

    w = raw_worker(tmp_path, "w.py", """
        import json, os, sys
        sys.stdin.readline()
        print(json.dumps({"type": "result", "outputs": {"ppid": os.environ.get("VRPHOTO_PARENT_PID")}}))
    """)
    out = run_worker(sys.executable, w, "t", {}, {}, tmp_path)
    assert out["ppid"] == str(os.getpid())


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses /proc to watch the orphan")
def test_worker_exits_when_the_app_dies(tmp_path):
    """Killing the app (crash / Task Manager) must not leave a GPU-hogging worker behind:
    the worker runs in its own session, so only its parent watchdog can stop it."""
    import os
    import signal
    import subprocess

    pid_file = tmp_path / "worker.pid"
    app = tmp_path / "app.py"
    app.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from vrphoto.addons.runner import run_worker\n"
        f"run_worker(sys.executable, {str(DUMMY)!r}, 'sleep', {{}}, "
        f"{{'seconds': 120, 'pid_file': {str(pid_file)!r}}}, Path({str(tmp_path)!r}))\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(app)])
    try:
        deadline = time.monotonic() + 15
        while not (pid_file.is_file() and pid_file.read_text().strip()):
            assert time.monotonic() < deadline, "worker did not start"
            assert proc.poll() is None, "app died early"
            time.sleep(0.05)
        worker_pid = int(pid_file.read_text())
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(5)
        assert wait_dead(worker_pid, timeout=6.0), "orphaned worker kept running"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_parent_watchdog_needs_a_foreign_pid(monkeypatch):
    import os

    sys.path.insert(0, str(workers_dir()))
    try:
        import _common
    finally:
        sys.path.remove(str(workers_dir()))
    monkeypatch.setattr(_common, "_watchdog_started", False)
    monkeypatch.delenv("VRPHOTO_PARENT_PID", raising=False)
    assert _common.start_parent_watchdog() is False
    monkeypatch.setenv("VRPHOTO_PARENT_PID", str(os.getpid()))
    assert _common.start_parent_watchdog() is False
    monkeypatch.setenv("VRPHOTO_PARENT_PID", "not-a-pid")
    assert _common.start_parent_watchdog() is False
    assert _common._parent_alive_posix(os.getpid()) or sys.platform == "win32"


def test_result_in_hand_is_not_turned_into_a_timeout(tmp_path, monkeypatch):
    from vrphoto.addons import runner

    monkeypatch.setattr(runner, "_EXIT_GRACE_S", 3.0)
    w = raw_worker(tmp_path, "w.py", """
        import sys, time
        sys.stdin.readline()
        print('{"type": "result", "outputs": {"ok": 1}}', flush=True)
        time.sleep(60)
    """)
    assert run_worker(sys.executable, w, "t", {}, {}, tmp_path, timeout=2.0) == {"ok": 1}


def test_error_hint_for_locked_files():
    assert "in use" in error_hint("PermissionError: [WinError 32] The process cannot access the file because it is "
                                  "being used by another process")
    assert "gated" in error_hint("GatedRepoError: Access to model x/y is restricted")
