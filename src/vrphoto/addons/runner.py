"""Subprocess plumbing for add-ons: run tools and protocol workers with logging, progress,
cooperative cancel (kills the whole process tree) and timeouts.

Two entry points:

* :func:`run_process` - run a command (uv, COLMAP, Brush, ffmpeg ...), stream its merged
  stdout/stderr line by line into a log file and an optional ``on_line`` callback.
* :func:`run_worker` - run an add-on worker script (``<python> -u worker.py``), send the
  JSON request (:mod:`vrphoto.addons.protocol`) and translate the replies into progress
  callbacks, a result dict or an :class:`~vrphoto.core.errors.AddonError`.
"""

from __future__ import annotations

import collections
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from ..core.errors import UI_ADDONS, UI_SETTINGS_MODELS, AddonError, Cancelled
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter
from . import protocol
from .env import assign_to_kill_job, clean_env, kill_orphans, kill_process_tree, popen_kwargs

log = get_logger(__name__)

LineCallback = Callable[[str], None]
_POLL_S = 0.05
_ORPHAN_PIPE_S = 3.0  # stop waiting for EOF this long after the process itself exited
_EXIT_GRACE_S = 20.0  # a worker that sent its result/error but does not exit is killed after this
PARENT_PID_ENV = "VRPHOTO_PARENT_PID"  # workers exit on their own when this process dies
_TAIL_LINES = 60
_LOG_MAX_BYTES = 8_000_000


# --------------------------------------------------------------------------- log sink
class LogSink:
    """Thread-safe append-only text log (one per add-on install / run).

    Lines are timestamped.  The file is truncated when it grows beyond ``max_bytes``
    so a long-lived add-on does not fill the disk with logs.
    """

    def __init__(self, path: str | os.PathLike[str] | None, *, max_bytes: int = _LOG_MAX_BYTES):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._fh: IO[str] | None = None
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.is_file() and self.path.stat().st_size > max_bytes:
                    self.path.unlink()
                self._fh = open(self.path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115
            except OSError as e:
                log.warning("Cannot open add-on log %s: %s", self.path, e)
                self._fh = None

    def write(self, line: str, prefix: str = "") -> None:
        """Append one line (newline added)."""
        if self._fh is None:
            return
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            try:
                self._fh.write(f"{stamp} {prefix}{line.rstrip()}\n")
                self._fh.flush()
            except (OSError, ValueError):
                pass

    def header(self, title: str) -> None:
        """Write a visually separated section header."""
        if self._fh is None:
            return
        with self._lock:
            try:
                self._fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {title} =====\n")
                self._fh.flush()
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None

    def __enter__(self) -> LogSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _ensure_sink(log_sink: LogSink | str | os.PathLike[str] | None) -> tuple[LogSink, bool]:
    if isinstance(log_sink, LogSink):
        return log_sink, False
    return LogSink(log_sink), True


# --------------------------------------------------------------------------- line reader
def _pump_lines(stream: IO[bytes], put: Callable[[str], None]) -> None:
    """Read ``stream`` until EOF, splitting on ``\\n`` and ``\\r`` (progress bars)."""
    buf = b""
    try:
        while True:
            chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)  # type: ignore[attr-defined]
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"\r\n|\r|\n", buf)
            buf = parts.pop()
            for p in parts:
                if p:
                    put(p.decode("utf-8", errors="replace"))
            if len(buf) > 1_000_000:  # pathological line without newline
                put(buf.decode("utf-8", errors="replace"))
                buf = b""
    except (OSError, ValueError):
        pass
    finally:
        if buf:
            put(buf.decode("utf-8", errors="replace"))
        try:
            stream.close()
        except OSError:
            pass


@dataclass
class ProcResult:
    """Outcome of :func:`run_process`."""

    returncode: int
    tail: list[str] = field(default_factory=list)  # last output lines (for error messages)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def tail_text(self, n: int = 15) -> str:
        return "\n".join(self.tail[-n:])


def _spawn(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    env: Mapping[str, str] | None,
    cwd: str | os.PathLike[str] | None,
    stdin: bool,
    split_stderr: bool,
    title: str,
) -> subprocess.Popen:
    args = [os.fspath(c) for c in cmd]
    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if split_stderr else subprocess.STDOUT,
            env=dict(env) if env is not None else clean_env(),
            cwd=os.fspath(cwd) if cwd else None,
            **popen_kwargs(),
        )
    except FileNotFoundError as e:
        raise AddonError(
            f"{title}: program not found ({args[0]}).",
            hint=f"The add-on installation may be incomplete; reinstall it from the {UI_ADDONS}.",
        ) from e
    except OSError as e:
        raise AddonError(f"{title}: cannot start {Path(args[0]).name}: {e}") from e
    # Windows: killed by the OS together with the app, even after a crash / Task Manager kill
    assign_to_kill_job(proc)
    return proc


def run_process(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    log_sink: LogSink | str | os.PathLike[str] | None = None,
    on_line: LineCallback | None = None,
    cancel: CancelToken | None = None,
    timeout: float | None = None,
    title: str = "process",
    stdin_data: str | None = None,
) -> ProcResult:
    """Run ``cmd`` to completion, streaming merged stdout+stderr lines.

    Every line goes to ``log_sink`` and ``on_line`` (called from this thread).  ``cancel``
    is polled every 50 ms; on cancel the whole process tree is killed and
    :class:`Cancelled` is raised.  On ``timeout`` (seconds) the tree is killed and an
    :class:`AddonError` raised.  A non-zero exit code is *returned* (not raised) so the
    caller can build a specific message.  ``env`` defaults to :func:`clean_env`.
    """
    sink, own = _ensure_sink(log_sink)
    sink.write("$ " + " ".join(_quote(os.fspath(c)) for c in cmd))
    t0 = time.monotonic()
    proc = _spawn(cmd, env=env, cwd=cwd, stdin=stdin_data is not None, split_stderr=False, title=title)
    q: queue.Queue[str | None] = queue.Queue()
    tail: collections.deque[str] = collections.deque(maxlen=_TAIL_LINES)

    def put(line: str) -> None:
        q.put(line)

    reader = threading.Thread(target=_reader_main, args=(proc.stdout, put, q), daemon=True)
    reader.start()
    if stdin_data is not None and proc.stdin is not None:
        _start_feeder(proc.stdin, stdin_data)
    try:
        eof = False
        exited_at: float | None = None
        while True:
            if cancel is not None and cancel.cancelled:
                sink.write(f"[cancelled - killing {title}]")
                kill_process_tree(proc)
                raise Cancelled(f"{title} cancelled")
            if timeout is not None and time.monotonic() - t0 > timeout:
                sink.write(f"[timeout after {timeout:.0f}s - killing {title}]")
                kill_process_tree(proc)
                raise AddonError(
                    f"{title} took longer than {_fmt_duration(timeout)} and was stopped.",
                    hint="Try smaller inputs or lower quality settings.",
                )
            try:
                item = q.get(timeout=_POLL_S)
            except queue.Empty:
                if proc.poll() is not None:
                    if eof:
                        break
                    # a grandchild may keep the pipe open after the process exited
                    exited_at = exited_at or time.monotonic()
                    if time.monotonic() - exited_at > _ORPHAN_PIPE_S:
                        sink.write(f"[{title} exited but a child process kept running - killing it]")
                        kill_orphans(proc)
                        break
                continue
            if item is None:
                eof = True
                continue
            tail.append(item)
            sink.write(item, "  ")
            if on_line is not None:
                try:
                    on_line(item)
                except (Cancelled, AddonError):
                    kill_process_tree(proc)
                    raise
                except Exception as e:  # a buggy parser must not kill the install
                    log.debug("on_line callback failed: %s", e)
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            kill_process_tree(proc)
        reader.join(timeout=2.0)
        sink.write(f"[exit code {proc.returncode} after {time.monotonic() - t0:.1f}s]")
        if own:
            sink.close()
    return ProcResult(rc, list(tail), time.monotonic() - t0)


def _reader_main(stream: IO[bytes] | None, put: Callable[[str], None], q: queue.Queue) -> None:
    if stream is not None:
        _pump_lines(stream, put)
    q.put(None)


def _start_feeder(stdin: IO[bytes], data: str) -> None:
    """Write ``data`` to the child's stdin from a thread (never blocks the caller)."""
    threading.Thread(target=_feed_stdin, args=(stdin, data), daemon=True).start()


def _feed_stdin(stdin: IO[bytes], data: str) -> None:
    try:
        stdin.write(data.encode("utf-8"))
        stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            stdin.close()
        except OSError:
            pass


def _quote(s: str) -> str:
    return f'"{s}"' if (" " in s or not s) else s


def _fmt_duration(sec: float) -> str:
    if sec >= 3600:
        return f"{sec / 3600:.1f} h"
    if sec >= 120:
        return f"{sec / 60:.0f} min"
    return f"{sec:.0f} s"


# --------------------------------------------------------------------------- error hints
_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation", re.I),
        "The GPU ran out of memory. Lower the quality settings (resolution, steps, texture size), "
        "close other GPU-heavy apps (games, video players, browsers) and try again.",
    ),
    (
        re.compile(
            r"CUDA is not available|no CUDA GPUs|Torch not compiled with CUDA|"
            r"CUDA driver version is insufficient|no kernel image is available|"
            r"Found no NVIDIA driver",
            re.I,
        ),
        "This add-on needs an NVIDIA GPU with an up-to-date driver (CUDA 12.8: driver 570 or newer).",
    ),
    (
        re.compile(r"GatedRepoError|gated repo|401 Client Error|Access to model .* is restricted|Invalid user token", re.I),
        "This model is gated on Hugging Face: accept its license on the model page and enter an "
        f"access token in {UI_SETTINGS_MODELS}.",
    ),
    (
        re.compile(r"No space left on device|not enough space on the disk|\[WinError 112\]|Errno 28", re.I),
        "The disk is full. Free some space (or move the data folder) and try again.",
    ),
    (
        re.compile(r"WinError 32\]|being used by another process|WinError 5\]|Access is denied", re.I),
        "A file is in use or locked. Close running jobs and other programs using the add-on folder "
        "(antivirus scans can also hold new files for a moment) and try again.",
    ),
    (
        re.compile(r"ModuleNotFoundError|ImportError|DLL load failed|No module named", re.I),
        f"The add-on installation looks incomplete or broken. Reinstall it from the {UI_ADDONS}.",
    ),
    (
        re.compile(r"ConnectionError|Max retries exceeded|Temporary failure in name resolution|getaddrinfo failed", re.I),
        "A download failed. Check your internet connection (and proxy/firewall) and try again.",
    ),
)


def error_hint(*texts: str | None) -> str | None:
    """Best human hint for an error message / traceback / stderr tail, or None."""
    blob = "\n".join(t for t in texts if t)
    if not blob:
        return None
    for pat, hint in _HINTS:
        if pat.search(blob):
            return hint
    return None


# --------------------------------------------------------------------------- workers
@dataclass
class WorkerRun:
    """Live state of one worker invocation (exposed for tests / diagnostics)."""

    hello: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    junk_lines: int = 0
    stderr_tail: collections.deque[str] = field(default_factory=lambda: collections.deque(maxlen=_TAIL_LINES))


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalars
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def run_worker(
    python: str | os.PathLike[str],
    worker: str | os.PathLike[str],
    task: str,
    inputs: Mapping[str, Any],
    params: Mapping[str, Any],
    out_dir: str | os.PathLike[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    log_sink: LogSink | str | os.PathLike[str] | None = None,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    timeout: float | None = None,
    title: str = "Add-on",
    args: Sequence[str] = (),
    send_request: bool = True,
    state: WorkerRun | None = None,
) -> dict[str, Any]:
    """Run ``<python> -u <worker> [args]`` and return the ``result.outputs`` dict.

    The request line is written to stdin which is then closed.  stdout is parsed as the
    protocol stream (junk lines are ignored), stderr is logged.  Raises
    :class:`AddonError` with the worker's message (plus a hint for CUDA OOM, missing
    modules, gated models ...), :class:`Cancelled` after killing the process tree when
    ``cancel`` fires, or :class:`AddonError` on ``timeout``.
    """
    sink, own = _ensure_sink(log_sink)
    st = state if state is not None else WorkerRun()
    progress = progress or ProgressReporter.null()
    out = Path(out_dir)
    cmd = [os.fspath(python), "-u", os.fspath(worker), *args]
    request = None
    if send_request:
        out.mkdir(parents=True, exist_ok=True)
        request = protocol.make_request(task, _to_jsonable(dict(inputs)), _to_jsonable(dict(params)), os.fspath(out))
    sink.header(f"{title}: {task or ' '.join(args)}")
    sink.write("$ " + " ".join(_quote(c) for c in cmd))
    t0 = time.monotonic()
    child_env = dict(env) if env is not None else clean_env()
    child_env[PARENT_PID_ENV] = str(os.getpid())
    proc = _spawn(cmd, env=child_env, cwd=cwd, stdin=True, split_stderr=True, title=title)

    events: queue.Queue[tuple[str, Any]] = queue.Queue()

    def on_stdout(line: str) -> None:
        msg = protocol.decode(line)
        if msg is None:
            st.junk_lines += 1
            log.debug("%s: ignoring non-protocol stdout line: %s", title, line[:200])
            sink.write(line, "[stdout] ")
            return
        events.put(("msg", msg))

    def on_stderr(line: str) -> None:
        st.stderr_tail.append(line)
        sink.write(line, "  ")

    t_out = threading.Thread(target=_reader_main, args=(proc.stdout, on_stdout, _Sentinel(events, "out")), daemon=True)
    t_err = threading.Thread(target=_reader_main, args=(proc.stderr, on_stderr, _Sentinel(events, "err")), daemon=True)
    t_out.start()
    t_err.start()
    if proc.stdin is not None:
        _start_feeder(proc.stdin, (request + "\n") if request is not None else "")

    open_streams = {"out", "err"}
    exited_at: float | None = None
    finished_at: float | None = None  # when the result / error message arrived
    try:
        while True:
            if cancel is not None and cancel.cancelled:
                sink.write(f"[cancelled - killing {title} worker]")
                kill_process_tree(proc)
                raise Cancelled(f"{title} cancelled")
            if finished_at is None and (st.result is not None or st.error is not None):
                finished_at = time.monotonic()
            if finished_at is None and timeout is not None and time.monotonic() - t0 > timeout:
                sink.write(f"[timeout after {timeout:.0f}s - killing {title} worker]")
                kill_process_tree(proc)
                raise AddonError(
                    f"{title} took longer than {_fmt_duration(timeout)} and was stopped.",
                    hint="Try a smaller input or lower quality settings.",
                )
            if finished_at is not None and proc.poll() is None and time.monotonic() - finished_at > _EXIT_GRACE_S:
                # e.g. a CUDA / driver teardown or a stray non-daemon thread keeps the
                # interpreter alive after the work is done - the result is already in hand
                sink.write(f"[{title} worker did not exit {_EXIT_GRACE_S:.0f}s after finishing - killing it]")
                kill_process_tree(proc)
                break
            try:
                kind, payload = events.get(timeout=_POLL_S)
            except queue.Empty:
                if proc.poll() is not None:
                    if not open_streams:
                        break
                    exited_at = exited_at or time.monotonic()
                    if time.monotonic() - exited_at > _ORPHAN_PIPE_S:
                        sink.write(f"[{title} worker exited but a child process kept running - killing it]")
                        kill_orphans(proc)
                        break
                continue
            if kind == "eof":
                open_streams.discard(payload)
                continue
            _handle_msg(payload, st, progress, sink, title)
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            kill_process_tree(proc)
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        sink.write(f"[worker exit code {proc.returncode} after {time.monotonic() - t0:.1f}s]")
        if own:
            sink.close()

    stderr_text = "\n".join(st.stderr_tail)
    if st.error is not None:
        message = str(st.error.get("message") or "The add-on failed.").strip()
        trace = str(st.error.get("trace") or "")
        hint = st.error.get("hint") or error_hint(message, trace, stderr_text)
        raise AddonError(f"{title}: {message}", hint=hint)
    if st.result is not None:
        outputs = st.result.get("outputs")
        return dict(outputs) if isinstance(outputs, Mapping) else {}
    last = next((ln for ln in reversed(st.stderr_tail) if ln.strip()), "")
    detail = f" Last message: {last.strip()[:300]}" if last else ""
    raise AddonError(
        f"{title} stopped unexpectedly (exit code {rc}) without a result.{detail}",
        hint=error_hint(stderr_text)
        or "See the add-on log for details; reinstalling the add-on often fixes broken environments.",
    )


class _Sentinel:
    """Adapter so :func:`_reader_main` can signal EOF of one stream into the event queue."""

    def __init__(self, events: queue.Queue, name: str):
        self._events = events
        self._name = name

    def put(self, item: object) -> None:
        if item is None:
            self._events.put(("eof", self._name))


def _handle_msg(msg: dict[str, Any], st: WorkerRun, progress: ProgressReporter, sink: LogSink, title: str) -> None:
    t = msg.get("type")
    if t == "hello":
        st.hello = msg
        sink.write(f"worker hello: {msg.get('worker', '?')} {msg.get('version', '')}".rstrip(), "[proto] ")
    elif t == "progress":
        try:
            frac = float(msg.get("frac", 0.0))
        except (TypeError, ValueError):
            frac = 0.0
        progress(frac, str(msg.get("msg", "")))
    elif t == "log":
        level = str(msg.get("level", "info")).lower()
        text = str(msg.get("msg", ""))
        sink.write(text, f"[{level}] ")
        log.log(_LEVELS.get(level, 20), "%s: %s", title, text)
    elif t == "result":
        if st.result is None and st.error is None:
            st.result = msg
    elif t == "error":
        if st.error is None and st.result is None:
            st.error = msg
            sink.write(str(msg.get("message", "")), "[error] ")
            if msg.get("trace"):
                for ln in str(msg["trace"]).splitlines():
                    sink.write(ln, "[trace] ")


_LEVELS = {"debug": 10, "info": 20, "warning": 30, "warn": 30, "error": 40, "critical": 50}


__all__ = [
    "PARENT_PID_ENV",
    "LogSink",
    "ProcResult",
    "WorkerRun",
    "error_hint",
    "run_process",
    "run_worker",
]
