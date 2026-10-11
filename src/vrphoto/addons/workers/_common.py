"""Protocol plumbing shared by all add-on workers.

This file runs INSIDE an add-on virtual environment (Python 3.11+, torch etc.), never in
the app process: standard library only, no ``vrphoto`` imports.  A worker script does::

    from _common import WorkerError, run

    def handle(task, inputs, params, out_dir, progress):
        progress(0.1, "Loading model")
        ...
        return {"glb": str(out_dir / "model.glb")}

    if __name__ == "__main__":
        run(handle, name="mything", version="1.0", check=check_fn)

Protocol (see ``vrphoto/addons/protocol.py``): one JSON request line on stdin, JSON
messages on the *protocol stream*.  The protocol stream is a duplicate of the original
stdout file descriptor; fd 1 itself is redirected to stderr so ``print()`` calls and C
libraries writing to stdout cannot corrupt the stream.

Command line flags::

    --check      import heavy modules, verify the install, print one result and exit
    --no-gpu     with --check: do not require a CUDA GPU (CI machines)
    --prefetch   download model weights into the add-on folder, then exit
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1

_lock = threading.Lock()
_stream: io.TextIOBase | None = None


class WorkerError(Exception):
    """An expected, user-facing failure (message shown as-is in the app).

    ``hint`` is an optional suggestion shown under the message.
    """

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


# --------------------------------------------------------------------------- streams
def setup_streams() -> io.TextIOBase:
    """Create the protocol stream and point fd 1 / ``sys.stdout`` at stderr (idempotent)."""
    global _stream
    if _stream is not None:
        return _stream
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        proto_fd = os.dup(1)
        os.dup2(2, 1)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.setmode(proto_fd, os.O_BINARY)
        raw = os.fdopen(proto_fd, "wb")
        _stream = io.TextIOWrapper(raw, encoding="utf-8", newline="\n", write_through=True)
    except OSError:
        # no real fd 1 (embedded / unusual hosts): fall back to the stdout object
        _stream = sys.stdout  # type: ignore[assignment]
    sys.stdout = sys.stderr
    return _stream  # type: ignore[return-value]


def emit(type_: str, **fields: Any) -> None:
    """Write one protocol message (thread-safe)."""
    stream = setup_streams()
    msg = {"type": type_, **fields}
    line = json.dumps(msg, ensure_ascii=True, separators=(",", ":"), default=str)
    with _lock:
        try:
            stream.write(line + "\n")
            stream.flush()
        except (OSError, ValueError):
            pass


def log(msg: str, level: str = "info") -> None:
    """Send a log line to the app's add-on log."""
    emit("log", level=level, msg=str(msg))


# --------------------------------------------------------------------------- progress
class Progress:
    """Callable progress reporter ``progress(frac, msg)`` with sub-ranges.

    Rate-limits identical consecutive fractions to keep the stream small.
    """

    def __init__(self, lo: float = 0.0, hi: float = 1.0, _shared: dict | None = None):
        self.lo = lo
        self.hi = hi
        self._shared = _shared if _shared is not None else {"last": (-1.0, ""), "t": 0.0}

    def __call__(self, frac: float, msg: str = "") -> None:
        try:
            f = float(frac)
        except (TypeError, ValueError):
            f = 0.0
        f = self.lo + (self.hi - self.lo) * min(1.0, max(0.0, f))
        last_f, last_m = self._shared["last"]
        now = time.monotonic()
        if msg == last_m and abs(f - last_f) < 0.002 and now - self._shared["t"] < 1.0:
            return
        self._shared["last"] = (f, msg)
        self._shared["t"] = now
        emit("progress", frac=round(f, 4), msg=msg)

    def sub(self, a: float, b: float) -> Progress:
        """Reporter mapping its [0,1] onto [a,b] of this one."""
        span = self.hi - self.lo
        return Progress(self.lo + span * a, self.lo + span * b, self._shared)


def track(iterable: Iterable[Any], progress: Progress, total: int | None = None, msg: str = "") -> Iterator[Any]:
    """Wrap an iterable and report progress per item (a drop-in for simple tqdm uses)."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    try:  # a tensor / numpy count must not be evaluated as a boolean below
        total = int(total) if total is not None else None
    except (TypeError, ValueError, OverflowError):  # OverflowError: total=float("inf")
        total = None
    for i, item in enumerate(iterable):
        if total:
            progress(i / total, msg)
        yield item
    progress(1.0, msg)


def tqdm_factory(progress: Progress, msg: str = "") -> Callable[..., Any]:
    """Return a minimal ``tqdm`` replacement reporting to ``progress``.

    Use it to monkeypatch ``module.tqdm`` in upstream code that loops with tqdm.
    Supports iteration, ``update()``, ``set_description()``, ``close()`` and use as a
    context manager - enough for typical sampler loops.
    """

    class _Tqdm:
        def __init__(self, iterable: Iterable[Any] | None = None, *args: Any, **kwargs: Any):
            # ``iterable`` may be a torch tensor or a numpy array (e.g. a sampler's timesteps):
            # never evaluate it as a boolean - bool() of a multi-element tensor/array raises
            self.iterable = iterable
            total = kwargs.get("total")
            if total is None and iterable is not None:
                try:
                    total = len(iterable)  # type: ignore[arg-type]
                except TypeError:  # generators, 0-d tensors / arrays
                    total = None
            try:
                self.total = int(total) if total is not None else None
            except (TypeError, ValueError, OverflowError):  # OverflowError: total=float("inf")
                self.total = None
            self.n = 0
            self.desc = kwargs.get("desc") or msg

        def __iter__(self) -> Iterator[Any]:
            if self.iterable is None:
                return
            for item in self.iterable:
                yield item
                self.update(1)

        def update(self, n: int = 1) -> None:
            self.n += n
            if self.total:
                progress(self.n / self.total, msg or str(self.desc or ""))

        def set_description(self, desc: str = "", refresh: bool = True) -> None:
            self.desc = desc

        def set_postfix(self, *a: Any, **k: Any) -> None:
            pass

        def refresh(self) -> None:
            pass

        def close(self) -> None:
            pass

        def __enter__(self) -> _Tqdm:
            return self

        def __exit__(self, *exc: object) -> None:
            self.close()

        def __len__(self) -> int:
            return int(self.total or 0)

    return _Tqdm


# --------------------------------------------------------------------------- helpers
def addon_dir() -> Path:
    """The add-on install folder (``VRPHOTO_ADDON_DIR``; defaults to the cwd)."""
    return Path(os.environ.get("VRPHOTO_ADDON_DIR") or os.getcwd())


def src_dir(name: str = "") -> Path:
    """``<addon_dir>/src[/name]`` - where github_archive steps extract source trees."""
    base = Path(os.environ.get("VRPHOTO_ADDON_SRC") or (addon_dir() / "src"))
    return base / name if name else base


def add_sys_path(path: str | os.PathLike[str]) -> None:
    """Prepend a source folder to ``sys.path`` (no-op when already present)."""
    p = os.fspath(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def param(params: dict[str, Any], name: str, default: Any, cast: Callable[[Any], Any] | None = None,
          lo: float | None = None, hi: float | None = None) -> Any:
    """Read ``params[name]`` with a default, optional cast and numeric clamping."""
    value = params.get(name, default)
    if value is None:
        value = default
    if cast is not None:
        try:
            value = cast(value)
        except (TypeError, ValueError) as e:
            raise WorkerError(f"Invalid value for '{name}': {params.get(name)!r}") from e
    if lo is not None and value < lo:
        value = type(value)(lo)
    if hi is not None and value > hi:
        value = type(value)(hi)
    return value


def require_input(inputs: dict[str, Any], name: str) -> Path:
    """Return ``inputs[name]`` as an existing path or raise :class:`WorkerError`."""
    v = inputs.get(name)
    if not v:
        raise WorkerError(f"Missing input '{name}'.")
    p = Path(v)
    if not p.exists():
        raise WorkerError(f"Input file not found: {p}")
    return p


_ARCH_RE = re.compile(r"(sm|compute)_(\d{2,})([a-z]?)")


def _parse_arch(entry: Any) -> tuple[str, int, int, str] | None:
    """``"sm_86"`` -> ``("sm", 8, 6, "")``; ``"compute_120"`` -> ``("compute", 12, 0, "")``;
    ``"sm_90a"`` -> ``("sm", 9, 0, "a")``.  None for anything else (e.g. ROCm ``gfx90a``)."""
    m = _ARCH_RE.fullmatch(str(entry).strip().lower())
    if m is None:
        return None
    digits = m.group(2)
    return m.group(1), int(digits[:-1]), int(digits[-1]), m.group(3)


def gpu_arch_supported(capability: Iterable[int], arch_list: Iterable[str]) -> bool | None:
    """Whether CUDA code built for ``arch_list`` (``torch.cuda.get_arch_list()``) runs on a
    GPU of compute ``capability`` ``(major, minor)``.  None when ``arch_list`` has no CUDA
    entries (unknown: do not block).

    * ``sm_XY`` (SASS machine code) runs on GPUs of the same major version with a minor
      version >= Y (``sm_61`` runs on 6.1 and 6.2, not on 6.0 or 7.0);
    * ``compute_XY`` (PTX) is compiled by the driver for any GPU >= X.Y;
    * ``sm_XYa`` / ``compute_XYa`` (architecture-specific features) run on exactly X.Y;
      ``sm_XYf`` / ``compute_XYf`` (family-specific) on the same major with minor >= Y.
    """
    cap = tuple(int(c) for c in capability)
    major, minor = cap[0], cap[1]
    known = False
    for entry in arch_list:
        parsed = _parse_arch(entry)
        if parsed is None:
            continue
        known = True
        kind, a_major, a_minor, suffix = parsed
        if suffix == "f" or (kind == "sm" and not suffix):
            ok = a_major == major and a_minor <= minor
        elif kind == "compute" and not suffix:
            ok = (a_major, a_minor) <= (major, minor)
        else:  # "a" (or an unknown suffix): this exact architecture only
            ok = (a_major, a_minor) == (major, minor)
        if ok:
            return True
    return False if known else None


def gpu_arch_problem(device: int = 0) -> tuple[str, str] | None:
    """``(message, hint)`` when the installed PyTorch has no CUDA code for the GPU, else None
    (also when there is no GPU or it cannot be determined)."""
    import torch

    try:
        if not torch.cuda.is_available():
            return None
        cap = tuple(int(c) for c in torch.cuda.get_device_capability(device))
        archs = [str(a) for a in torch.cuda.get_arch_list()]
    except Exception:
        return None
    if gpu_arch_supported(cap, archs) is not False:
        return None
    try:
        name = str(torch.cuda.get_device_name(device))
    except Exception:
        name = "NVIDIA GPU"
    sm = f"sm_{cap[0]}{cap[1]}"
    built = ", ".join(archs)
    message = (
        f"This GPU ({name}, {sm}) is not supported by the installed PyTorch build "
        f"(torch {getattr(torch, '__version__', '?')}, built for {built})."
    )
    versions = sorted((p[1], p[2]) for p in map(_parse_arch, archs) if p is not None)
    oldest, newest = versions[0], versions[-1]
    if (cap[0], cap[1]) < oldest:
        hint = (
            f"This graphics card is too old for this add-on: it needs an NVIDIA GPU of compute capability "
            f"{oldest[0]}.{oldest[1]} or newer. Use another method, or a PC with a newer NVIDIA GPU."
        )
    elif (cap[0], cap[1]) > newest:
        hint = ("This graphics card is newer than this add-on's PyTorch build supports. "
                "Update VR Photo Studio and reinstall the add-on.")
    else:  # between the built architectures (e.g. sm_80 + sm_90 builds on a 7.5 card)
        hint = ("This add-on's PyTorch build has no code for this graphics card's generation. "
                "Use another method, or update VR Photo Studio and reinstall the add-on.")
    return message, hint


def require_supported_gpu(device: int = 0) -> None:
    """Raise :class:`WorkerError` naming the GPU and its ``sm_XY`` when the installed PyTorch
    build has no CUDA code for it (otherwise the first kernel fails mid-job with "no kernel
    image is available for execution on the device")."""
    problem = gpu_arch_problem(device)
    if problem is not None:
        raise WorkerError(problem[0], hint=problem[1])


def cuda_info(require: bool = True) -> dict[str, Any]:
    """Describe torch/CUDA.

    With ``require``, raise :class:`WorkerError` when CUDA is missing or the installed PyTorch
    build does not support the GPU's architecture (:func:`require_supported_gpu`); without
    it, an unsupported GPU is reported as ``gpu_supported = False`` plus ``gpu_problem``.
    """
    import torch

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda": bool(torch.cuda.is_available()),
    }
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["vram_gb"] = round(props.total_memory / 1024**3, 1)
        info["capability"] = f"{props.major}.{props.minor}"
        problem = gpu_arch_problem(0)
        if problem is not None:
            if require:
                raise WorkerError(problem[0], hint=problem[1])
            info["gpu_supported"] = False
            info["gpu_problem"] = problem[0]
    elif require:
        raise WorkerError(
            "CUDA is not available: PyTorch cannot see an NVIDIA GPU.",
            hint="This add-on needs an NVIDIA GPU with a recent driver (CUDA 12.8: driver 570 or newer).",
        )
    return info


def free_cuda() -> None:
    """Release cached CUDA memory (best effort)."""
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- parent watchdog
PARENT_PID_ENV = "VRPHOTO_PARENT_PID"
_watchdog_started = False


def _parent_alive_posix(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0 = existence check (POSIX only!)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _windows_wait_handle(pid: int) -> Any:
    """SYNCHRONIZE handle of ``pid`` (None if unavailable).  Never use ``os.kill`` on
    Windows for this: it calls TerminateProcess for every signal except Ctrl events."""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    synchronize = 0x00100000
    handle = kernel32.OpenProcess(synchronize, 0, pid)
    return (kernel32, handle) if handle else None


def start_parent_watchdog(poll_s: float = 1.0) -> bool:
    """Exit this worker when the app process (``VRPHOTO_PARENT_PID``) disappears.

    Without it a worker keeps running - and holding gigabytes of VRAM - after the app
    crashed or was killed from the Task Manager.  Returns True if a watchdog runs.
    """
    global _watchdog_started
    if _watchdog_started:
        return True
    try:
        pid = int(os.environ.get(PARENT_PID_ENV, "0"))
    except ValueError:
        return False
    if pid <= 0 or pid == os.getpid():
        return False

    def _die() -> None:
        try:
            sys.stderr.write(f"[worker] the app process {pid} is gone - exiting\n")
            sys.stderr.flush()
        finally:
            os._exit(3)

    if sys.platform == "win32":
        try:
            waiter = _windows_wait_handle(pid)
        except Exception:
            waiter = None
        if waiter is None:
            return False
        kernel32, handle = waiter

        def _watch_win() -> None:
            infinite = 0xFFFFFFFF
            if kernel32.WaitForSingleObject(handle, infinite) == 0:  # WAIT_OBJECT_0: process ended
                _die()

        threading.Thread(target=_watch_win, name="parent-watchdog", daemon=True).start()
    else:
        if not _parent_alive_posix(pid):
            _die()

        def _watch_posix() -> None:
            while True:
                time.sleep(poll_s)
                if not _parent_alive_posix(pid):
                    _die()

        threading.Thread(target=_watch_posix, name="parent-watchdog", daemon=True).start()
    _watchdog_started = True
    return True


# --------------------------------------------------------------------------- main loop
Handler = Callable[[str, dict, dict, Path, Progress], dict]
CheckFn = Callable[[bool], dict]
PrefetchFn = Callable[[Progress], dict]


def _error_fields(e: BaseException) -> dict[str, Any]:
    if isinstance(e, WorkerError):
        fields: dict[str, Any] = {"message": str(e), "trace": traceback.format_exc()}
        if e.hint:
            fields["hint"] = e.hint
        return fields
    text = str(e).strip()
    message = f"{type(e).__name__}: {text}" if text else type(e).__name__
    if "out of memory" in text.lower():
        message = "CUDA out of memory: " + text.splitlines()[0][:300]
    return {"message": message, "trace": traceback.format_exc()}


def run(
    handler: Handler,
    *,
    name: str,
    version: str = "1",
    check: CheckFn | None = None,
    prefetch: PrefetchFn | None = None,
    tasks: Iterable[str] | None = None,
    argv: list[str] | None = None,
) -> int:
    """Worker entry point: parse flags, serve one request, exit.  Returns the exit code.

    ``handler(task, inputs, params, out_dir, progress) -> outputs`` does the work.
    ``check(require_gpu) -> dict`` verifies the install for ``--check``;
    ``prefetch(progress) -> dict`` downloads weights for ``--prefetch``.
    """
    setup_streams()
    start_parent_watchdog()
    args = list(sys.argv[1:] if argv is None else argv)
    known = set(tasks) if tasks is not None else None
    try:
        if "--check" in args:
            emit("hello", protocol=PROTOCOL_VERSION, worker=name, version=version, mode="check")
            info = check("--no-gpu" not in args) if check is not None else {}
            emit("result", outputs=info or {})
            return 0
        if "--prefetch" in args:
            emit("hello", protocol=PROTOCOL_VERSION, worker=name, version=version, mode="prefetch")
            info = prefetch(Progress()) if prefetch is not None else {}
            emit("result", outputs=info or {})
            return 0

        raw = sys.stdin.buffer.readline() if hasattr(sys.stdin, "buffer") else sys.stdin.readline().encode()
        if not raw.strip():
            raise WorkerError("No request received on stdin.")
        try:
            req = json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise WorkerError(f"Malformed request: {e}") from e
        if not isinstance(req, dict) or req.get("type") != "request":
            raise WorkerError("Malformed request: not a request message.")
        if int(req.get("protocol", 0)) != PROTOCOL_VERSION:
            raise WorkerError(
                f"Protocol mismatch (app {req.get('protocol')}, worker {PROTOCOL_VERSION}).",
                hint="Update or reinstall the add-on.",
            )
        emit("hello", protocol=PROTOCOL_VERSION, worker=name, version=version)
        task = str(req.get("task", ""))
        if known is not None and task not in known:
            raise WorkerError(f"Unknown task '{task}' for the {name} add-on (supported: {', '.join(sorted(known))}).")
        out_dir = Path(req.get("out_dir") or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        inputs = req.get("inputs") or {}
        params = req.get("params") or {}
        outputs = handler(task, inputs, params, out_dir, Progress())
        emit("result", outputs=outputs or {})
        return 0
    except KeyboardInterrupt:
        emit("error", message="Interrupted.")
        return 130
    except SystemExit as e:
        emit("error", message=f"The worker exited early (code {e.code}).", trace=traceback.format_exc())
        return 1
    except BaseException as e:
        emit("error", **_error_fields(e))
        return 1


__all__ = [
    "Progress",
    "WorkerError",
    "add_sys_path",
    "addon_dir",
    "cuda_info",
    "emit",
    "free_cuda",
    "gpu_arch_problem",
    "gpu_arch_supported",
    "log",
    "param",
    "require_input",
    "require_supported_gpu",
    "run",
    "setup_streams",
    "src_dir",
    "start_parent_watchdog",
    "tqdm_factory",
    "track",
]
