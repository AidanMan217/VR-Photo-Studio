"""Add-on manager: list / install / uninstall / run optional heavy add-ons.

Add-ons are described by TOML manifests (:mod:`vrphoto.addons.manifest`).  ``venv``
add-ons get their own uv-managed virtual environment in
``<addons_dir>/<id>/venv`` (PyTorch/CUDA etc. never enter the app process) and run as
subprocess workers speaking the JSON-lines protocol of :mod:`vrphoto.addons.protocol`.
``binary`` add-ons download native tools (COLMAP, Brush) and are driven by Python code
in the app (:mod:`vrphoto.addons.capture`).

Folder layout per add-on::

    <addons_dir>/<id>/state.json     {installed, manifest_sha256, version, date, steps_done, ...}
    <addons_dir>/<id>/install.log    everything the installer ran / printed
    <addons_dir>/<id>/run.log        worker logs
    <addons_dir>/<id>/venv/          Python environment (kind = "venv")
    <addons_dir>/<id>/src/           source archives (github_archive steps)
    <addons_dir>/<id>/hf/            Hugging Face cache (HF_HOME of the worker)

Environment overrides (tests / offline use): ``VRPHOTO_ADDON_PYTHON`` (interpreter used
for the venv instead of a uv-managed download), ``VRPHOTO_ADDON_MANIFESTS`` (extra
manifest folders, ``os.pathsep``-separated), ``VRPHOTO_UV`` (uv executable).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import glob
import importlib
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.config import get_settings, is_frozen
from ..core.errors import AddonError, AddonNotInstalled, Cancelled
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter
from . import fetch
from .env import clean_env, prepend_path, venv_python
from .installer import InstallContext, resume_point, run_steps, worker_cwd, worker_env_extra
from .manifest import AddonManifest, load_manifest, manifests_dir
from .runner import LogSink, run_worker

if TYPE_CHECKING:  # binary add-on entry modules (imported by name at run time; also a PyInstaller hint)
    from . import capture  # noqa: F401

log = get_logger(__name__)

MANIFESTS_ENV = "VRPHOTO_ADDON_MANIFESTS"
UV_ENV = "VRPHOTO_UV"
STATE_FILE = "state.json"
INSTALL_LOG = "install.log"
RUN_LOG = "run.log"

__all__ = [
    "AddonManifest",
    "addon_dir",
    "addons_for_task",
    "check",
    "find_uv",
    "get_manifest",
    "install",
    "install_log_path",
    "is_installed",
    "list_addons",
    "reload_manifests",
    "resolve_binary",
    "run",
    "status",
    "tool_command",
    "uninstall",
    "venv_python_of",
]

# --------------------------------------------------------------------------- manifests
_manifest_lock = threading.Lock()
_manifests: dict[str, AddonManifest] | None = None
_manifest_key: tuple[str, ...] | None = None


def _manifest_dirs() -> list[Path]:
    dirs = [manifests_dir()]
    for part in os.environ.get(MANIFESTS_ENV, "").split(os.pathsep):
        if part.strip():
            dirs.append(Path(part.strip()))
    return dirs


def _load_all() -> dict[str, AddonManifest]:
    global _manifests, _manifest_key
    dirs = _manifest_dirs()
    key = tuple(str(d) for d in dirs)
    with _manifest_lock:
        if _manifests is not None and _manifest_key == key:
            return _manifests
        found: dict[str, AddonManifest] = {}
        for d in dirs:
            if not d.is_dir():
                if d != dirs[0]:
                    log.warning("Add-on manifest folder not found: %s", d)
                continue
            for path in sorted(d.glob("*.toml")):
                try:
                    m = load_manifest(path)
                except AddonError as e:
                    log.error("%s", e)
                    continue
                if m.id in found:
                    log.warning("Add-on %s defined twice; using %s", m.id, path)
                found[m.id] = m
        _manifests, _manifest_key = found, key
        return found


def reload_manifests() -> None:
    """Forget cached manifests (they are re-read on next access)."""
    global _manifests
    with _manifest_lock:
        _manifests = None


def list_addons(include_experimental: bool = False) -> list[AddonManifest]:
    """All known add-ons (stable first, then by title); experimental only when asked."""
    items = [m for m in _load_all().values() if include_experimental or not m.experimental]
    return sorted(items, key=lambda m: (m.experimental, m.title.lower()))


def get_manifest(addon_id: str) -> AddonManifest:
    """Manifest for ``addon_id`` (raises :class:`AddonError` for unknown ids)."""
    m = _load_all().get(addon_id)
    if m is None:
        known = ", ".join(sorted(_load_all())) or "none"
        raise AddonError(f"Unknown add-on '{addon_id}'.", hint=f"Known add-ons: {known}.")
    return m


def addons_for_task(task: str, *, include_experimental: bool = True) -> list[AddonManifest]:
    """Add-ons that implement ``task`` (e.g. ``"outpaint_360"``)."""
    return [m for m in list_addons(include_experimental) if task in m.tasks]


# --------------------------------------------------------------------------- paths & state
def addon_dir(addon_id: str) -> Path:
    """Install folder of an add-on (not created)."""
    return get_settings().addons_dir / addon_id


def install_log_path(addon_id: str) -> Path:
    return addon_dir(addon_id) / INSTALL_LOG


def venv_python_of(addon_id: str) -> Path:
    """Interpreter of the add-on's virtual environment."""
    return venv_python(addon_dir(addon_id) / "venv")


def _read_state(d: Path) -> dict[str, Any]:
    p = d / STATE_FILE
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(d: Path, state: dict[str, Any]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, d / STATE_FILE)
            return
        except PermissionError:  # Windows: a concurrent status() read has the file open
            if attempt == 19:
                raise
            time.sleep(0.05)


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_installed(m: AddonManifest, d: Path, state: dict[str, Any]) -> bool:
    if not state.get("installed"):
        return False
    if m.kind == "venv" and not venv_python(d / "venv").is_file():
        return False
    return d.is_dir()


def is_installed(addon_id: str) -> bool:
    """True when the add-on finished installing (and its environment still exists)."""
    m = get_manifest(addon_id)
    d = addon_dir(addon_id)
    return _is_installed(m, d, _read_state(d))


def status(addon_id: str, *, refresh_size: bool = False) -> dict[str, Any]:
    """Install status::

        {"installed": bool, "version": str | None, "path": str, "size_mb": float,
         "update_available": bool, "partial": bool, "supported": bool, "date": str | None,
         "error": str | None, "busy": bool}

    ``size_mb`` is the size recorded at install time (cheap); pass ``refresh_size=True``
    (or for partial installs) to walk the folder.
    """
    m = get_manifest(addon_id)
    d = addon_dir(addon_id)
    state = _read_state(d)
    installed = _is_installed(m, d, state)
    if d.is_dir() and (refresh_size or not installed or "size_bytes" not in state):
        size = fetch.dir_size(d)
    else:
        size = int(state.get("size_bytes", 0)) if d.is_dir() else 0
    return {
        "installed": installed,
        "version": state.get("version") if installed else None,
        "path": str(d),
        "size_mb": round(size / 1e6, 1),
        "update_available": bool(installed and state.get("manifest_sha256") != m.sha256),
        "partial": bool(not installed and state.get("steps_done")),
        "supported": m.supported,
        "date": state.get("date"),
        "error": state.get("error"),
        "busy": _busy(addon_id),
        "warnings": list(state.get("warnings", [])),
    }


# --------------------------------------------------------------------------- locking
_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}
_active_runs: Counter[str] = Counter()


def _lock_for(addon_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(addon_id, threading.Lock())


def _busy(addon_id: str) -> bool:
    with _locks_guard:
        running = _active_runs[addon_id] > 0
    return running or _lock_for(addon_id).locked()


class _InterprocessLock:
    """Exclusive lock file held while installing/uninstalling (released if the process dies)."""

    def __init__(self, path: Path):
        self.path = path
        self._fh: Any = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")  # noqa: SIM115
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            self._fh = None
            return False
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                with contextlib.suppress(OSError):
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


@contextlib.contextmanager
def _exclusive(addon_id: str, action: str) -> Iterator[None]:
    lock = _lock_for(addon_id)
    if not lock.acquire(blocking=False):
        raise AddonError(f"The add-on '{addon_id}' is busy (another install/uninstall is running).")
    plock = _InterprocessLock(get_settings().addons_dir / f".{addon_id}.lock")
    try:
        if not plock.acquire():
            raise AddonError(
                f"The add-on '{addon_id}' is being changed by another VR Photo Studio window.",
                hint="Wait for it to finish, then try again.",
            )
        try:
            with _locks_guard:
                running = _active_runs[addon_id]
            if running:
                raise AddonError(
                    f"Cannot {action} '{addon_id}' while it is running.",
                    hint="Wait for the running job to finish or cancel it.",
                )
            yield
        finally:
            plock.release()
    finally:
        lock.release()


# --------------------------------------------------------------------------- uv
def find_uv() -> str:
    """Path of the ``uv`` executable used to build add-on environments.

    Order: ``VRPHOTO_UV``; the binary bundled in the frozen app
    (``_MEIPASS/bin/uv(.exe)``); ``uv.find_uv_bin()`` from the ``uv`` pip package;
    ``uv`` on PATH.  Raises :class:`AddonError` with a hint otherwise.
    """
    exe_name = "uv.exe" if sys.platform == "win32" else "uv"
    override = os.environ.get(UV_ENV, "").strip()
    if override:
        if Path(override).is_file():
            return override
        log.warning("%s=%s does not exist; ignoring", UV_ENV, override)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cand = Path(meipass) / "bin" / exe_name
        if cand.is_file():
            return str(cand)
    try:
        import uv  # type: ignore[import-not-found]

        path = uv.find_uv_bin()
        if path and Path(path).is_file():
            return str(path)
    except (ImportError, FileNotFoundError, AttributeError, OSError):
        pass
    which = shutil.which("uv")
    if which:
        return which
    raise AddonError(
        "The 'uv' package manager needed to install add-ons was not found.",
        hint="Reinstall VR Photo Studio (it ships uv), or install uv from https://docs.astral.sh/uv/.",
    )


# --------------------------------------------------------------------------- install
def _resolve_token(hf_token: str | None) -> str | None:
    if hf_token:
        return hf_token.strip() or None
    try:
        tok = (get_settings().hf_token or "").strip()
    except Exception:  # pragma: no cover - settings must never block installs
        tok = ""
    return tok or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _check_disk(m: AddonManifest, d: Path) -> None:
    need = m.disk_gb * 1e9
    if need <= 0:
        return
    have_now = fetch.dir_size(d) if d.is_dir() else 0
    remaining = max(0.0, need - have_now)
    root = get_settings().addons_dir
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        return
    if free < remaining + 1e9:
        raise AddonError(
            f"Not enough disk space for {m.title}: needs about {remaining / 1e9:.0f} GB, "
            f"{free / 1e9:.0f} GB free on the data drive.",
            hint="Free some disk space, or move the data folder (VRPHOTO_HOME) to a larger drive.",
        )


def install(
    addon_id: str,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    hf_token: str | None = None,
    check_gpu: bool = True,
    force: bool = False,
) -> None:
    """Install (or resume / update) an add-on.

    Runs the manifest steps in order with progress and cooperative cancel; everything
    is logged to ``install.log``.  Completed steps are skipped on a re-run; a changed
    step definition re-runs from that step on.  ``check_gpu=False`` makes the final
    verification import-only (machines without an NVIDIA GPU, CI).  ``force`` deletes
    the add-on folder first (clean reinstall).
    """
    m = get_manifest(addon_id)
    progress = progress or ProgressReporter.null()
    if not m.supported:
        raise AddonError(
            f"{m.title} is not available on this operating system.",
            hint=f"Supported: {', '.join(m.platforms)}.",
        )
    token = _resolve_token(hf_token)
    if m.hf_gated and not token:
        raise AddonError(
            f"{m.title} downloads a gated Hugging Face model and needs an access token.",
            hint="Create a free account at huggingface.co, accept the model license, create a read "
            "token and enter it in Settings.",
        )
    with _exclusive(addon_id, "install"):
        d = addon_dir(addon_id)
        if force and d.exists():
            progress(0.0, "Removing the previous installation")
            fetch.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        _check_disk(m, d)
        uv = find_uv() if m.kind == "venv" else ""
        state = _read_state(d)
        steps_done: list[str] = list(state.get("steps_done", []))
        if m.kind == "venv" and not venv_python(d / "venv").is_file():
            steps_done = []  # environment gone: everything after it must re-run
        # keep only the unbroken prefix of completed steps: records of later steps are
        # stale once an earlier step re-runs (and must not make a later install skip them)
        _start, steps_done = resume_point(m, steps_done)
        state.update(
            {
                "id": m.id,
                "installed": False,
                "version": m.version,
                "manifest_sha256": m.sha256,
                "steps_done": steps_done,
                "error": None,
                "started": _now(),
            }
        )
        _write_state(d, state)
        sink = LogSink(d / INSTALL_LOG)
        sink.header(f"install {m.id} {m.version} (manifest {m.sha256[:12]})")
        ctx = InstallContext(
            manifest=m,
            addon_dir=d,
            uv=uv,
            sink=sink,
            cancel=cancel,
            hf_token=token,
            check_gpu=check_gpu,
        )

        def on_step_done(key: str) -> None:
            if key not in state["steps_done"]:
                state["steps_done"].append(key)
            _write_state(d, state)

        try:
            warnings = run_steps(ctx, steps_done, progress=progress, on_step_done=on_step_done)
        except Cancelled:
            state["error"] = "Cancelled"
            _write_state(d, state)
            sink.write("Install cancelled by the user")
            raise
        except AddonError as e:
            state["error"] = str(e)
            _write_state(d, state)
            raise
        except Exception as e:  # unexpected bug: keep the message human-readable
            state["error"] = str(e)
            _write_state(d, state)
            sink.write(f"Unexpected error: {e!r}")
            log.exception("Add-on install crashed")
            raise AddonError(f"Installing {m.title} failed unexpectedly: {e}",
                             hint=f"See {d / INSTALL_LOG}") from e
        finally:
            sink.close()
        state.update(
            {
                "installed": True,
                "date": _now(),
                "error": None,
                "warnings": warnings,
                "size_bytes": fetch.dir_size(d),
            }
        )
        _write_state(d, state)
        progress(1.0, f"{m.title} installed")
        log.info("Add-on %s installed (%s)", m.id, d)


def uninstall(addon_id: str) -> None:
    """Delete the add-on folder (environment, downloads, models, logs)."""
    get_manifest(addon_id)
    with _exclusive(addon_id, "uninstall"):
        d = addon_dir(addon_id)
        if d.exists():
            try:
                fetch.rmtree(d)
            except OSError as e:
                raise AddonError(
                    f"Could not remove {d}: {e}",
                    hint="A file is in use. Close running jobs (and apps using the add-on folder) and retry.",
                ) from e
            log.info("Add-on %s uninstalled", addon_id)
    _prune_shared_dirs()


#: uv's package cache and managed interpreters, shared by all venv add-ons (see InstallContext.uv_env)
SHARED_DIRS = (".uv-cache", ".uv-python")


def _prune_shared_dirs() -> None:
    """Delete uv's shared download cache and managed Pythons once no add-on is left.

    With ``UV_LINK_MODE=copy`` (Windows) every installed wheel exists twice - in the
    cache and in the venv - so after the last uninstall several GB (PyTorch!) would
    otherwise stay behind in the data folder.
    """
    root = get_settings().addons_dir
    try:
        if any(p.is_dir() and not p.name.startswith(".") for p in root.iterdir()):
            return
    except OSError:
        return
    for name in SHARED_DIRS:
        p = root / name
        if p.exists():
            try:
                fetch.rmtree(p)
                log.info("Removed the shared add-on folder %s", p)
            except OSError as e:
                log.warning("Could not remove %s: %s", p, e)


# --------------------------------------------------------------------------- run
def _worker_env(m: AddonManifest, d: Path) -> dict[str, str]:
    env = clean_env(worker_env_extra(m, d, _resolve_token(None)))
    return prepend_path(env, venv_python(d / "venv").parent)


@contextlib.contextmanager
def _running(addon_id: str) -> Iterator[None]:
    lock = _lock_for(addon_id)
    with _locks_guard:
        if lock.locked():
            raise AddonError(
                f"The add-on '{addon_id}' is being installed or removed right now.",
                hint="Wait for that to finish, then try again.",
            )
        _active_runs[addon_id] += 1
    try:
        yield
    finally:
        with _locks_guard:
            _active_runs[addon_id] -= 1


def run(
    addon_id: str,
    task: str,
    inputs: dict,
    params: dict,
    out_dir: Path,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    timeout: float | None = None,
) -> dict:
    """Run ``task`` of an installed add-on and return the worker's ``result.outputs``.

    ``venv`` add-ons run ``<venv python> -u <worker>`` in a clean environment; ``binary``
    add-ons call their ``entry`` function in-process.  Raises
    :class:`AddonNotInstalled`, :class:`AddonError` (worker error, with a hint for CUDA
    out-of-memory etc.) or :class:`Cancelled`.
    """
    m = get_manifest(addon_id)
    if task not in m.tasks:
        raise AddonError(f"{m.title} cannot do '{task}' (supports: {', '.join(m.tasks)}).")
    d = addon_dir(addon_id)
    if not _is_installed(m, d, _read_state(d)):
        raise AddonNotInstalled(
            f"The add-on '{m.title}' is not installed.",
            hint="Install it in Settings > Add-ons (it is downloaded once).",
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress = progress or ProgressReporter.null()
    with _running(addon_id):
        if m.kind == "binary":
            if not m.entry:
                raise AddonError(f"{m.title} has no runnable entry point.")
            mod_name, _, fn_name = m.entry.partition(":")
            fn = getattr(importlib.import_module(mod_name), fn_name)
            return dict(fn(task, dict(inputs), dict(params), out, progress=progress, cancel=cancel, timeout=timeout))
        worker = m.worker_path()
        if worker is None or not worker.is_file():
            raise AddonError(
                f"The worker script of {m.title} is missing ({m.worker}).",
                hint="Reinstall VR Photo Studio.",
            )
        return run_worker(
            venv_python(d / "venv"),
            worker,
            task,
            inputs,
            params,
            out,
            env=_worker_env(m, d),
            cwd=worker_cwd(m, d),
            log_sink=d / RUN_LOG,
            progress=progress,
            cancel=cancel,
            timeout=timeout,
            title=m.title,
        )


def check(addon_id: str, *, check_gpu: bool = True, timeout: float = 1800, cancel: CancelToken | None = None) -> dict:
    """Verify an installed add-on (``worker --check``; tools launch for binary add-ons).

    Returns the check's info dict (e.g. torch/CUDA versions).  Raises :class:`AddonError`.
    """
    m = get_manifest(addon_id)
    d = addon_dir(addon_id)
    if not d.is_dir():
        raise AddonNotInstalled(f"The add-on '{m.title}' is not installed.")
    if m.kind == "binary":
        from .installer import step_check

        with LogSink(d / INSTALL_LOG) as sink:
            ctx = InstallContext(m, d, "", sink, cancel, None, check_gpu)
            step_check(ctx, {"type": "check"}, ProgressReporter.null())
        return {name: resolve_binary(m, name) for name in m.binaries}
    if not venv_python(d / "venv").is_file():
        raise AddonNotInstalled(f"The add-on '{m.title}' is not installed (no Python environment).")
    worker = m.worker_path()
    if worker is None:
        raise AddonError(f"{m.title} has no worker.")
    with _running(addon_id):
        return run_worker(
            venv_python(d / "venv"), worker, "", {}, {}, d,
            env=_worker_env(m, d), cwd=worker_cwd(m, d), log_sink=d / INSTALL_LOG,
            cancel=cancel, timeout=timeout, title=f"{m.title} check",
            args=["--check", *([] if check_gpu else ["--no-gpu"])], send_request=False,
        )


# --------------------------------------------------------------------------- binaries
def tool_command(path: str) -> list[str]:
    """Command prefix to launch a resolved tool.  A ``.py`` path (fake tools in tests or a
    developer override) runs with the current interpreter - not possible when frozen."""
    if path.lower().endswith(".py"):
        if is_frozen():
            raise AddonError(f"Cannot run a Python script as a tool in the packaged app: {path}")
        return [sys.executable, path]
    return [path]


def resolve_binary(m: AddonManifest, name: str, *, base_dir: Path | None = None) -> str:
    """Locate a tool of a ``binary`` add-on: env override, add-on folder, then PATH.

    Raises :class:`AddonNotInstalled` when the tool cannot be found.
    """
    spec = m.binaries.get(name)
    if spec is None:
        raise AddonError(f"{m.title} does not provide '{name}'.")
    env_var = spec.get("env_var")
    if env_var and os.environ.get(env_var):
        p = os.environ[env_var]
        if Path(p).exists():
            return p
        log.warning("%s=%s does not exist; ignoring", env_var, p)
    base = base_dir if base_dir is not None else addon_dir(m.id)
    for pat in spec.get("patterns", []):
        # escape the folder: "[" / "]" in a user / data path are glob syntax
        hits = sorted(glob.glob(os.path.join(glob.escape(str(base)), pat), recursive=True), key=len)
        hits = [h for h in hits if Path(h).is_file()]
        if hits:
            return hits[0]
    for exe in spec.get("path_names", []):
        found = shutil.which(exe)
        if found:
            return found
    raise AddonNotInstalled(
        f"{name} was not found.",
        hint=f"Install the '{m.title}' add-on in Settings > Add-ons.",
    )
