"""Clean environments and spawn flags for add-on subprocesses.

When VR Photo Studio runs as a PyInstaller-frozen exe, the bootloader and the Qt runtime
hooks put variables into ``os.environ`` (``_MEIPASS2``, ``_PYI_*``, ``QT_PLUGIN_PATH``,
``PATH`` entries inside the ``_internal`` folder, ...).  Child processes inherit them,
and a different Python interpreter (an add-on venv) or a third-party tool (COLMAP ships
its own Qt) then loads the wrong DLLs / plugins and crashes in confusing ways.
:func:`clean_env` removes that leakage; :func:`popen_kwargs` gives the platform flags
that keep add-on processes windowless and killable as a whole process tree.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core.log import get_logger

log = get_logger(__name__)

#: Variables that are always removed (they make another interpreter / Qt app misbehave).
LEAKY_VARS: tuple[str, ...] = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONEXECUTABLE",
    "PYTHONUSERBASE",
    "VIRTUAL_ENV",
    "VIRTUAL_ENV_PROMPT",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_PYTHON_EXE",
    "__PYVENV_LAUNCHER__",
    "QT_PLUGIN_PATH",
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "QML2_IMPORT_PATH",
    "QML_IMPORT_PATH",
    "QTWEBENGINEPROCESS_PATH",
    "OPENSSL_MODULES",
    "TCL_LIBRARY",
    "TK_LIBRARY",
    "_MEIPASS2",
)

#: Removed only when they point into the frozen bundle (``sys._MEIPASS``).
BUNDLE_PATH_VARS: tuple[str, ...] = (
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "GDK_PIXBUF_MODULE_FILE",
    "GIO_MODULE_DIR",
    "FONTCONFIG_FILE",
    "FONTCONFIG_PATH",
)

#: Path-list variables whose entries inside the bundle are stripped.
PATH_LIST_VARS: tuple[str, ...] = ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")

#: Always set for children.
FORCED_VARS: dict[str, str] = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYINSTALLER_RESET_ENVIRONMENT": "1",
}

_IS_WINDOWS = sys.platform == "win32"


def meipass_dir() -> Path | None:
    """The PyInstaller bundle directory (``sys._MEIPASS``) or None when not frozen."""
    mp = getattr(sys, "_MEIPASS", None)
    return Path(mp) if mp else None


def _norm(p: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(p))).rstrip("\\/")


def _inside(path: str, root: str | None) -> bool:
    """True if ``path`` is ``root`` or below it (case-insensitive on Windows)."""
    if not root or not path:
        return False
    try:
        p = _norm(path)
    except (OSError, ValueError):
        return False
    return p == root or p.startswith(root + os.sep)


def _key(env: Mapping[str, str], name: str) -> str | None:
    """Actual key for ``name`` in ``env`` (environment names are case-insensitive on Windows)."""
    if name in env:
        return name
    if _IS_WINDOWS:
        up = name.upper()
        for k in env:
            if k.upper() == up:
                return k
    return None


def _pop(env: dict[str, str], name: str) -> str | None:
    k = _key(env, name)
    return env.pop(k) if k is not None else None


def clean_env(extra: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return a copy of ``os.environ`` that is safe for add-on subprocesses.

    Removes PyInstaller / Qt / Python leakage (see :data:`LEAKY_VARS`), every ``_PYI_*``
    variable, CA-bundle style variables pointing into the frozen bundle and ``PATH`` /
    ``LD_LIBRARY_PATH`` entries inside ``sys._MEIPASS`` (restoring ``*_ORIG`` values saved
    by the PyInstaller bootloader on Linux/macOS).  Then forces ``PYTHONUNBUFFERED=1``,
    ``PYTHONIOENCODING=utf-8`` and ``PYINSTALLER_RESET_ENVIRONMENT=1``.

    ``extra`` is applied last; a value of ``None`` removes that variable, other values are
    converted with ``str()``.
    """
    env: dict[str, str] = dict(os.environ)
    mp = meipass_dir()
    root = _norm(mp) if mp is not None else None

    for name in LEAKY_VARS:
        _pop(env, name)
    for k in [k for k in env if k.upper().startswith("_PYI_")]:
        env.pop(k, None)
    for name in BUNDLE_PATH_VARS:
        k = _key(env, name)
        if k is not None and _inside(env[k], root):
            env.pop(k)

    for name in PATH_LIST_VARS:
        k = _key(env, name)
        orig = _pop(env, name + "_ORIG")  # saved by the PyInstaller bootloader (POSIX)
        if k is None and orig is None:
            continue
        value = orig if orig is not None and name != "PATH" else env.get(k or name, "")
        parts = [p for p in value.split(os.pathsep) if p and not _inside(p, root)]
        if k is not None:
            env.pop(k)
        if parts:
            env[k or name] = os.pathsep.join(dict.fromkeys(parts))

    for name, value in FORCED_VARS.items():
        k = _key(env, name)
        if k is not None and k != name:
            env.pop(k)
        env[name] = value

    if extra:
        for name, value in extra.items():
            k = _key(env, name)
            if k is not None:
                env.pop(k)
            if value is not None:
                env[str(name)] = str(value)
    return env


def prepend_path(env: dict[str, str], *dirs: str | os.PathLike[str], var: str = "PATH") -> dict[str, str]:
    """Prepend directories to a path-list variable of ``env`` (in place; returns ``env``)."""
    k = _key(env, var) or var
    parts = [os.fspath(d) for d in dirs if d] + [p for p in env.get(k, "").split(os.pathsep) if p]
    env[k] = os.pathsep.join(dict.fromkeys(parts))
    return env


def _reset_dll_directory() -> None:
    """Undo ``SetDllDirectoryW(_MEIPASS)`` done by the PyInstaller bootloader (Windows).

    The DLL search directory is inherited by child processes; leaving it pointing at the
    bundle makes e.g. a CUDA-enabled torch load our bundled (older) MSVC/Qt/OpenSSL DLLs.
    """
    if not _IS_WINDOWS:
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetDllDirectoryW(None)  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover - best effort
        log.debug("SetDllDirectoryW(None) failed: %s", e)


def popen_kwargs() -> dict[str, Any]:
    """Platform keyword arguments for :class:`subprocess.Popen` of add-on processes.

    Windows: no console window and a new process group (``CREATE_NO_WINDOW |
    CREATE_NEW_PROCESS_GROUP``); also resets the inherited DLL directory.  POSIX: a new
    session so the whole process tree can be killed with :func:`os.killpg`.
    """
    if _IS_WINDOWS:
        _reset_dll_directory()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
        )
        return {"creationflags": flags}
    return {"start_new_session": True}


def kill_orphans(proc: subprocess.Popen) -> None:
    """POSIX: kill what is left of the process group of an exited ``proc`` (grandchildren
    that kept running).  The group id cannot be reused while members exist."""
    if _IS_WINDOWS:
        return
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGKILL)


def kill_process_tree(proc: subprocess.Popen, *, wait: float = 5.0) -> None:
    """Forcefully terminate ``proc`` and all of its children.

    Windows: ``taskkill /T /F /PID``; POSIX: ``SIGKILL`` to the process group (the process
    was started with ``start_new_session=True``).  Falls back to ``proc.kill()``.
    """
    if proc.poll() is not None:
        return
    pid = proc.pid
    try:
        if _IS_WINDOWS:
            subprocess.run(
                [_taskkill(), "/T", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
                timeout=max(wait, 1.0),
                check=False,
            )
        else:
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                pgid = None
            if pgid is not None and pgid == pid:  # only kill a group we own
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("process tree kill of %s failed: %s", pid, e)
    with contextlib.suppress(OSError):
        if proc.poll() is None:
            proc.kill()
    try:
        proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:  # pragma: no cover - extremely unlikely after SIGKILL
        log.warning("Process %s did not exit after kill", pid)


def _taskkill() -> str:
    """Absolute path of ``taskkill.exe`` (PATH may be odd inside a frozen app)."""
    root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
    cand = Path(root) / "System32" / "taskkill.exe"
    return str(cand) if cand.is_file() else "taskkill"


def native_tool_env(exe: str | os.PathLike[str], extra: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Clean environment for a bundled native tool such as COLMAP or Brush.

    The tool's own folder (and a sibling ``lib``) go first on ``PATH`` so its DLLs win
    over anything else installed; ``QT_PLUGIN_PATH`` points at its Qt plugins - COLMAP
    4.x Windows builds ship them in ``<root>/plugins`` next to ``bin`` (what its
    ``COLMAP.bat`` sets), older builds in ``lib``.  ``QT_QPA_PLATFORM=offscreen`` is only
    forced on a Linux machine without a display; on Windows the native platform plugin
    works without a visible window and is needed for COLMAP's OpenGL (non-CUDA) SIFT.
    A ``.py`` path (fake tools in the tests) gets no PATH changes.
    """
    env_extra: dict[str, Any] = {}
    p = Path(exe)
    if p.suffix.lower() != ".py":
        bin_dir = p.resolve().parent
        root = bin_dir.parent
        dirs = [str(bin_dir)]
        if (root / "lib").is_dir():
            dirs.append(str(root / "lib"))
        env_extra["PATH"] = os.pathsep.join([*dirs, os.environ.get("PATH", "")])
        for plugins in (root / "plugins", bin_dir / "plugins", root / "lib"):
            if (plugins / "platforms").is_dir() or (plugins.name == "lib" and plugins.is_dir()):
                env_extra["QT_PLUGIN_PATH"] = str(plugins)
                break
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        env_extra["QT_QPA_PLATFORM"] = "offscreen"
    if extra:
        env_extra.update(extra)
    return clean_env(env_extra)


def venv_python(venv: str | os.PathLike[str]) -> Path:
    """Interpreter path inside a virtual environment."""
    v = Path(venv)
    return v / "Scripts" / "python.exe" if _IS_WINDOWS else v / "bin" / "python"


__all__ = [
    "LEAKY_VARS",
    "clean_env",
    "kill_orphans",
    "kill_process_tree",
    "meipass_dir",
    "native_tool_env",
    "popen_kwargs",
    "prepend_path",
    "venv_python",
]
