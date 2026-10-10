"""Getting content onto a Meta Quest from the PC: adb (USB debugging) and MTP (Windows).

* :func:`find_adb` / :func:`install_platform_tools` - locate ``adb`` or download Google's
  SDK Platform-Tools into ``<data_root>/tools/platform-tools``.
* :func:`adb_devices`, :func:`adb_reverse` (``http://localhost:PORT`` on the Quest reaches
  the PC - a secure context, so WebXR works without certificates), :func:`adb_push`
  (copy videos/photos into ``/sdcard/Movies`` for the Quest's own media player) and
  :func:`open_url_on_quest` (opens Quest Browser on a URL).
* :func:`mtp_copy` - no developer mode needed: copies files through Windows Explorer's
  MTP support (``Shell.Application``) using a generated PowerShell script.

All functions degrade gracefully: when adb or a headset is missing they return
``False`` / ``[]`` / ``None`` and log the reason; they never raise for such conditions.
:func:`last_error` returns that reason (with what to do about it) for the GUI.
Every subprocess uses a list of arguments, a timeout and ``CREATE_NO_WINDOW`` on Windows.
"""

from __future__ import annotations

import http.client
import logging
import os
import posixpath
import queue
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .. import __version__
from ..core.config import data_root
from ..core.errors import Cancelled, DownloadError
from ..core.log import get_logger
from . import net as _net

if TYPE_CHECKING:
    from ..core.types import CancelToken, ProgressReporter

log = get_logger(__name__)

__all__ = [
    "PLATFORM_TOOLS_INFO_URL",
    "find_adb",
    "install_platform_tools",
    "platform_tools_url",
    "adb_devices",
    "adb_device_states",
    "adb_reverse",
    "adb_reverse_remove",
    "adb_push",
    "open_url_on_quest",
    "mtp_copy",
    "last_error",
]

PLATFORM_TOOLS_INFO_URL = "https://developer.android.com/tools/releases/platform-tools"
_PT_BASE = "https://dl.google.com/android/repository/platform-tools-latest-{}.zip"
_MAX_DOWNLOAD = 200 * 1024 * 1024
_MAX_UNPACKED = 1024 * 1024 * 1024  # zip-bomb guard (platform-tools unpacks to ~30 MB)
QUEST_BROWSER = "com.oculus.browser"
# Quest 1, 2, Pro, 3, 3S device code names as reported by ``adb devices -l``
_QUEST_CODENAMES = ("monterey", "hollywood", "seacliff", "eureka", "panther")

_server_lock = threading.Lock()
_servers_started: set[str] = set()
_install_lock = threading.Lock()
_state = threading.local()

_HINT_NO_ADB = (
    "Install the Android platform tools from the Quest page (one-time download), "
    "or use the Wi-Fi address instead."
)
_HINT_NO_DEVICE = (
    "Connect the Quest with a USB cable, enable developer mode in the Meta Horizon phone app, "
    "then put on the headset and choose 'Allow USB debugging'."
)
_HINT_UNAUTHORIZED = "Put on the headset and choose 'Allow' (tick 'Always allow from this computer')."


def last_error() -> str:
    """Why the last function of this module called on *this thread* failed, with what to
    do about it (a sentence for the GUI); ``""`` after a success."""
    return getattr(_state, "error", "")


def _set_error(message: str, hint: str = "", level: int = logging.INFO) -> bool:
    """Record (and log) a user-facing failure reason; returns False for ``return _set_error(...)``."""
    text = f"{message} {hint}".strip()
    _state.error = text
    log.log(level, "%s", text)
    return False


def _clear_error() -> None:
    _state.error = ""


def _is_windows() -> bool:
    return sys.platform == "win32"


def _adb_name() -> str:
    return "adb.exe" if _is_windows() else "adb"


def tools_dir() -> Path:
    """``<data_root>/tools`` - where downloaded helper tools live."""
    return data_root() / "tools"


# --------------------------------------------------------------------------- locating adb
def _adb_candidates() -> list[Path]:
    name = _adb_name()
    cands: list[Path] = []
    override = os.environ.get("VRPHOTO_ADB")
    if override:
        cands.append(Path(override))
    # adb.exe only on Windows: a PATH shim named adb.bat/adb.cmd would run through cmd.exe,
    # which re-parses our arguments (URLs contain '&').
    on_path = shutil.which("adb.exe" if _is_windows() else "adb")
    if on_path:
        cands.append(Path(on_path))
    if _is_windows():
        local = os.environ.get("LOCALAPPDATA")
        if local:
            cands.append(Path(local) / "Android" / "Sdk" / "platform-tools" / name)
    cands.append(tools_dir() / "platform-tools" / name)
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        val = os.environ.get(var)
        if val:
            cands.append(Path(val) / "platform-tools" / name)
    home = Path.home()
    if _is_windows():
        appdata = os.environ.get("APPDATA")
        if appdata:  # SideQuest ships its own platform-tools
            cands.append(Path(appdata) / "SideQuest" / "platform-tools" / name)
        pf = os.environ.get("ProgramFiles")
        if pf:  # Meta Quest Developer Hub bundles adb
            cands.append(Path(pf) / "Meta Quest Developer Hub" / "resources" / "bin" / name)
    elif sys.platform == "darwin":
        cands.append(home / "Library" / "Android" / "sdk" / "platform-tools" / name)
    else:
        cands.append(home / "Android" / "Sdk" / "platform-tools" / name)
    return cands


def find_adb() -> str | None:
    """Path of a usable ``adb`` executable, or None.

    Search order: ``$VRPHOTO_ADB``, ``PATH``, ``%LOCALAPPDATA%/Android/Sdk/platform-tools``,
    ``<data_root>/tools/platform-tools`` (see :func:`install_platform_tools`),
    ``$ANDROID_HOME`` / ``$ANDROID_SDK_ROOT``, then SideQuest / Meta Quest Developer Hub
    / default SDK locations.
    """
    seen: set[str] = set()
    for cand in _adb_candidates():
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand.is_file() and (_is_windows() or os.access(cand, os.X_OK)):
                return str(cand)
        except OSError:
            continue
    return None


# --------------------------------------------------------------------------- platform-tools install
def platform_tools_url() -> str:
    """Download URL of Google's SDK Platform-Tools for this OS."""
    if _is_windows():
        return _PT_BASE.format("windows")
    if sys.platform == "darwin":
        return _PT_BASE.format("darwin")
    return _PT_BASE.format("linux")


def _download(url: str, dest: Path, progress: Callable[[float, str], None], cancel: CancelToken | None) -> None:
    hint = (
        "Check your internet connection, or install Google's 'SDK Platform-Tools' yourself "
        f"({PLATFORM_TOOLS_INFO_URL}) and add the folder to PATH."
    )
    req = urllib.request.Request(url, headers={"User-Agent": f"VRPhotoStudio/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            if total > _MAX_DOWNLOAD:
                raise DownloadError("The platform tools download is unexpectedly large.", hint=hint)
            done = 0
            progress(0.0, "Downloading Android platform tools")
            while True:
                if cancel is not None:
                    cancel.check()
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if done > _MAX_DOWNLOAD:
                    raise DownloadError("The platform tools download is unexpectedly large.", hint=hint)
                if total:
                    progress(done / total, f"Downloading Android platform tools ({done >> 20} MB)")
            if total and done < total:
                raise DownloadError("The platform tools download was interrupted.", hint=hint)
    except (Cancelled, DownloadError):
        raise
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
        reason = getattr(exc, "reason", None) or exc
        raise DownloadError(f"Could not download the Android platform tools: {reason}", hint=hint) from exc


def _extract(zpath: Path, dest: Path, progress: Callable[[float, str], None], cancel: CancelToken | None) -> None:
    """Extract ``platform-tools/`` from the archive with zip-slip protection."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    try:
        with zipfile.ZipFile(zpath) as zf:
            members = [m for m in zf.infolist() if m.filename.replace("\\", "/").startswith("platform-tools/")]
            if not members:
                raise DownloadError("The downloaded archive does not contain the Android platform tools.")
            if sum(m.file_size for m in members) > _MAX_UNPACKED:
                raise DownloadError("The downloaded platform tools archive is unexpectedly large.")
            for i, m in enumerate(members):
                if cancel is not None:
                    cancel.check()
                name = m.filename.replace("\\", "/")
                parts = [p for p in name.split("/") if p]
                if name.startswith("/") or any(p in ("..", ".") or ":" in p for p in parts):
                    raise DownloadError(f"The downloaded archive contains an unsafe path: {m.filename}")
                mode = (m.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    continue  # never materialise symlinks from an archive
                target = root.joinpath(*parts)
                if not target.resolve().is_relative_to(root):
                    raise DownloadError(f"The downloaded archive contains an unsafe path: {m.filename}")
                if m.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # ZipExtFile stops at the declared file_size and verifies the CRC
                with zf.open(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, 256 * 1024)
                if not _is_windows():
                    perm = mode & 0o777
                    if not perm:
                        perm = 0o755 if parts[-1] in ("adb", "fastboot") else 0o644
                    os.chmod(target, perm | stat.S_IRUSR | stat.S_IWUSR)
                progress((i + 1) / len(members), "Unpacking Android platform tools")
    except (zipfile.BadZipFile, EOFError, NotImplementedError) as exc:
        raise DownloadError(
            "The downloaded platform tools archive is damaged.", hint="Try the installation again."
        ) from exc
    except OSError as exc:
        raise DownloadError(
            f"Could not unpack the Android platform tools: {exc}",
            hint="Check that there is free disk space and the data folder is writable.",
        ) from exc


def _rmtree(path: Path) -> None:
    def _onerror(func, p, _exc) -> None:  # noqa: ANN001 - shutil callback
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_onerror)
    else:  # pragma: no cover - Python 3.11
        shutil.rmtree(path, onerror=_onerror)


def install_platform_tools(
    progress: ProgressReporter | None = None, cancel: CancelToken | None = None
) -> str:
    """Download and unpack Google's SDK Platform-Tools; return the path of ``adb``.

    Installs into ``<data_root>/tools/platform-tools`` (replacing an older copy; a running
    adb server from that copy is stopped first).  By downloading, the user accepts the
    Android SDK license terms - the GUI should say so.  Raises
    :class:`~vrphoto.core.errors.DownloadError` (with ``hint``) or ``Cancelled``.
    """

    def report(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, msg)

    def sub(a: float, b: float) -> Callable[[float, str], None]:
        return lambda f, m: report(a + (b - a) * min(1.0, max(0.0, f)), m)

    with _install_lock:
        return _install_platform_tools(report, sub, cancel)


def _replace_with_retry(src: Path, dst: Path, attempts: int = 10) -> None:
    """``os.replace`` that waits briefly for Windows to release files of a just-stopped adb."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.3)


def _install_platform_tools(report: Callable[[float, str], None],
                            sub: Callable[[float, float], Callable[[float, str], None]],
                            cancel: CancelToken | None) -> str:
    tools = tools_dir()
    try:
        tools.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DownloadError(
            f"Cannot create the folder {tools}: {exc}",
            hint="Check that the VR Photo Studio data folder is writable.",
        ) from exc
    url = platform_tools_url()
    tag = os.getpid()  # another process (GUI + CLI) must not share our temporary files
    zpath = tools / f"platform-tools-download-{tag}.zip"
    part = tools / f"platform-tools-download-{tag}.zip.part"
    stage = tools / f".platform-tools.new-{tag}"
    try:
        log.info("Downloading Android platform tools from %s", url)
        _download(url, part, sub(0.0, 0.85), cancel)
        os.replace(part, zpath)
        _rmtree(stage)
        _extract(zpath, stage, sub(0.85, 0.97), cancel)
        new_dir = stage / "platform-tools"
        if not (new_dir / _adb_name()).is_file():
            raise DownloadError("The downloaded archive does not contain adb.")
        if cancel is not None:
            cancel.check()
        dest = tools / "platform-tools"
        if dest.exists():
            old_adb = dest / _adb_name()
            if old_adb.is_file():
                _net._run([str(old_adb), "kill-server"], timeout=15)  # Windows locks a running adb.exe
            old = tools / f".platform-tools.old-{tag}"
            _rmtree(old)
            try:
                _replace_with_retry(dest, old)
            except OSError as exc:
                raise DownloadError(
                    f"Cannot replace the old platform tools in {dest}: {exc}",
                    hint="Close programs that use adb (SideQuest, Android Studio, Meta Quest Developer Hub) "
                    "and try again.",
                ) from exc
            _rmtree(old)
        try:
            _replace_with_retry(new_dir, dest)
        except OSError as exc:
            raise DownloadError(
                f"Cannot install the platform tools into {dest}: {exc}",
                hint="Close programs that use adb and try again.",
            ) from exc
        adb = dest / _adb_name()
        with _server_lock:
            _servers_started.discard(str(adb))
        report(1.0, "Android platform tools installed")
        log.info("Installed Android platform tools to %s", dest)
        return str(adb)
    finally:
        for p in (part, zpath):
            try:
                p.unlink()
            except OSError:
                pass
        _rmtree(stage)


# --------------------------------------------------------------------------- adb plumbing
def _ensure_server(adb: str) -> None:
    """Start the adb server once with stdio detached.

    A server spawned by a command whose output we capture can inherit our pipes and make
    that command hang until its timeout, so start it explicitly first.
    """
    with _server_lock:
        if adb in _servers_started:
            return
    try:
        rc = subprocess.run(
            [adb, "start-server"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
            **_net._no_window_kwargs(),
        ).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("Could not start the adb server: %s", exc)
        return
    if rc == 0:
        with _server_lock:
            _servers_started.add(adb)


def _adb(args: list[str], *, serial: str | None = None, timeout: float = 30.0,
         adb: str | None = None) -> subprocess.CompletedProcess[str] | None:
    adb = adb or find_adb()
    if not adb:
        log.info("adb was not found; install the Android platform tools first")
        return None
    _ensure_server(adb)
    cmd = [adb] + (["-s", serial] if serial else []) + list(args)
    cp = _net._run(cmd, timeout=timeout)
    if cp is None:
        log.info("adb %s did not complete", " ".join(args[:2]))
    elif cp.returncode != 0:
        log.info("adb %s failed: %s", " ".join(args[:2]), (cp.stderr or cp.stdout or "").strip()[:500])
    return cp


_PROP_RE = re.compile(r"^[a-z_]+:\S+$")


def _device_table() -> list[tuple[str, str, dict[str, str]]]:
    """``[(serial, state, props)]`` from ``adb devices -l``."""
    cp = _adb(["devices", "-l"], timeout=20)
    if cp is None or cp.returncode != 0:
        return []
    rows: list[tuple[str, str, dict[str, str]]] = []
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if not line or line.startswith("*") or line.lower().startswith("list of devices"):
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        props = dict(t.split(":", 1) for t in tokens[1:] if _PROP_RE.match(t))
        state = " ".join(t for t in tokens[1:] if not _PROP_RE.match(t))
        rows.append((tokens[0], state, props))
    return rows


def adb_device_states() -> dict[str, str]:
    """``{serial: state}`` for every attached device, e.g. ``"device"``, ``"unauthorized"``
    (accept "Allow USB debugging" inside the headset), ``"offline"``."""
    return {serial: state for serial, state, _ in _device_table()}


def adb_devices() -> list[str]:
    """Serials of connected, authorised devices (state ``device``); ``[]`` without adb."""
    out = []
    for serial, state, _ in _device_table():
        if state == "device":
            out.append(serial)
        elif state.startswith("unauthorized"):
            log.info("Device %s is not authorised: put on the headset and allow USB debugging", serial)
    return out


def _pick_serial(serial: str | None) -> str | None:
    """The given serial, else the only device, else the first one that looks like a Quest."""
    if serial:
        return serial
    rows = _device_table()
    table = [(s, p) for s, st, p in rows if st == "device"]
    if not table:
        if any(st.startswith("unauthorized") for _, st, _ in rows):
            _set_error("The headset has not allowed USB debugging yet.", _HINT_UNAUTHORIZED)
        elif any(st.startswith("no permissions") for _, st, _ in rows):
            _set_error("The headset is connected but this user may not access it.",
                       "On Linux, add a udev rule for the Quest (vendor id 2833) and reconnect the cable.")
        else:
            _set_error("No Quest is connected over USB with developer mode.", _HINT_NO_DEVICE)
        return None
    if len(table) > 1:
        for s, props in table:
            if "quest" in props.get("model", "").lower() or props.get("device", "") in _QUEST_CODENAMES:
                return s
        log.info("Several devices are connected; using %s", table[0][0])
    return table[0][0]


def _valid_port(port: int) -> int | None:
    try:
        p = int(port)
    except (TypeError, ValueError):
        return None
    return p if 0 < p < 65536 else None


def adb_reverse(port: int, serial: str | None = None) -> bool:
    """``adb reverse tcp:PORT tcp:PORT`` so ``http://localhost:PORT`` on the Quest reaches
    the PC.  Returns True on success, False if adb/the device is missing or it failed."""
    p = _valid_port(port)
    if p is None:
        return _set_error(f"Invalid port {port!r} for USB forwarding.", level=logging.WARNING)
    if not find_adb():
        return _set_error("The Android tools (adb) are not installed.", _HINT_NO_ADB)
    s = _pick_serial(serial)
    if s is None:
        return False
    cp = _adb(["reverse", f"tcp:{p}", f"tcp:{p}"], serial=s, timeout=20)
    if cp is None or cp.returncode != 0:
        detail = (cp.stderr or cp.stdout or "").strip()[-200:] if cp is not None else "no answer from adb"
        return _set_error(f"USB forwarding could not be set up ({detail}).",
                          "Reconnect the USB cable and try again.")
    _clear_error()
    log.info("USB forwarding active: http://localhost:%d on %s reaches this PC", p, s)
    return True


def adb_reverse_remove(port: int, serial: str | None = None) -> bool:
    """Undo :func:`adb_reverse` for ``port``."""
    p = _valid_port(port)
    if p is None or not find_adb():
        return False
    s = _pick_serial(serial)
    if s is None:
        return False
    cp = _adb(["reverse", "--remove", f"tcp:{p}"], serial=s, timeout=20)
    return cp is not None and cp.returncode == 0


def _run_cancellable(cmd: list[str], timeout: float, cancel: CancelToken | None,
                     on_line: Callable[[str], None] | None = None) -> tuple[int, str] | None:
    """Run ``cmd`` collecting merged output line by line; kill it on cancel/timeout.

    Returns ``(returncode, output)`` or None if it could not start / timed out.  Raises
    :class:`Cancelled` when cancelled.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_net._no_window_kwargs(),
        )
    except (OSError, ValueError) as exc:
        log.info("Could not start %s: %s", Path(cmd[0]).name, exc)
        return None
    lines: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                lines.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    t = threading.Thread(target=reader, name="vrphoto-subprocess-reader", daemon=True)
    t.start()
    out: list[str] = []
    deadline = time.monotonic() + timeout
    eof = False
    try:
        while not eof:
            if cancel is not None and cancel.cancelled:
                raise Cancelled("Operation cancelled")
            if time.monotonic() > deadline:
                log.warning("%s timed out after %.0f s", Path(cmd[0]).name, timeout)
                return None
            try:
                item = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                eof = True
                break
            out.append(item)
            if on_line is not None:
                on_line(item)
        try:
            rc = proc.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return None
        return rc, "\n".join(out)
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        t.join(timeout=2)


def adb_push(
    local: Path,
    remote_dir: str = "/sdcard/Movies",
    serial: str | None = None,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> bool:
    """Copy ``local`` (file or folder) into ``remote_dir`` on the headset via adb.

    ``/sdcard/Movies`` / ``/sdcard/Pictures`` show up in the Quest's media gallery; a
    media scan is requested afterwards (best effort).  Returns False if adb, the device
    or the file is missing or the copy failed; raises ``Cancelled`` on cancel.
    """
    local = Path(local)
    if not local.exists():
        return _set_error(f"Cannot copy {local.name} to the headset: the file was not found.",
                          level=logging.WARNING)
    adb = find_adb()
    if not adb:
        return _set_error("The Android tools (adb) are not installed.", _HINT_NO_ADB)
    s = _pick_serial(serial)
    if s is None:
        return False
    remote_dir = (remote_dir or "/sdcard/Movies").rstrip("/") or "/"
    remote = posixpath.join(remote_dir, local.name)
    # A folder is pushed *into* remote_dir (which exists after mkdir -p): adb then creates
    # or merges remote_dir/<name>.  Pushing to remote_dir/<name> would nest a second copy
    # (remote_dir/<name>/<name>) when the folder was copied before.
    push_target = remote_dir if local.is_dir() else remote
    if progress is not None:
        progress(0.0, f"Copying {local.name} to the headset")
    _adb(["shell", f"mkdir -p {shlex.quote(remote_dir)}"], serial=s, timeout=20, adb=adb)
    try:
        size = local.stat().st_size if local.is_file() else sum(
            f.stat().st_size for f in local.rglob("*") if f.is_file()
        )
    except OSError:
        size = 0
    timeout = 120.0 + size / (4 * 1024 * 1024)  # generous: >= 4 MB/s
    res = _run_cancellable([adb, "-s", s, "push", str(local), push_target], timeout, cancel)
    if res is None or res[0] != 0:
        detail = (res[1].strip().splitlines() or [""])[-1][-200:] if res else "it took too long"
        return _set_error(f"Copying {local.name} to the headset failed ({detail}).",
                          "Check the USB cable and that the headset has enough free space.",
                          level=logging.WARNING)
    _adb(
        ["shell", f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d {shlex.quote('file://' + remote)}"],
        serial=s, timeout=20, adb=adb,
    )
    if progress is not None:
        progress(1.0, f"Copied {local.name} to {remote_dir}")
    _clear_error()
    log.info("Copied %s to %s:%s", local, s, remote)
    return True


# RFC 3986 characters only: no whitespace, quotes, backticks or angle brackets.
_URL_RE = re.compile(r"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+")
_AM_ERROR_RE = re.compile(r"^\s*(Error|Security exception|java\.lang\.\w*Exception)", re.MULTILINE)


def open_url_on_quest(url: str, serial: str | None = None) -> bool:
    """Open ``url`` in Quest Browser on the USB-connected headset (``adb shell am start``).

    Falls back to the default browser if Quest Browser is not installed.  Returns False
    for non-http(s) URLs or when adb/the device is unavailable.
    """
    if not _URL_RE.fullmatch(url or ""):
        return _set_error(f"Not a web address that can be opened on the headset: {url!r}",
                          level=logging.WARNING)
    if not find_adb():
        return _set_error("The Android tools (adb) are not installed.", _HINT_NO_ADB)
    s = _pick_serial(serial)
    if s is None:
        return False
    base = f"am start -a android.intent.action.VIEW -d {shlex.quote(url)}"
    for cmd in (f"{base} {QUEST_BROWSER}", base):
        cp = _adb(["shell", cmd], serial=s, timeout=20)
        if cp is None:
            return _set_error("adb did not answer.", "Reconnect the USB cable and try again.")
        text = f"{cp.stdout}\n{cp.stderr}"
        if cp.returncode == 0 and not _AM_ERROR_RE.search(text):
            _clear_error()
            log.info("Opened %s on %s", url, s)
            return True
        log.info("am start failed (%s); retrying", text.strip()[:300])
    return _set_error("The browser could not be opened on the headset.",
                      "Put on the headset (it must be awake) and try again.")


# --------------------------------------------------------------------------- MTP (Windows)
_MTP_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$files = @(__FILES__)
$sub = @(__SUB__)
$hint = __HINT__
function Say([string]$s) { [Console]::Out.WriteLine($s); [Console]::Out.Flush() }
function FindChild($folder, [string]$name, [bool]$wantFolder) {
  # exact name first; Explorer may hide known extensions, so then try the bare name
  $base = [System.IO.Path]::GetFileNameWithoutExtension($name)
  $loose = $null
  foreach ($it in $folder.Items()) {
    if ($wantFolder -and -not $it.IsFolder) { continue }
    if ($it.Name -eq $name) { return $it }
    if (-not $wantFolder -and -not $loose -and $it.Name -eq $base) { $loose = $it }
  }
  return $loose
}
function ItemSize($it) {
  # FolderItem.Size is a 32-bit Long (wrong above 2 GB); System.Size is 64-bit
  try {
    $v = $it.ExtendedProperty('System.Size')
    if ($null -ne $v -and "$v" -ne '') { return [int64]$v }
  } catch { }
  try { $s = [int64]$it.Size; if ($s -gt 0 -and $s -lt 2147483648) { return $s } } catch { }
  return [int64]-1
}
try {
  $shell = New-Object -ComObject Shell.Application
  $pc = $shell.NameSpace(17)
  if (-not $pc) { Say 'ERR:NOPC:The This PC folder is not available.'; exit 3 }
  $dev = $null
  foreach ($it in $pc.Items()) { if ($it.Name -like "*$hint*") { $dev = $it; break } }
  if (-not $dev) { Say 'ERR:NODEVICE:No Quest headset was found under This PC.'; exit 2 }
  $items = @($dev.GetFolder.Items())
  $storage = $null
  foreach ($it in $items) { if ($it.IsFolder -and ($it.Name -like '*shared*' -or $it.Name -like '*Internal*')) { $storage = $it; break } }
  if (-not $storage) { foreach ($it in $items) { if ($it.IsFolder) { $storage = $it; break } } }
  if (-not $storage) { Say 'ERR:NOSTORAGE:The headset storage is not accessible.'; exit 4 }
  $target = $storage.GetFolder
  foreach ($name in $sub) {
    $next = FindChild $target $name $true
    if (-not $next) {
      $target.NewFolder($name)
      for ($w = 0; ($w -lt 50) -and (-not $next); $w++) { Start-Sleep -Milliseconds 200; $next = FindChild $target $name $true }
      if (-not $next) { Say "ERR:MKDIR:Could not create the folder $name on the headset."; exit 5 }
    }
    $target = $next.GetFolder
  }
  $i = 0
  foreach ($f in $files) {
    $i++
    $leaf = [System.IO.Path]::GetFileName($f)
    $size = [int64](Get-Item -LiteralPath $f).Length
    Say "START $i $leaf"
    # An existing file of the same name would otherwise be taken for the finished copy
    # (the script would exit and abort the real transfer).  Same size -> already there.
    $old = FindChild $target $leaf $false
    $oldSize = [int64]-2
    if ($old) {
      $oldSize = ItemSize $old
      if ($oldSize -eq $size) { Say "SKIP $i $leaf"; Say "OK $i"; continue }
    }
    $target.CopyHere($f, 4 + 16 + 512 + 1024)
    $deadline = (Get-Date).AddSeconds(120 + $size / 2MB)
    $seen = 0
    while ($true) {
      Start-Sleep -Milliseconds 500
      $it = FindChild $target $leaf $false
      if ($it) {
        $cur = ItemSize $it
        if ($cur -eq $size) { break }
        # size not reported: accept once the item has been present for ~2 s (only if it
        # did not exist before, else it may still be the old file)
        if ($cur -lt 0 -and -not $old) { $seen++; if ($seen -ge 4) { break } }
      }
      if ((Get-Date) -gt $deadline) { Say "ERR:TIMEOUT:Copying $leaf to the headset timed out."; exit 6 }
    }
    Say "OK $i"
  }
  # CopyHere runs on a background thread of this process: give the copy engine a moment
  # to finish its bookkeeping before PowerShell exits.
  Start-Sleep -Seconds 2
  Say 'DONE'
  exit 0
} catch {
  Say ('ERR:EXCEPTION:' + $_.Exception.Message)
  exit 1
}
"""

_MTP_HINTS = {
    "NODEVICE": "Connect the Quest with a USB cable, put it on and choose 'Allow' when asked about file access.",
    "NOSTORAGE": "Put on the headset and allow access to data when the USB prompt appears.",
    "TIMEOUT": "Keep the headset awake and connected; Windows may be showing a dialog waiting for an answer.",
    "MKDIR": "Put on the headset and allow access to data when the USB prompt appears.",
}


def _mtp_script(files: list[Path], subfolder_parts: list[str], device_hint: str = "Quest") -> str:
    """PowerShell source for :func:`mtp_copy` (exposed for tests)."""
    q = _net._ps_quote
    return (
        _MTP_SCRIPT.replace("__FILES__", ", ".join(q(str(f)) for f in files))
        .replace("__SUB__", ", ".join(q(p) for p in subfolder_parts))
        .replace("__HINT__", q(device_hint))
    )


def mtp_copy(
    files: list[Path],
    subfolder: str = "Movies",
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> bool:
    """Copy ``files`` to ``Internal shared storage/<subfolder>`` of a USB-connected Quest
    through Windows' MTP support (no developer mode needed).

    Uses ``Shell.Application`` (namespace 17 = This PC) from a generated PowerShell script
    and waits until each copy has finished (Explorer may show its own progress window).
    Returns False with a logged reason when not on Windows, no headset is found, or a copy
    fails; raises ``Cancelled`` if cancelled.
    """
    if not _is_windows():
        return _set_error("Copying over USB without developer mode is only available on Windows.",
                          "Use the Wi-Fi viewer or adb instead.")
    paths = [Path(f).resolve() for f in files]
    missing = [p.name for p in paths if not p.is_file()]
    if not paths or missing:
        return _set_error("Nothing to copy: " + (", ".join(missing) + " not found." if missing else "no files given."),
                          level=logging.WARNING)
    try:
        total = sum(p.stat().st_size for p in paths)
    except OSError as exc:
        return _set_error(f"The files to copy cannot be read: {exc}", level=logging.WARNING)
    parts = [p for p in re.split(r"[\\/]+", subfolder or "") if p]
    if any(p in (".", "..") or ":" in p or any(ord(c) < 32 for c in p) for p in parts):
        return _set_error(f"Invalid folder on the headset: {subfolder!r}", level=logging.WARNING)
    try:
        script = _net._write_temp_script(_mtp_script(paths, parts), "vrphoto_mtp_")
    except OSError as exc:
        return _set_error(f"Cannot write the copy helper script: {exc}",
                          "Check that the TEMP folder is writable.", level=logging.WARNING)
    try:
        args = _net._ps_args(script, sta=True)
        if args is None:
            return _set_error("Copying over USB needs Windows PowerShell, which was not found.",
                              level=logging.WARNING)
        # must outlast the script's own per-file deadlines (120 s + 2 MB/s each) + waits
        timeout = 120.0 + total / (2 * 1024 * 1024) + 150.0 * len(paths)
        n = len(paths)
        state = {"error": ""}

        def on_line(line: str) -> None:
            if line.startswith("ERR:"):
                state["error"] = line
                return
            if progress is None:
                return
            m = re.match(r"^(START|OK) (\d+)(?: (.*))?$", line)
            if m is None:
                return
            idx = min(max(int(m.group(2)), 1), n)
            if m.group(1) == "START":
                progress((idx - 1) / n, f"Copying {m.group(3) or ''} to the headset".replace("  ", " "))
            else:
                progress(idx / n, "Copied to the headset")

        res = _run_cancellable(args, timeout, cancel, on_line)
        if res is None:
            return _set_error("Copying to the headset did not finish in time.",
                              "Check the USB cable and try again with fewer files.", level=logging.WARNING)
        rc, out = res
        if rc != 0 or "DONE" not in out.splitlines():
            err = state["error"] or out.strip()[-300:]
            code = err.split(":")[1] if err.startswith("ERR:") and err.count(":") >= 2 else ""
            msg = err.split(":", 2)[2] if code else (err or "PowerShell gave no answer.")
            return _set_error(f"Copying to the headset failed: {msg}", _MTP_HINTS.get(code, ""),
                              level=logging.WARNING)
        _clear_error()
        log.info("Copied %d file(s) to the headset folder %s", n, "/".join(parts) or "(root)")
        return True
    finally:
        try:
            script.unlink()
        except OSError:
            pass
