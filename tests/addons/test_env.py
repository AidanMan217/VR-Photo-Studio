"""clean_env / popen_kwargs / kill_process_tree."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from _addon_helpers import wait_dead

from vrphoto.addons import env as addon_env
from vrphoto.addons.env import clean_env, kill_process_tree, popen_kwargs, prepend_path, venv_python


@pytest.fixture
def fake_meipass(tmp_path, monkeypatch) -> Path:
    mp = tmp_path / "_internal"
    mp.mkdir()
    monkeypatch.setattr(sys, "_MEIPASS", str(mp), raising=False)
    return mp


LEAKY = {
    "PYTHONHOME": "x",
    "PYTHONPATH": "x",
    "PYTHONSTARTUP": "x",
    "VIRTUAL_ENV": "x",
    "CONDA_PREFIX": "x",
    "QT_PLUGIN_PATH": "x",
    "QML2_IMPORT_PATH": "x",
    "OPENSSL_MODULES": "x",
    "TCL_LIBRARY": "x",
    "TK_LIBRARY": "x",
    "_MEIPASS2": "x",
    "_PYI_APPLICATION_HOME_DIR": "x",
    "_PYI_ARCHIVE_FILE": "x",
    "_PYI_PARENT_PROCESS_LEVEL": "1",
}


def test_removes_leaking_vars(monkeypatch, fake_meipass):
    for k, v in LEAKY.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("KEEP_ME", "yes")
    env = clean_env()
    for k in LEAKY:
        assert k not in env, k
    assert env["KEEP_ME"] == "yes"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert "KEEP_ME" in os.environ  # os.environ itself untouched
    assert os.environ["PYTHONHOME"] == "x"


def test_strips_meipass_path_entries(monkeypatch, fake_meipass, tmp_path):
    other = tmp_path / "bin"
    inside = fake_meipass / "PySide6"
    path = os.pathsep.join([str(fake_meipass), str(inside), str(other), str(other)])
    monkeypatch.setenv("PATH", path)
    env = clean_env()
    parts = env["PATH"].split(os.pathsep)
    assert str(fake_meipass) not in parts and str(inside) not in parts
    assert parts == [str(other)]  # de-duplicated


def test_meipass_prefix_not_confused(monkeypatch, fake_meipass, tmp_path):
    sibling = Path(str(fake_meipass) + "_other")  # shares the prefix, not inside
    monkeypatch.setenv("PATH", str(sibling))
    assert clean_env()["PATH"] == str(sibling)


def test_ssl_cert_only_inside_bundle(monkeypatch, fake_meipass, tmp_path):
    monkeypatch.setenv("SSL_CERT_FILE", str(fake_meipass / "certifi" / "cacert.pem"))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(fake_meipass / "certifi" / "cacert.pem"))
    assert "SSL_CERT_FILE" not in clean_env()
    assert "REQUESTS_CA_BUNDLE" not in clean_env()
    user_ca = tmp_path / "corp.pem"
    monkeypatch.setenv("SSL_CERT_FILE", str(user_ca))
    assert clean_env()["SSL_CERT_FILE"] == str(user_ca)


def test_ld_library_path_orig_restored(monkeypatch, fake_meipass):
    monkeypatch.setenv("LD_LIBRARY_PATH", str(fake_meipass))
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/usr/local/cuda/lib64")
    env = clean_env()
    assert env["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64"
    assert "LD_LIBRARY_PATH_ORIG" not in env


def test_ld_library_path_dropped_when_only_bundle(monkeypatch, fake_meipass):
    monkeypatch.setenv("LD_LIBRARY_PATH", str(fake_meipass))
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    assert "LD_LIBRARY_PATH" not in clean_env()


def test_not_frozen_keeps_path(monkeypatch):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setenv("PATH", "/a" + os.pathsep + "/b")
    assert clean_env()["PATH"].split(os.pathsep) == ["/a", "/b"]


def test_extra_overrides_and_removes(monkeypatch):
    monkeypatch.setenv("FOO", "1")
    monkeypatch.setenv("BAR", "2")
    env = clean_env({"FOO": None, "BAZ": 3, "PYTHONUNBUFFERED": "0"})
    assert "FOO" not in env and env["BAR"] == "2" and env["BAZ"] == "3"
    assert env["PYTHONUNBUFFERED"] == "0"  # extra wins


def test_windows_case_insensitive(monkeypatch, fake_meipass):
    monkeypatch.setattr(addon_env, "_IS_WINDOWS", True)
    monkeypatch.setenv("PythonPath", "x")
    if sys.platform != "win32":
        monkeypatch.delenv("PATH", raising=False)
    monkeypatch.setenv("Path", os.pathsep.join([str(fake_meipass), "/keep"]))
    env = clean_env()
    assert not any(k.upper() == "PYTHONPATH" for k in env)
    path_keys = [k for k in env if k.upper() == "PATH"]
    assert len(path_keys) == 1 and env[path_keys[0]] == "/keep"


def test_prepend_path():
    env = {"PATH": "/b"}
    prepend_path(env, "/a", "/b")
    assert env["PATH"].split(os.pathsep) == ["/a", "/b"]


def test_popen_kwargs_posix(monkeypatch):
    monkeypatch.setattr(addon_env, "_IS_WINDOWS", False)
    assert popen_kwargs() == {"start_new_session": True}


def test_popen_kwargs_windows_branch(monkeypatch):
    calls = []
    monkeypatch.setattr(addon_env, "_IS_WINDOWS", True)
    monkeypatch.setattr(addon_env, "_reset_dll_directory", lambda: calls.append(1))
    kw = popen_kwargs()
    assert calls == [1]
    assert kw["creationflags"] & 0x08000000  # CREATE_NO_WINDOW
    assert kw["creationflags"] & 0x00000200  # CREATE_NEW_PROCESS_GROUP


def test_venv_python_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(addon_env, "_IS_WINDOWS", True)
    assert venv_python(tmp_path) == tmp_path / "Scripts" / "python.exe"
    monkeypatch.setattr(addon_env, "_IS_WINDOWS", False)
    assert venv_python(tmp_path) == tmp_path / "bin" / "python"


def test_kill_process_tree_kills_children(tmp_path):
    marker = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time, pathlib\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(c.pid))\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], **popen_kwargs())
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists()
    child = int(marker.read_text())
    t0 = time.monotonic()
    kill_process_tree(proc)
    assert proc.poll() is not None
    assert time.monotonic() - t0 < 3
    if sys.platform.startswith("linux"):
        assert wait_dead(child), "child process survived the tree kill"
    kill_process_tree(proc)  # idempotent on a dead process



# ------------------------------------------------------------------ native tools (COLMAP / Brush)
def _layout(root: Path, *dirs: str) -> Path:
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    exe = root / "bin" / "colmap.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"MZ")
    return exe


def test_native_tool_env_colmap4_layout(tmp_path, monkeypatch):
    """COLMAP 4.x Windows zip: bin/colmap.exe + plugins/platforms (what COLMAP.bat sets)."""
    exe = _layout(tmp_path / "colmap", "plugins/platforms")
    monkeypatch.setenv("QT_PLUGIN_PATH", "/leaked/from/pyside")
    env = addon_env.native_tool_env(exe, {"RUST_LOG": "info"})
    assert env["QT_PLUGIN_PATH"] == str(tmp_path / "colmap" / "plugins")
    assert env["PATH"].split(os.pathsep)[0] == str((tmp_path / "colmap" / "bin").resolve())
    assert env["RUST_LOG"] == "info"


def test_native_tool_env_legacy_lib_layout(tmp_path):
    exe = _layout(tmp_path / "colmap", "lib")
    env = addon_env.native_tool_env(exe)
    assert env["QT_PLUGIN_PATH"] == str(tmp_path / "colmap" / "lib")
    assert str(tmp_path / "colmap" / "lib") in env["PATH"].split(os.pathsep)


def test_native_tool_env_no_plugins_no_qt_path(tmp_path, monkeypatch):
    exe = tmp_path / "brush" / "brush_app.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("QT_PLUGIN_PATH", "/leaked")
    assert "QT_PLUGIN_PATH" not in addon_env.native_tool_env(exe)


def test_native_tool_env_offscreen_only_headless_linux(tmp_path, monkeypatch):
    exe = _layout(tmp_path / "c")
    monkeypatch.setattr(addon_env.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    assert addon_env.native_tool_env(exe)["QT_QPA_PLATFORM"] == "offscreen"
    monkeypatch.setenv("DISPLAY", ":0")
    assert "QT_QPA_PLATFORM" not in addon_env.native_tool_env(exe)
    monkeypatch.setattr(addon_env.sys, "platform", "win32")
    monkeypatch.delenv("DISPLAY")
    assert "QT_QPA_PLATFORM" not in addon_env.native_tool_env(exe)  # native qwindows works hidden


def test_taskkill_is_absolute_when_available(tmp_path, monkeypatch):
    (tmp_path / "System32").mkdir()
    (tmp_path / "System32" / "taskkill.exe").write_bytes(b"MZ")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    assert addon_env._taskkill() == str(tmp_path / "System32" / "taskkill.exe")
    monkeypatch.setenv("SystemRoot", str(tmp_path / "nope"))
    monkeypatch.delenv("windir", raising=False)
    assert addon_env._taskkill() == "taskkill"


# ------------------------------------------------------------------ WIN-3: kill-on-close job object
@pytest.mark.skipif(sys.maxsize <= 2**32, reason="layout check for 64-bit Windows")
def test_job_structures_match_windows_x64_layout():
    import ctypes

    assert ctypes.sizeof(addon_env._JOBOBJECT_BASIC_LIMIT_INFORMATION) == 64
    assert ctypes.sizeof(addon_env._IO_COUNTERS) == 48
    assert ctypes.sizeof(addon_env._JOBOBJECT_EXTENDED_LIMIT_INFORMATION) == 144
    assert addon_env._JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags.offset == 16


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_assign_to_kill_job_is_noop_off_windows():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        assert addon_env.assign_to_kill_job(p) is False
    finally:
        p.wait(10)


class _FakeKernel32:
    def __init__(self, *, create_ok=True, set_ok=True, assign_ok=True):
        self.create_ok, self.set_ok, self.assign_ok = create_ok, set_ok, assign_ok
        self.calls: list[tuple] = []

    def CreateJobObjectW(self, attrs, name):  # noqa: N802
        self.calls.append(("create", attrs, name))
        return 1234 if self.create_ok else 0

    def SetInformationJobObject(self, job, cls, info, size):  # noqa: N802
        self.calls.append(("set", job, cls, info._obj.BasicLimitInformation.LimitFlags, size))
        return 1 if self.set_ok else 0

    def AssignProcessToJobObject(self, job, handle):  # noqa: N802
        self.calls.append(("assign", job, handle))
        return 1 if self.assign_ok else 0

    def CloseHandle(self, h):  # noqa: N802
        self.calls.append(("close", h))
        return 1


@pytest.fixture
def fake_windows_job(monkeypatch):
    import ctypes

    def install(**kw):
        k32 = _FakeKernel32(**kw)
        monkeypatch.setattr(addon_env, "_IS_WINDOWS", True)
        monkeypatch.setattr(addon_env, "_kernel32", lambda: k32)
        monkeypatch.setattr(addon_env, "_kill_job", None)
        monkeypatch.setattr(addon_env, "_kill_job_failed", False)
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
        monkeypatch.setattr(ctypes, "WinError", lambda code=None: OSError(code, "fake"), raising=False)
        return k32

    return install


def test_assign_to_kill_job_creates_one_kill_on_close_job(fake_windows_job):
    import ctypes
    from types import SimpleNamespace

    k32 = fake_windows_job()
    p1, p2 = SimpleNamespace(_handle=11, pid=1), SimpleNamespace(_handle=22, pid=2)
    assert addon_env.assign_to_kill_job(p1) is True
    assert addon_env.assign_to_kill_job(p2) is True
    kinds = [c[0] for c in k32.calls]
    assert kinds == ["create", "set", "assign", "assign"]  # one job for the whole app
    _, job, cls, flags, size = k32.calls[1]
    assert job == 1234 and cls == 9 and flags == addon_env.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE == 0x2000
    assert size == ctypes.sizeof(addon_env._JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
    assert k32.calls[2] == ("assign", 1234, 11) and k32.calls[3] == ("assign", 1234, 22)
    assert "close" not in kinds  # the job handle lives as long as the app


def test_assign_to_kill_job_failures_never_raise(fake_windows_job):
    from types import SimpleNamespace

    k32 = fake_windows_job(set_ok=False)
    assert addon_env.assign_to_kill_job(SimpleNamespace(_handle=1, pid=1)) is False
    assert ("close", 1234) in k32.calls  # half-made job is cleaned up
    assert addon_env.assign_to_kill_job(SimpleNamespace(_handle=1, pid=1)) is False
    assert [c[0] for c in k32.calls].count("create") == 1  # not retried for every process

    k32 = fake_windows_job(assign_ok=False)
    assert addon_env.assign_to_kill_job(SimpleNamespace(_handle=1, pid=1)) is False
    assert addon_env.assign_to_kill_job(SimpleNamespace(pid=1)) is False  # no handle


def test_spawned_add_on_processes_join_the_kill_job(monkeypatch, tmp_path):
    from vrphoto.addons import runner

    seen = []
    monkeypatch.setattr(runner, "assign_to_kill_job", lambda proc: seen.append(proc.pid) or True)
    res = runner.run_process([sys.executable, "-c", "print('hi')"], title="t")
    assert res.ok and len(seen) == 1


def _win_pid_alive(pid: int) -> bool:
    import ctypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    h = k32.OpenProcess(0x00100000, 0, pid)  # SYNCHRONIZE
    if not h:
        return False
    try:
        return k32.WaitForSingleObject(h, 0) != 0  # WAIT_OBJECT_0 -> exited
    finally:
        k32.CloseHandle(h)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
def test_native_tool_dies_with_the_app_on_windows(tmp_path):
    """Ending the app in Task Manager (TerminateProcess) must also end COLMAP / Brush."""
    pid_file = tmp_path / "child.pid"
    child = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(120)"
    app = tmp_path / "app.py"
    app.write_text(
        "import sys\n"
        "from vrphoto.addons.runner import run_process\n"
        f"run_process([sys.executable, '-c', {child!r}], title='tool')\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(app)])
    try:
        deadline = time.monotonic() + 20
        while not (pid_file.is_file() and pid_file.read_text().strip()):
            assert time.monotonic() < deadline, "tool did not start"
            assert proc.poll() is None, "app died early"
            time.sleep(0.05)
        pid = int(pid_file.read_text())
        proc.kill()  # TerminateProcess, like Task Manager's "End task"
        proc.wait(10)
        deadline = time.monotonic() + 10
        while _win_pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _win_pid_alive(pid), "the add-on tool kept running after the app died"
    finally:
        if proc.poll() is None:
            proc.kill()
        if pid_file.is_file() and pid_file.read_text().strip():
            with contextlib.suppress(OSError):
                os.kill(int(pid_file.read_text()), 9)  # never leave the sleeper behind
