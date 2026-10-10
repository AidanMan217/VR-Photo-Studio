"""clean_env / popen_kwargs / kill_process_tree."""

from __future__ import annotations

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
