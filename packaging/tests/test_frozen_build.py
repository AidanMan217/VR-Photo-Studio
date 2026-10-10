"""End-to-end PyInstaller build of the real spec (slow, opt-in).

Run with::

    VRPHOTO_TEST_PYINSTALLER=1 python -m pytest -q packaging/tests/test_frozen_build.py

It bundles the app into a temporary folder, checks the bundle contents and runs the frozen
CLI (``--help``, plus ``--selftest --json`` when ``VRPHOTO_TEST_FROZEN_SELFTEST=1``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import spec_helpers as H

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("VRPHOTO_TEST_PYINSTALLER") != "1", reason="set VRPHOTO_TEST_PYINSTALLER=1"
    ),
]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    # opted in with VRPHOTO_TEST_PYINSTALLER=1: a missing prerequisite is a failure, not a skip
    assert importlib.util.find_spec("PyInstaller") is not None, "pip install pyinstaller (in .[dev])"
    assert importlib.util.find_spec("vrphoto.__main__") is not None, "src/vrphoto/__main__.py is missing"
    base = tmp_path_factory.mktemp("pyi")
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(H.HERE / "vrphoto.spec"),
        "--noconfirm",
        "--distpath",
        str(base / "dist"),
        "--workpath",
        str(base / "build"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=H.ROOT)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return base / "dist" / H.COLLECT_NAME


def _exe(bundle: Path, name: str) -> Path:
    return bundle / (f"{name}.exe" if sys.platform == "win32" else name)


def test_bundle_contents(bundle):
    assert H.check_bundle(bundle) == []
    internal = bundle / "_internal"
    assert (internal / "vrphoto" / "webxr" / "viewer" / "scene.schema.json").is_file()
    assert any((internal / "imageio_ffmpeg" / "binaries").glob("ffmpeg*"))
    uv = internal / "bin" / H.uv_exe_name()
    assert uv.is_file() and os.access(uv, os.X_OK)


def test_frozen_cli_help(bundle):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run(
        [str(_exe(bundle, H.CLI_EXE)), "--help"], capture_output=True, text=True, timeout=300, env=env
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.skipif(
    os.environ.get("VRPHOTO_TEST_FROZEN_SELFTEST") != "1", reason="set VRPHOTO_TEST_FROZEN_SELFTEST=1"
)
def test_frozen_selftest(bundle, tmp_path):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", VRPHOTO_HOME=str(tmp_path))
    proc = subprocess.run(
        [str(_exe(bundle, H.CLI_EXE)), "--selftest", "--json"],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    start = proc.stdout.find("{")
    assert start >= 0
    result, _ = json.JSONDecoder().raw_decode(proc.stdout[start:])
    assert result.get("ok") is True, result


def test_frozen_gui_exe_runs_cli_commands(bundle):
    """The windowed exe with arguments behaves like the CLI (stdout may be None on Windows)."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run(
        [str(_exe(bundle, H.GUI_EXE)), "--version"], capture_output=True, text=True, timeout=300, env=env
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="windowed exe: use the CI build job on Windows")
def test_frozen_gui_starts(bundle, tmp_path):
    """The GUI starts in the frozen app (offscreen) and stays up; its log says it is ready."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", VRPHOTO_HOME=str(tmp_path))
    proc = subprocess.Popen(
        [str(_exe(bundle, H.GUI_EXE))],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    log = tmp_path / "logs" / "vrphoto.log"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"GUI exited with code {proc.returncode}:\n{out[-4000:]}")
            if log.is_file() and "ready" in log.read_text(encoding="utf-8", errors="replace"):
                break
            time.sleep(0.25)
        else:
            pytest.fail("the GUI did not log 'ready' within 60 s")
        assert "Traceback" not in log.read_text(encoding="utf-8", errors="replace")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
