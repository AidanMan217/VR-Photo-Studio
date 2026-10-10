"""Helpers shared by the add-on tests (imported by conftest and test modules)."""

from __future__ import annotations

import shutil
import sys
import textwrap
import time
from pathlib import Path

import pytest

PY_MM = f"{sys.version_info.major}.{sys.version_info.minor}"


def write_manifest(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{name}.toml"
    p.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return p


DUMMY_HEADER = f"""
id = "{{id}}"
title = "Dummy {{id}}"
description = "Test add-on"
version = "{{version}}"
license = "MIT"
kind = "venv"
python = "{PY_MM}"
tasks = ["echo", "sleep", "crash"]
worker = "dummy_worker.py"
"""


def dummy_manifest(addon_id: str = "dummy", *, pip: bool = False, version: str = "1.0", extra_steps: str = "",
                   header_extra: str = "") -> str:
    body = DUMMY_HEADER.format(id=addon_id, version=version) + header_extra + '\n[[steps]]\ntype = "venv"\n'
    if pip:
        body += '\n[[steps]]\ntype = "pip"\ntitle = "Installing six"\npackages = ["six==1.17.0"]\n'
    body += extra_steps
    body += '\n[[steps]]\ntype = "worker"\nargs = ["--prefetch"]\n\n[[steps]]\ntype = "check"\n'
    return body


def raw_worker(tmp_path: Path, name: str, code: str) -> Path:
    """A worker script that writes raw lines to stdout (no _common plumbing)."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(code).lstrip(), encoding="utf-8")
    return p


def which_or_skip(name: str) -> str:
    p = shutil.which(name)
    if not p:
        pytest.skip(f"{name} not available")
    return p


def wait_dead(pid: int, timeout: float = 3.0) -> bool:
    """True once ``pid`` is gone or a zombie (containers may not reap orphans)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
                state = fh.read().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False
