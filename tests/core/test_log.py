"""core.log: rotation tolerates a log file locked by another process (WIN-1)."""

from __future__ import annotations

import logging
import os

import pytest

from vrphoto.core import log as core_log


def _record(msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord("vrphoto.test", logging.INFO, __file__, 1, msg, None, None)


def _handler(path, max_bytes=200, backups=3):
    h = core_log.SafeRotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    h.setFormatter(logging.Formatter(core_log.LOG_FORMAT))
    return h


def test_rotation_works_normally(tmp_path):
    p = tmp_path / "vrphoto.log"
    p.write_text("x" * 500, encoding="utf-8")
    (tmp_path / "vrphoto.log.1").write_text("one", encoding="utf-8")
    h = _handler(p)
    try:
        h.emit(_record("after"))
    finally:
        h.close()
    assert (tmp_path / "vrphoto.log.1").read_text(encoding="utf-8") == "x" * 500
    assert (tmp_path / "vrphoto.log.2").read_text(encoding="utf-8") == "one"
    assert "after" in p.read_text(encoding="utf-8")
    assert not (tmp_path / "vrphoto.log.rotating").exists()


def test_locked_file_keeps_logging_without_tracebacks(tmp_path, monkeypatch, capsys):
    """Simulates Windows WinError 32: the live log cannot be renamed while another process holds it."""
    p = tmp_path / "vrphoto.log"
    p.write_text("x" * 500, encoding="utf-8")
    (tmp_path / "vrphoto.log.1").write_text("one", encoding="utf-8")
    (tmp_path / "vrphoto.log.2").write_text("two", encoding="utf-8")
    real_rename = os.rename
    attempts = {"n": 0}

    def locked_rename(src, dst):
        if os.path.abspath(src) == os.path.abspath(p):
            attempts["n"] += 1
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real_rename(src, dst)

    monkeypatch.setattr(core_log.os, "rename", locked_rename)
    monkeypatch.setattr(logging, "raiseExceptions", True)
    h = _handler(p)
    try:
        for i in range(5):
            h.emit(_record(f"line {i}"))
    finally:
        h.close()
    text = p.read_text(encoding="utf-8")
    assert all(f"line {i}" in text for i in range(5)), "records must not be lost"
    assert attempts["n"] == 1, "rotation must be postponed, not retried on every record"
    assert "Logging error" not in capsys.readouterr().err
    # backups were not shifted by the failed attempt
    assert (tmp_path / "vrphoto.log.1").read_text(encoding="utf-8") == "one"
    assert (tmp_path / "vrphoto.log.2").read_text(encoding="utf-8") == "two"
    assert not (tmp_path / "vrphoto.log.3").exists()


def test_rotation_retried_after_backoff(tmp_path, monkeypatch):
    p = tmp_path / "vrphoto.log"
    p.write_text("x" * 500, encoding="utf-8")
    real_rename = os.rename
    state = {"locked": True, "n": 0}

    def rename(src, dst):
        if state["locked"] and os.path.abspath(src) == os.path.abspath(p):
            state["n"] += 1
            raise PermissionError(13, "locked")
        return real_rename(src, dst)

    clock = {"t": 1000.0}
    monkeypatch.setattr(core_log.os, "rename", rename)
    monkeypatch.setattr(core_log, "_now", lambda: clock["t"])
    h = _handler(p)
    try:
        h.emit(_record("a"))
        assert state["n"] == 1
        clock["t"] += h.RETRY_FIRST_S - 1
        h.emit(_record("b"))
        assert state["n"] == 1
        clock["t"] += 2
        h.emit(_record("c"))  # second failure: next pause is doubled
        assert state["n"] == 2
        clock["t"] += h.RETRY_FIRST_S + 1
        h.emit(_record("d"))
        assert state["n"] == 2
        state["locked"] = False
        clock["t"] += h.RETRY_FIRST_S * 2
        h.emit(_record("e"))
    finally:
        h.close()
    assert (tmp_path / "vrphoto.log.1").exists()
    old = (tmp_path / "vrphoto.log.1").read_text(encoding="utf-8")
    assert old.startswith("x" * 500) and "d" in old
    assert p.read_text(encoding="utf-8").strip().endswith("e")


def test_other_logging_errors_print_one_traceback_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(logging, "raiseExceptions", True)
    h = _handler(tmp_path / "vrphoto.log", max_bytes=0)
    try:
        def broken(record):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(logging.FileHandler, "emit", lambda self, record: broken(record))
        for _ in range(4):
            h.emit(_record())
    finally:
        monkeypatch.undo()
        h.close()
    assert capsys.readouterr().err.count("Logging error") == 1


@pytest.fixture
def fresh_logging(monkeypatch):
    root = logging.getLogger()
    before = list(root.handlers)
    monkeypatch.setattr(core_log, "_configured", False)
    yield
    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)
            h.close()
    core_log.setup_logging._path = None  # type: ignore[attr-defined]


def test_setup_logging_custom_filename(tmp_path, fresh_logging):
    path = core_log.setup_logging(logging.INFO, log_dir=tmp_path, console=False, filename=core_log.CLI_LOG_FILE)
    assert path == tmp_path / "cli.log"
    handlers = [h for h in logging.getLogger().handlers if isinstance(h, core_log.SafeRotatingFileHandler)]
    assert handlers and os.path.abspath(handlers[-1].baseFilename) == os.path.abspath(path)
    # a second call returns the configured path unchanged
    assert core_log.setup_logging() == path


def test_setup_logging_default_filename(tmp_path, fresh_logging):
    assert core_log.setup_logging(log_dir=tmp_path, console=False) == tmp_path / "vrphoto.log"
