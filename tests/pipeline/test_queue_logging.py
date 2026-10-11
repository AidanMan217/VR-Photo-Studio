"""How JobQueue logs failed jobs: expected user errors stay readable in the Log panel."""

from __future__ import annotations

import logging

import pytest

from vrphoto.core.errors import AddonError, InputError, ModelUnavailable, VRPhotoError
from vrphoto.pipeline.jobs import StereoJob
from vrphoto.pipeline.queue import JobQueue

TIMEOUT = 10.0
LOGGER = "vrphoto.pipeline.queue"


def _run_failing(exc: BaseException, caplog) -> tuple[str, list[logging.LogRecord]]:
    def runner(sp, ctx, *, progress, cancel):
        raise exc

    jq = JobQueue(runner, ctx=object())
    try:
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            jid = jq.submit(StereoJob(inputs=["a.jpg"]))
            assert jq.wait(jid, TIMEOUT)
    finally:
        jq.shutdown(wait=True)
    assert jq.get(jid).status == "failed"
    return jid, [r for r in caplog.records if r.name == LOGGER]


@pytest.mark.parametrize(
    "exc",
    [
        InputError("Capture needs at least 3 photos.", hint="Add more photos of the object."),
        AddonError("The TRELLIS add-on is not installed."),
        ModelUnavailable("Model download failed."),
        VRPhotoError("Something the user can fix."),
    ],
    ids=lambda e: type(e).__name__,
)
def test_expected_error_is_one_error_line_with_traceback_at_debug(exc, caplog):
    jid, records = _run_failing(exc, caplog)
    errors = [r for r in records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    err = errors[0]
    assert err.levelno == logging.ERROR
    expected = str(exc) + (f" {exc.hint}" if exc.hint else "")
    assert err.getMessage() == f"Job {jid} failed: {expected}"
    assert not err.exc_info and not err.exc_text  # no traceback in the Log panel
    debug_tb = [r for r in records if r.levelno == logging.DEBUG and r.exc_info]
    assert len(debug_tb) == 1
    assert debug_tb[0].exc_info[1] is exc and jid in debug_tb[0].getMessage()


def test_expected_error_traceback_hidden_at_info_level(caplog):
    """With the usual INFO level (GUI Log panel), only the one readable line shows."""
    exc = InputError("Capture needs at least 3 photos.")

    def runner(sp, ctx, *, progress, cancel):
        raise exc

    jq = JobQueue(runner, ctx=object())
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            jid = jq.submit(StereoJob(inputs=["a.jpg"]))
            assert jq.wait(jid, TIMEOUT)
    finally:
        jq.shutdown(wait=True)
    records = [r for r in caplog.records if r.name == LOGGER]
    assert not any(r.exc_info for r in records)
    assert [r.getMessage() for r in records if r.levelno >= logging.WARNING] == [
        f"Job {jid} failed: Capture needs at least 3 photos."
    ]
    assert "Traceback" not in caplog.text


def test_unexpected_error_keeps_full_traceback_at_error(caplog):
    exc = ZeroDivisionError("boom")
    jid, records = _run_failing(exc, caplog)
    errors = [r for r in records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    err = errors[0]
    assert err.levelno == logging.ERROR and jid in err.getMessage()
    assert err.exc_info and err.exc_info[1] is exc
    assert "Traceback" in caplog.text and "ZeroDivisionError: boom" in caplog.text
