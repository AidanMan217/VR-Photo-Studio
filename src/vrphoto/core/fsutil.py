"""Small Windows-safe filesystem helpers shared by every module.

On Windows a rename onto (or of) a file fails with ``PermissionError`` (WinError 5/32)
while another process has that file open without ``FILE_SHARE_DELETE``: Windows
Defender and the Search indexer briefly open freshly written files, the viewer server
may be streaming the old copy, a cached ONNX session keeps its file mapped.  These
locks are released within a moment, so :func:`replace_with_retry` simply waits them out.
Everywhere else (and for any other error) it behaves exactly like :func:`os.replace`.
"""

from __future__ import annotations

import os
import secrets
import sys
import time
from pathlib import Path

__all__ = ["atomic_write_bytes", "replace_with_retry", "temp_sibling"]

DEFAULT_ATTEMPTS = 20
_MIN_DELAY = 0.05
_MAX_DELAY = 0.25


def _delay(attempt: int) -> float:
    """Backoff before retry number ``attempt`` (0-based): 0.05, 0.10, ... capped at 0.25 s."""
    return min(_MAX_DELAY, _MIN_DELAY * (attempt + 1))


def replace_with_retry(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
) -> None:
    """:func:`os.replace` that retries transient Windows sharing violations.

    Only ``PermissionError`` on win32 is retried (``attempts`` tries in total, 0.05-0.25 s
    apart, i.e. about 4 s with the default); every other error - and any error on other
    platforms - is raised immediately.
    """
    n = max(1, int(attempts))
    for i in range(n):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if sys.platform != "win32" or i >= n - 1:
                raise
            time.sleep(_delay(i))


def temp_sibling(path: str | os.PathLike[str], suffix: str = ".tmp") -> Path:
    """A unique hidden temp path next to ``path`` (same volume, so the final rename is atomic)."""
    p = Path(path)
    return p.with_name(f".{p.name}.{os.getpid()}.{secrets.token_hex(4)}{suffix}")


def atomic_write_bytes(
    path: str | os.PathLike[str],
    data: bytes | bytearray | memoryview,
    *,
    fsync: bool = True,
    attempts: int = DEFAULT_ATTEMPTS,
) -> None:
    """Write ``data`` to ``path`` atomically: readers see the old file or the new one, never a mix.

    The bytes go to a unique sibling temp file through Python's buffered file I/O (which
    loops on short writes and raises ``OSError`` - e.g. ENOSPC on a full disk - instead of
    silently truncating), are optionally flushed to disk, and the temp file then replaces
    ``path`` via :func:`replace_with_retry`.  On any failure the temp file is removed and
    an existing ``path`` is left untouched.  The parent folder must exist.
    """
    p = Path(path)
    tmp = temp_sibling(p)
    try:
        with open(tmp, "xb") as fh:
            fh.write(data)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        replace_with_retry(tmp, p, attempts=attempts)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
