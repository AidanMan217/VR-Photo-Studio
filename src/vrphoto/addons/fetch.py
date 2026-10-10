"""Downloads and archive extraction for add-on installs (stdlib only).

* :func:`download` - resumable HTTP(S) download (``.part`` file + ``Range``), retries with
  backoff, sha256 verification, byte progress and cooperative cancel.
* :func:`extract` - safe zip / tar extraction (no absolute paths, no ``..``, no links that
  escape), optionally stripping a single top-level folder, written atomically.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import os
import posixpath
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from ..core import fsutil
from ..core.errors import AddonError, Cancelled
from ..core.log import get_logger
from ..core.types import CancelToken

log = get_logger(__name__)

BytesCallback = Callable[[int, int | None], None]
USER_AGENT = "VRPhotoStudio-addons/1 (+https://github.com)"
CHUNK = 1 << 20
MAX_ATTEMPTS = 5


def sha256_file(path: str | os.PathLike[str], cancel: CancelToken | None = None) -> str:
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            if cancel is not None:
                cancel.check()
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(
    url: str,
    dest: str | os.PathLike[str],
    *,
    sha256: str | None = None,
    on_bytes: BytesCallback | None = None,
    cancel: CancelToken | None = None,
    title: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` to ``dest`` (resumable, retried, verified, atomic rename).

    An existing ``dest`` whose sha256 matches is reused without network access.
    Raises :class:`AddonError` on network failure / checksum mismatch and
    :class:`Cancelled` when ``cancel`` fires.
    """
    dest = Path(dest)
    title = title or dest.name
    want = sha256.lower().strip() if sha256 else None
    if dest.is_file():
        if want is None or sha256_file(dest, cancel) == want:
            log.info("Using cached download %s", dest)
            return dest
        log.warning("Cached %s has a wrong checksum; downloading again", dest)
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    # A stale .part (from an older manifest URL or a server that changed the file) can
    # make a resumed download fail its checksum: then start once more from scratch.
    for fresh_retry in (False, True):
        resumed = part.is_file() and part.stat().st_size > 0
        _download_to_part(url, part, title, on_bytes, cancel, headers or {}, timeout)
        if want is None:
            break
        got = sha256_file(part, cancel)
        if got == want:
            break
        with contextlib.suppress(OSError):
            part.unlink()
        if resumed and not fresh_retry:
            log.warning("Resumed download of %s has a wrong checksum; downloading it again from the start", title)
            continue
        raise AddonError(
            f"The download of {title} is corrupted (checksum mismatch).",
            hint="Try again. If it keeps failing, your network (proxy, antivirus) may be altering downloads.",
        )
    replace_with_retry(part, dest)
    return dest


def _download_to_part(
    url: str,
    part: Path,
    title: str,
    on_bytes: BytesCallback | None,
    cancel: CancelToken | None,
    headers: dict[str, str],
    timeout: float,
) -> None:
    attempt = 0
    while True:
        if cancel is not None:
            cancel.check()
        try:
            if _attempt(url, part, on_bytes, cancel, headers, timeout):
                return
        except Cancelled:
            raise
        except _Retryable as e:
            attempt += 1
            if attempt >= MAX_ATTEMPTS:
                raise AddonError(
                    f"Download of {title} failed: {e}",
                    hint="Check your internet connection (and proxy/firewall) and try again; "
                    "the install resumes where it stopped.",
                ) from e
            delay = min(30.0, 2.0 * 2 ** (attempt - 1))
            log.warning("Download of %s interrupted (%s); retry %d/%d in %.0fs", title, e, attempt, MAX_ATTEMPTS - 1, delay)
            _sleep(delay, cancel)


def replace_with_retry(src: str | os.PathLike[str], dst: str | os.PathLike[str], *, attempts: int = 75) -> None:
    """``os.replace`` that waits out transient Windows sharing violations
    (:func:`vrphoto.core.fsutil.replace_with_retry` with a longer default: up to ~18 s).

    Antivirus scanners (Windows Defender) and search indexers briefly open freshly
    written files - especially the ``.exe`` / ``.dll`` files of COLMAP and Brush - and a
    rename of the file or its folder then fails with ``PermissionError`` for a moment.
    """
    fsutil.replace_with_retry(src, dst, attempts=attempts)


class _Retryable(Exception):
    pass


def _attempt(
    url: str,
    part: Path,
    on_bytes: BytesCallback | None,
    cancel: CancelToken | None,
    headers: dict[str, str],
    timeout: float,
) -> bool:
    offset = part.stat().st_size if part.is_file() else 0
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - https URLs from manifests
    except urllib.error.HTTPError as e:
        if e.code == 416 and offset:
            # already complete (or server refuses): restart from scratch once
            part.unlink()
            raise _Retryable("server refused to resume") from e
        if e.code in (408, 429) or 500 <= e.code < 600:
            raise _Retryable(f"HTTP {e.code}") from e
        if e.code in (401, 403):
            raise AddonError(
                f"Access denied while downloading {url} (HTTP {e.code}).",
                hint="The file may require a login/token or your network blocks this site.",
            ) from e
        if e.code == 404:
            raise AddonError(
                f"File not found on the server: {url}",
                hint="The add-on manifest may be outdated; update VR Photo Studio.",
            ) from e
        raise AddonError(f"Download failed (HTTP {e.code}): {url}") from e
    except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as e:
        raise _Retryable(str(getattr(e, "reason", e)) or type(e).__name__) from e
    with resp:
        status = getattr(resp, "status", 200)
        if offset and status != 206:
            offset = 0  # server ignored Range: start over
        length = resp.headers.get("Content-Length")
        total = (int(length) + offset) if length and length.isdigit() else None
        mode = "ab" if offset else "wb"
        got = offset
        try:
            with open(part, mode) as fh:
                while True:
                    if cancel is not None and cancel.cancelled:
                        raise Cancelled("Download cancelled")
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if on_bytes is not None:
                        on_bytes(got, total)
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as e:
            # IncompleteRead (connection closed before Content-Length) is an HTTPException
            raise _Retryable(str(e) or type(e).__name__) from e
        except OSError as e:
            if getattr(e, "errno", None) == 28 or getattr(e, "winerror", None) == 112:
                raise AddonError("The disk is full.", hint="Free some disk space and try again.") from e
            raise _Retryable(str(e)) from e
    if total is not None and got < total:
        raise _Retryable(f"connection closed early ({got} of {total} bytes)")
    return True


def _sleep(seconds: float, cancel: CancelToken | None) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cancel is not None:
            cancel.check()
        time.sleep(min(0.1, max(0.0, end - time.monotonic())))


# --------------------------------------------------------------------------- extraction
_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {f"{d}{i}" for d in ("COM", "LPT") for i in range(1, 10)}


def _bad_part(part: str) -> bool:
    """A path component that is unsafe on Windows or POSIX.

    ``:`` makes ``sub/D:evil`` a drive-relative path on Windows (pathlib joins it as
    ``D:evil`` - outside the target folder) or an NTFS alternate data stream; components
    made only of dots/spaces are normalised to ``..`` / nothing by Win32; device names
    (``NUL``, ``COM1`` ...) are not files.
    """
    if part == ".." or ":" in part or "\x00" in part:
        return True
    if part.strip(". ") == "":
        return True
    return part.split(".")[0].rstrip(" ").upper() in _WIN_RESERVED


def _safe_rel(name: str) -> PurePosixPath | None:
    n = name.replace("\\", "/")
    p = PurePosixPath(n)
    if p.is_absolute() or n.startswith("/") or (len(n) > 1 and n[1] == ":"):
        return None
    parts = [x for x in p.parts if x not in ("", ".")]
    if any(_bad_part(x) for x in parts):
        return None
    return PurePosixPath(*parts) if parts else None


def _inside(root: Path, target: Path, *, resolve: bool = False) -> bool:
    """True if ``target`` is ``root`` or below it (optionally following symlinks)."""
    r = os.path.realpath(root) if resolve else os.path.abspath(root)
    t = os.path.realpath(target) if resolve else os.path.abspath(target)
    r, t = os.path.normcase(r), os.path.normcase(t)
    try:
        return os.path.commonpath([r, t]) == r
    except ValueError:  # different drives on Windows
        return False


def _strip_root(rels: list[PurePosixPath], strip: bool | None) -> str | None:
    """Top-level folder to strip (all members below one folder), or None."""
    if strip is False or not rels:
        return None
    roots = {r.parts[0] for r in rels}
    if len(roots) != 1 or not any(len(r.parts) > 1 for r in rels):
        return None
    return next(iter(roots))


def extract(
    archive: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    strip: bool | None = None,
    cancel: CancelToken | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Extract a ``.zip`` / ``.tar(.gz|.xz|.bz2)`` into ``dest`` (replaced atomically).

    ``strip``: remove a single top-level folder (``None`` = automatically when the
    archive has exactly one top-level folder).  Unsafe member names raise
    :class:`AddonError`.
    """
    archive = Path(archive)
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".extracting")
    if tmp.exists():
        with contextlib.suppress(OSError):
            rmtree(tmp)
    if tmp.exists():  # a leftover that cannot be removed (file in use): never merge into it
        tmp = dest.with_name(f"{dest.name}.extracting-{os.getpid()}-{time.monotonic_ns()}")
    tmp.mkdir(parents=True)
    try:
        name = archive.name.lower()
        if name.endswith(".zip") or zipfile.is_zipfile(archive):
            _extract_zip(archive, tmp, strip, cancel, on_progress)
        elif tarfile.is_tarfile(archive):
            _extract_tar(archive, tmp, strip, cancel, on_progress)
        else:
            raise AddonError(f"Unsupported archive format: {archive.name}")
        if dest.exists():
            rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        replace_with_retry(tmp, dest)
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as e:
        raise AddonError(f"The downloaded archive {archive.name} is damaged: {e}", hint="Retry the install.") from e
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                rmtree(tmp)
    return dest


def _target(root: Path, rel: PurePosixPath, strip_root: str | None) -> Path | None:
    parts = rel.parts
    if strip_root is not None:
        if parts[0] != strip_root:
            return None
        parts = parts[1:]
        if not parts:
            return None
    target = root.joinpath(*parts)
    if not _inside(root, target):  # defence in depth (names are validated by _safe_rel)
        raise AddonError(f"Refusing unsafe path in archive: {rel}")
    return target


def _check_write_target(root: Path, target: Path, archive: Path) -> None:
    """Refuse to write through a symlink that leads outside the extraction folder."""
    if not _inside(root, target.parent, resolve=True) or (target.is_symlink() and not _inside(root, target, resolve=True)):
        raise AddonError(f"Refusing unsafe path in archive {archive.name}: {target.name} (link outside the folder)")


def _extract_zip(
    archive: Path, root: Path, strip: bool | None, cancel: CancelToken | None,
    on_progress: Callable[[float], None] | None,
) -> None:
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        rels: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        for info in infos:
            rel = _safe_rel(info.filename)
            if rel is None:
                if info.filename.strip("/\\.") == "":
                    continue
                raise AddonError(f"Refusing unsafe path in archive {archive.name}: {info.filename}")
            rels.append((info, rel))
        strip_root = _strip_root([r for _, r in rels], strip)
        total = sum(i.file_size for i, _ in rels) or 1
        done = 0
        for info, rel in rels:
            if cancel is not None:
                cancel.check()
            target = _target(root, rel, strip_root)
            if target is None:
                continue
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            _check_write_target(root, target, archive)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            mode = (info.external_attr >> 16) & 0o777
            if mode and os.name != "nt":
                with contextlib.suppress(OSError):
                    os.chmod(target, mode | 0o600)
            done += info.file_size
            if on_progress is not None:
                on_progress(done / total)


def _extract_tar(
    archive: Path, root: Path, strip: bool | None, cancel: CancelToken | None,
    on_progress: Callable[[float], None] | None,
) -> None:
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        rels: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        for m in members:
            rel = _safe_rel(m.name)
            if rel is None:
                if m.name.strip("/\\.") == "":
                    continue
                raise AddonError(f"Refusing unsafe path in archive {archive.name}: {m.name}")
            if (m.issym() and _symlink_dest(rel, m.linkname) is None) or (m.islnk() and _safe_rel(m.linkname) is None):
                log.warning("Skipping link %s -> %s in %s", m.name, m.linkname, archive.name)
                continue
            if m.isdev() or m.isfifo():
                continue
            rels.append((m, rel))
        strip_root = _strip_root([r for _, r in rels], strip)
        total = sum(m.size for m, _ in rels) or 1
        done = 0
        for m, rel in rels:
            if cancel is not None:
                cancel.check()
            target = _target(root, rel, strip_root)
            if target is None:
                continue
            if m.isdir():
                _check_write_target(root, target, archive)
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            _check_write_target(root, target, archive)
            if m.issym():
                # relative link whose destination stays inside the extracted folder;
                # symlinks need a privilege on Windows - skipped there
                link_dest = _symlink_dest(rel, m.linkname)
                if link_dest is None or (strip_root is not None and
                                         (not link_dest.parts or link_dest.parts[0] != strip_root)):
                    log.warning("Skipping link %s -> %s in %s", m.name, m.linkname, archive.name)
                    continue
                with contextlib.suppress(OSError, NotImplementedError):
                    os.symlink(m.linkname.replace("\\", "/"), target)
                continue
            if m.islnk():  # hard link: names are relative to the archive root
                linked_rel = _safe_rel(m.linkname)
                linked = _target(root, linked_rel, strip_root) if linked_rel is not None else None
                if linked is not None and linked.is_file() and _inside(root, linked, resolve=True):
                    with contextlib.suppress(OSError):
                        shutil.copy2(linked, target)
                continue
            fobj = tf.extractfile(m)
            if fobj is None:
                continue
            with fobj, open(target, "wb") as dst:
                shutil.copyfileobj(fobj, dst, CHUNK)
            if os.name != "nt":
                with contextlib.suppress(OSError):
                    os.chmod(target, (m.mode & 0o777) | 0o600)
            done += m.size
            if on_progress is not None:
                on_progress(done / total)


def _symlink_dest(rel: PurePosixPath, linkname: str) -> PurePosixPath | None:
    """Archive-relative destination of a symlink member, or None if it leaves the archive."""
    link = linkname.replace("\\", "/")
    if not link or link.startswith("/") or (len(link) > 1 and link[1] == ":") or "\x00" in link:
        return None
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(rel.as_posix()), link))
    if joined == ".." or joined.startswith("../") or joined.startswith("/"):
        return None
    if any(":" in part for part in joined.split("/")):
        return None
    return PurePosixPath(joined)


def _onerror(func: Callable, path: str, exc_info: object) -> None:
    """rmtree helper: clear the read-only flag (Windows) and retry; waits out short-lived
    sharing violations (antivirus / indexer holding a file that was just written)."""
    for i in range(10 if sys.platform == "win32" else 1):
        try:
            os.chmod(path, 0o700)
            func(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if i == (9 if sys.platform == "win32" else 0):
                raise
            time.sleep(0.1 * (i + 1))


def long_path(path: str | os.PathLike[str]) -> str:
    """On Windows, the ``\\\\?\\`` form of an absolute path so file APIs accept paths longer
    than 260 characters (deep ``site-packages`` trees of CUDA wheels); unchanged elsewhere."""
    p = os.fspath(path)
    if sys.platform != "win32" or p.startswith("\\\\?\\"):
        return p
    p = os.path.abspath(p)
    if p.startswith("\\\\"):  # UNC share
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


def rmtree(path: str | os.PathLike[str]) -> None:
    """Remove a directory tree, coping with read-only files and long paths on Windows."""
    p = Path(path)
    if p.is_symlink() or p.is_file():
        p.unlink()
        return
    if not p.exists():
        return
    target = long_path(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(target, onexc=_onerror)
    else:  # pragma: no cover - Python 3.11
        shutil.rmtree(target, onerror=_onerror)


def dir_size(path: str | os.PathLike[str]) -> int:
    """Total size in bytes of all files below ``path`` (0 if missing)."""
    total = 0
    for root, _dirs, files in os.walk(long_path(path)):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(root, f)).st_size
    return total


__all__ = ["dir_size", "download", "extract", "long_path", "replace_with_retry", "rmtree", "sha256_file"]
