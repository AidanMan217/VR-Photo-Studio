"""Resumable, verified model downloads.

Files are streamed to ``<models_dir>/<id>/<repo path>.part`` with HTTP ``Range`` resume
(guarded by ``If-Range`` + a small sidecar so a changed upstream file is never spliced),
retried with exponential backoff on transient errors, verified against the pinned
checksum (sha256, or md5 for upstreams that only publish md5) and atomically renamed into
place.  Unknown checksums are computed and recorded in ``<models_dir>/recorded_hashes.json``
with a logged warning.

Only the standard library is used (``urllib``): it honours ``HTTPS_PROXY`` / system proxy
settings and, on Windows, the default SSL context loads the system certificate store.

Environment override: ``VRPHOTO_MODEL_<ID>`` (e.g. ``VRPHOTO_MODEL_DA2_SMALL``) pointing at
an existing file bypasses the download entirely (tests, offline installs).
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import http.client
import json
import os
import shutil
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from .. import __version__
from ..core.config import get_settings
from ..core.errors import UI_SETTINGS_MODELS, Cancelled, DownloadError, ModelUnavailable
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter
from .registry import ModelSpec, env_var_for, file_url, get_spec, hf_endpoint

log = get_logger(__name__)

CHUNK_SIZE = 1 << 20
TIMEOUT_S = 30.0
MAX_ATTEMPTS = 5  # consecutive attempts without progress before giving up
OFFLINE_ATTEMPTS = 3  # same, when the network is unreachable (DNS failure, no route)
BACKOFF_BASE_S = 1.0  # tests set this to 0
BACKOFF_MAX_S = 30.0
DISK_MARGIN_BYTES = 64 << 20
PROGRESS_INTERVAL_S = 0.2  # progress callback throttle
RECORDED_HASHES = "recorded_hashes.json"
USER_AGENT = f"VRPhotoStudio/{__version__} (+https://github.com/AidanMan217/VR-Photo-Studio)"

BytesCallback = Callable[[int, "int | None"], None]

_locks_guard = threading.Lock()
_model_locks: dict[str, threading.Lock] = {}
_record_lock = threading.Lock()


# --------------------------------------------------------------------------- paths
def model_dir(model_id: str) -> Path:
    """Directory holding the files of ``model_id`` (not created)."""
    get_spec(model_id)  # validates the id
    return get_settings().models_dir / model_id


def env_override(model_id: str) -> Path | None:
    """Path from ``VRPHOTO_MODEL_<ID>`` if set.  Raises if it points to a missing file."""
    var = env_var_for(model_id)
    val = os.environ.get(var, "").strip().strip('"').strip()  # Windows users often quote paths
    if not val:
        return None
    p = Path(val).expanduser()
    if not p.is_file():
        raise ModelUnavailable(
            f"{var} points to a file that does not exist: {p}",
            hint=f"Fix or unset the {var} environment variable.",
        )
    return p


def variant_files(spec: ModelSpec, variant: str = "default") -> tuple[str, ...]:
    """Repo-relative files making up ``variant`` ("default" or "fp16") of a model."""
    fp16_extra: tuple[str, ...] = ()
    if spec.fp16_file:  # external-data files of the fp16 graph (``model_fp16.onnx_data``)
        stem = PurePosixPath(spec.fp16_file).name
        fp16_extra = tuple(f for f in spec.files[1:] if PurePosixPath(f).name.startswith(stem))
    if variant in ("default", "fp32", ""):
        return tuple(f for f in spec.files if f not in fp16_extra)
    if variant == "fp16":
        if not spec.fp16_file:
            raise ModelUnavailable(f"{spec.title} has no half-precision (fp16) variant")
        return (spec.fp16_file, *fp16_extra)
    raise ModelUnavailable(f"Unknown model variant '{variant}'")


def _local(spec: ModelSpec, rel: str) -> Path:
    """Local path of repo file ``rel``; refuses anything that could escape the model dir.

    Registry/lock paths are POSIX-style and relative.  Besides ``..`` and absolute paths
    this rejects Windows drive letters / ADS (``C:x``, ``f:stream``), backslashes and
    reserved trailing dots/spaces, which ``PurePosixPath`` would otherwise let through
    and ``WindowsPath.joinpath`` would interpret.
    """
    p = PurePosixPath(rel)
    parts = p.parts
    bad = (
        not parts
        or p.is_absolute()
        or any(
            part in ("", ".", "..") or "\\" in part or ":" in part or "\x00" in part or part[-1] in ". "
            for part in parts
        )
    )
    if bad:
        raise ModelUnavailable(f"Invalid file path '{rel}' in model {spec.id}")
    base = model_dir(spec.id)
    out = base.joinpath(*parts)
    try:
        out.resolve().relative_to(base.resolve())
    except (ValueError, OSError) as e:  # a symlinked sub-folder pointing elsewhere
        raise ModelUnavailable(f"Invalid file path '{rel}' in model {spec.id}") from e
    return out


def local_path(model_id: str, variant: str = "default") -> Path:
    """Where the entry-point file of ``model_id`` lives once downloaded."""
    ov = env_override(model_id)
    if ov is not None:
        return ov
    spec = get_spec(model_id)
    return _local(spec, variant_files(spec, variant)[0])


def _file_ok(spec: ModelSpec, rel: str) -> bool:
    p = _local(spec, rel)
    if not p.is_file():
        return False
    want = spec.sizes.get(rel)
    return want is None or p.stat().st_size == want


def is_downloaded(model_id: str, variant: str = "default") -> bool:
    """True if the model is available locally (env override or all files present)."""
    try:
        if env_override(model_id) is not None:
            return True
        spec = get_spec(model_id)
        return all(_file_ok(spec, f) for f in variant_files(spec, variant))
    except ModelUnavailable:
        return False


def downloaded_size_mb(model_id: str) -> float:
    """Disk usage of the model's directory in MB (0 if absent)."""
    d = model_dir(model_id)
    if not d.is_dir():
        return 0.0
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e6


# --------------------------------------------------------------------------- public API
def ensure_model(
    model_id: str,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    token: str | None = None,
    variant: str = "default",
) -> Path:
    """Make sure ``model_id`` is on disk and return the path of its ONNX entry point.

    Downloads missing files (resumable, verified).  ``token`` is a Hugging Face access
    token for gated repos (defaults to ``settings.hf_token`` / ``$HF_TOKEN``); it is only
    ever sent to the Hugging Face host.  Raises :class:`DownloadError` on failure and
    :class:`Cancelled` if ``cancel`` fires (partial files are kept for resuming).
    """
    ov = env_override(model_id)
    if ov is not None:
        return ov
    spec = get_spec(model_id)
    files = variant_files(spec, variant)
    entry = _local(spec, files[0])
    if all(_file_ok(spec, f) for f in files):
        return entry

    if token is None:
        token = get_settings().hf_token or os.environ.get("HF_TOKEN") or None
    rep = progress or ProgressReporter.null()
    with _model_lock(model_id, cancel):
        d = model_dir(model_id)
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise _os_error(e, spec.title, d / ".download.lock") from e
        waiting = f"Waiting for another download of {spec.title} to finish"
        with _interprocess_lock(d / ".download.lock", cancel, on_wait=lambda: rep(0.0, waiting)):
            todo = [f for f in files if not _file_ok(spec, f)]
            if not todo:
                return entry
            _check_disk_space(spec, todo)
            totals = {f: spec.sizes.get(f) for f in todo}
            agg = _Aggregate(spec.title, totals, spec.size_mb, rep)
            for f in todo:
                if cancel is not None:
                    cancel.check()
                url = file_url(spec, f)
                auth = token if (spec.is_hf and token) else None
                download_url(
                    url,
                    _local(spec, f),
                    checksum=spec.sha256.get(f) or None,
                    size=spec.sizes.get(f),
                    token=auth,
                    on_bytes=lambda done, total, _f=f: agg.update(_f, done, total),
                    cancel=cancel,
                    title=spec.title,
                    record_key=f"{spec.id}/{f}",
                    revision=spec.revision,
                )
            rep(1.0, f"Downloaded {spec.title}")
    log.info("Model %s ready at %s", model_id, entry)
    return entry


def delete_model(model_id: str) -> None:
    """Remove a downloaded model (all variants, partial files and recorded hashes).

    Files referenced by an env override are never deleted.  Cached ONNX sessions for the
    model are dropped first so Windows can delete the (otherwise mapped) files.
    """
    get_spec(model_id)
    ort_mod = sys.modules.get("vrphoto.models.ort")
    if ort_mod is not None:
        with contextlib.suppress(Exception):
            ort_mod.unload_model(model_id)
    d = model_dir(model_id)
    with _model_lock(model_id):
        if d.exists():
            try:
                _rmtree(d)
            except OSError as e:
                raise ModelUnavailable(
                    f"Could not delete model files in {d}: {e}",
                    hint="Close any running jobs that use this model and try again.",
                ) from e
        _forget_recorded(model_id)
    log.info("Deleted model %s", model_id)


def download_url(
    url: str,
    dest: str | Path,
    *,
    checksum: str | None = None,
    size: int | None = None,
    token: str | None = None,
    on_bytes: BytesCallback | None = None,
    cancel: CancelToken | None = None,
    title: str | None = None,
    record_key: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download ``url`` to ``dest`` (resumable, retried, verified, atomic).

    ``checksum``: hex sha256 or ``"md5:<hex>"``; when None the sha256 is computed and
    recorded (if ``record_key``) with a warning.  ``size``: expected byte count if known.
    ``token``: bearer token, dropped on redirects to other hosts.  ``on_bytes(done,
    total)`` reports progress.  Returns ``dest``.
    """
    dest = Path(dest)
    title = title or dest.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    meta_path = dest.with_name(dest.name + ".part.json")
    algo, want = _parse_checksum(checksum)

    state = _PartState(part, meta_path, url)
    hasher: _Hasher = _Hasher(algo)
    attempts = 0
    restarted_416 = False
    while True:
        if cancel is not None:
            cancel.check()
        offset = state.prepare()
        if offset == 0:
            hasher.reset()
        elif hasher.count != offset:
            hasher.invalidate()  # resuming bytes we did not hash: re-hash the file at the end
        before = offset
        try:
            complete = _attempt(url, state, offset, token, hasher, on_bytes, cancel, size, title)
        except _Transient as e:
            if state.size() > before:
                attempts = 0  # progress was made: do not count against the budget
            attempts += 1
            # no route / DNS failure: the user is offline -> fail fast (auto falls back)
            budget = OFFLINE_ATTEMPTS if _is_offline(e.cause) else MAX_ATTEMPTS
            if attempts >= budget:
                raise _final_network_error(title, url, e.cause) from e.cause
            delay = e.retry_after if e.retry_after is not None else _backoff(attempts)
            log.warning(
                "Download of %s interrupted (%s); retry %d/%d in %.1fs",
                title,
                e.cause,
                attempts,
                budget - 1,
                delay,
            )
            _sleep(delay, cancel)
            continue
        except _AccessDenied as e:
            if e.code == 401 and token:
                # HF rejects invalid/expired tokens even for public repos: retry anonymously
                log.warning("Access token rejected while downloading %s; retrying without it", title)
                token = None
                continue
            raise
        except _RangeNotSatisfiable:
            have = state.size()
            expected = size if size is not None else state.total
            if expected is not None and have == expected:
                complete = True
            elif restarted_416:
                raise DownloadError(f"Server refused to resume the download of {title}.") from None
            else:
                restarted_416 = True
                state.discard()
                hasher = _Hasher(algo)
                continue
        if complete:
            break

    got = state.size()
    if size is not None and got != size:
        state.discard()
        raise DownloadError(
            f"Downloaded {title} has the wrong size ({got} bytes, expected {size}).",
            hint="Try again; if this keeps happening the model registry may be outdated.",
        )
    digest = hasher.hexdigest() if hasher.valid else _hash_file(part, algo, cancel)
    if want is not None:
        if digest != want:
            state.discard()
            raise DownloadError(
                f"The downloaded file for {title} is corrupted (checksum mismatch).",
                hint="Try the download again. If it keeps failing, your network may be altering downloads.",
            )
    else:
        sha = digest if algo == "sha256" else _hash_file(part, "sha256", cancel)
        log.warning("No pinned checksum for %s; recorded sha256 %s", title, sha)
        if record_key:
            _record_hash(record_key, sha, got, url, revision)
    try:
        _replace_retry(part, dest)
    except OSError as e:
        raise _os_error(e, title, dest) from e
    with contextlib.suppress(OSError):
        meta_path.unlink()
    return dest


# --------------------------------------------------------------------------- one HTTP attempt
CANCEL_POLL_S = 0.2


class _CancelWatch:
    """Makes cancel take effect while a socket read is blocked on a stalled connection.

    Without it a dead connection would only notice the cancel after the socket timeout
    (:data:`TIMEOUT_S`).  A daemon thread polls the token and, when it fires, shuts the
    response's socket down, which wakes the blocked ``recv`` with an error.
    """

    def __init__(self, cancel: CancelToken | None, resp: object) -> None:
        self._cancel = cancel
        self._resp = resp
        self._done = threading.Event()
        self.fired = False
        if cancel is not None:
            threading.Thread(target=self._watch, name="vrphoto-download-cancel", daemon=True).start()

    def _watch(self) -> None:
        assert self._cancel is not None
        while not self._done.wait(CANCEL_POLL_S):
            if self._cancel.cancelled:
                self.fired = True
                _abort_response(self._resp)
                return

    def __enter__(self) -> _CancelWatch:
        return self

    def __exit__(self, *exc: object) -> None:
        self._done.set()


def _abort_response(resp: object) -> None:
    """Best-effort: shut down the socket under an ``http.client.HTTPResponse``."""
    sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        if sys.platform == "win32":
            # Winsock does not wake a recv() blocked in another thread on shutdown(); closing
            # the handle does.  socket.close() would be deferred while the response's makefile
            # still references the socket, so close the underlying handle directly.
            real_close = getattr(sock, "_real_close", None)
            with contextlib.suppress(OSError):
                if callable(real_close):
                    real_close()
                else:
                    sock.close()


class _Transient(Exception):
    def __init__(self, cause: BaseException, retry_after: float | None = None):
        super().__init__(str(cause))
        self.cause = cause
        self.retry_after = retry_after


class _RangeNotSatisfiable(Exception):
    pass


class _AccessDenied(DownloadError):
    """HTTP 401/403 (a friendly DownloadError; retried once without a token on 401)."""

    def __init__(self, message: str, *, hint: str | None, code: int):
        super().__init__(message, hint=hint)
        self.code = code


def _attempt(
    url: str,
    state: _PartState,
    offset: int,
    token: str | None,
    hasher: _Hasher,
    on_bytes: BytesCallback | None,
    cancel: CancelToken | None,
    size: int | None,
    title: str,
) -> bool:
    """One GET; appends to the part file.  Returns True when the body was complete."""
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
        if state.etag:
            headers["If-Range"] = state.etag
    req = urllib.request.Request(url, headers=headers)
    opener = _build_opener()
    try:
        resp = opener.open(req, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as e:
        _raise_http(e, title, url)
        raise  # unreachable
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        if cancel is not None and cancel.cancelled:
            raise Cancelled(f"Download of {title} cancelled") from e
        _raise_connect(e, title, url)
        raise  # unreachable

    with resp, _CancelWatch(cancel, resp):
        status = getattr(resp, "status", 200)
        if "text/html" in (resp.headers.get("Content-Type") or "").lower():
            # a captive portal (hotel / airport Wi-Fi) or a blocking proxy answered instead
            raise DownloadError(
                f"Downloading {title} returned a web page instead of the model file.",
                hint="A Wi-Fi login page or a company proxy may be blocking the download: open a "
                "web browser, complete any sign-in, then try again.",
            )
        total: int | None
        clen = _int_or_none(resp.headers.get("Content-Length"))
        if offset > 0 and status == 206:
            start, full = _parse_content_range(resp.headers.get("Content-Range"))
            if start != offset:
                log.warning("Server resumed %s at %s instead of %d; restarting", title, start, offset)
                state.discard()
                raise _Transient(ConnectionError("inconsistent Content-Range"), retry_after=0.0)
            total = full if full is not None else (offset + clen if clen is not None else size)
            mode = "ab"
        else:
            if offset > 0:
                log.info("Server ignored the resume request for %s; downloading from the start", title)
                hasher.reset()
            offset = 0
            total = clen if clen is not None else size
            mode = "wb"
        if size is not None and total is not None and total != size:
            log.warning("Server reports %d bytes for %s, registry expects %d", total, title, size)
        state.start(resp.headers.get("ETag"), total)
        done = offset
        if on_bytes:
            on_bytes(done, total)
        try:
            with open(state.part, mode) as fh:
                while True:
                    if cancel is not None and cancel.cancelled:
                        raise Cancelled(f"Download of {title} cancelled")
                    try:
                        chunk = resp.read(CHUNK_SIZE)
                    except (OSError, http.client.HTTPException, ValueError) as e:
                        # ValueError: read on a response the cancel watcher closed
                        if cancel is not None and cancel.cancelled:
                            raise Cancelled(f"Download of {title} cancelled") from e
                        raise _Transient(e) from e
                    if not chunk:
                        if cancel is not None and cancel.cancelled:
                            raise Cancelled(f"Download of {title} cancelled")
                        break
                    try:
                        fh.write(chunk)
                    except OSError as e:
                        raise _os_error(e, title, state.part) from e
                    hasher.update(chunk)
                    done += len(chunk)
                    if on_bytes:
                        on_bytes(done, total)
        except OSError as e:  # opening / flushing the part file
            raise _os_error(e, title, state.part) from e
    if total is not None and done < total:
        raise _Transient(http.client.IncompleteRead(b"", total - done))
    return True


def _raise_http(e: urllib.error.HTTPError, title: str, url: str) -> None:
    code = e.code
    host = urlparse(url).hostname or url
    with contextlib.suppress(Exception):
        e.close()
    if code == 416:
        raise _RangeNotSatisfiable() from e
    if code in (401, 403):
        is_hf = host == (urlparse(hf_endpoint()).hostname or "")
        hint = (
            "This model may require a free Hugging Face account: open the model page, accept "
            f"its licence, then paste an access token in {UI_SETTINGS_MODELS}. (Hugging Face also "
            "answers this way when a model has been moved or deleted - then check for an app update.)"
            if is_hf
            else "The server refused access; the download location may have changed - check for an app update."
        )
        raise _AccessDenied(
            f"Access to {title} was denied by {host} (HTTP {code}).", hint=hint, code=code
        ) from e
    if code == 404:
        raise DownloadError(
            f"{title} was not found on {host} (HTTP 404).",
            hint="The model may have moved; check for an app update.",
        ) from e
    if code in (408, 425, 429) or 500 <= code < 600:
        raise _Transient(e, retry_after=_retry_after(e)) from e
    raise DownloadError(f"Download of {title} failed: HTTP {code} {e.reason}") from e


def _raise_connect(e: BaseException, title: str, url: str) -> None:
    reason = getattr(e, "reason", e)
    if isinstance(reason, ssl.SSLCertVerificationError) or isinstance(e, ssl.SSLCertVerificationError):
        raise DownloadError(
            f"Secure connection to {urlparse(url).hostname} could not be verified while downloading {title}.",
            hint="A proxy or antivirus may be intercepting HTTPS. Check your network settings.",
        ) from e
    if isinstance(reason, OSError) and _is_disk_full(reason):
        raise _os_error(reason, title, None) from e
    raise _Transient(e if not isinstance(reason, BaseException) else reason) from e


def _final_network_error(title: str, url: str, cause: BaseException) -> DownloadError:
    host = urlparse(url).hostname or url
    if isinstance(cause, urllib.error.HTTPError):
        return DownloadError(
            f"Download of {title} failed: {host} answered HTTP {cause.code} repeatedly.",
            hint="The server may be overloaded; try again later.",
        )
    if _is_offline(cause):
        return DownloadError(
            f"Cannot reach {host} to download {title}. You appear to be offline.",
            hint="Check your internet connection (or proxy settings) and try again.",
        )
    return DownloadError(
        f"Download of {title} from {host} kept failing: {cause}",
        hint="Check your internet connection and try again; the download will resume where it stopped.",
    )


_OFFLINE_ERRNOS = {
    getattr(errno, "ENETUNREACH", -1),
    getattr(errno, "EHOSTUNREACH", -1),
    10051,  # WSAENETUNREACH
    10065,  # WSAEHOSTUNREACH
    11001,  # WSAHOST_NOT_FOUND
    11002,  # WSATRY_AGAIN
}


def _is_offline(cause: BaseException) -> bool:
    """True for errors meaning "no network at all" (DNS failure, no route, refused)."""
    if isinstance(cause, (socket.gaierror, ConnectionRefusedError)):
        return True
    if isinstance(cause, OSError) and (
        cause.errno in _OFFLINE_ERRNOS or getattr(cause, "winerror", None) in _OFFLINE_ERRNOS
    ):
        return True
    return "getaddrinfo" in str(cause)


def _os_error(e: OSError, title: str, path: Path | None) -> DownloadError:
    if _is_disk_full(e):
        where = f" on the drive holding {path.parent}" if path is not None else ""
        return DownloadError(
            f"Not enough disk space to download {title}.",
            hint=f"Free some space{where} and try again; the download will resume.",
        )
    if isinstance(e, PermissionError):
        return DownloadError(
            f"Cannot write model file {path}: permission denied.",
            hint="Another program (antivirus?) may be locking the file, or the folder is read-only.",
        )
    return DownloadError(f"Cannot write model file {path}: {e}")


def _is_disk_full(e: OSError) -> bool:
    return e.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)) or getattr(e, "winerror", None) in (
        39,
        112,
    )


def _retry_after(e: urllib.error.HTTPError) -> float | None:
    v = e.headers.get("Retry-After") if e.headers else None
    try:
        return min(60.0, max(0.0, float(v))) if v is not None else None
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    return min(BACKOFF_MAX_S, BACKOFF_BASE_S * (2 ** (attempt - 1)))


def _sleep(seconds: float, cancel: CancelToken | None) -> None:
    end = time.monotonic() + seconds
    while True:
        if cancel is not None:
            cancel.check()
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(0.1, left))


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop the Authorization header when a redirect leaves the original host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlparse(newurl).hostname != urlparse(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_RedirectHandler())


def _int_or_none(v: str | None) -> int | None:
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def _parse_content_range(v: str | None) -> tuple[int | None, int | None]:
    """``bytes 100-199/1000`` -> (100, 1000); unknown parts -> None."""
    if not v or not v.startswith("bytes "):
        return None, None
    rng, _, full = v[6:].partition("/")
    start = _int_or_none(rng.partition("-")[0])
    return start, _int_or_none(full) if full != "*" else None


# --------------------------------------------------------------------------- part bookkeeping
class _PartState:
    """The ``.part`` file plus a JSON sidecar recording URL / ETag / size it belongs to."""

    def __init__(self, part: Path, meta: Path, url: str):
        self.part = part
        self.meta = meta
        self.url = url
        self.etag: str | None = None
        self.total: int | None = None  # full size reported by the server for this part

    def size(self) -> int:
        try:
            return self.part.stat().st_size
        except OSError:
            return 0

    def prepare(self) -> int:
        """Return the resume offset (0 = fresh), discarding a part from another URL."""
        if not self.part.exists():
            self.etag = None
            return 0
        try:
            meta = json.loads(self.meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
        if not isinstance(meta, dict) or meta.get("url") != self.url:
            log.info("Discarding stale partial download %s", self.part.name)
            self.discard()
            return 0
        self.etag = meta.get("etag") or None
        total = meta.get("size")
        self.total = total if isinstance(total, int) and total > 0 else None
        return self.size()

    def start(self, etag: str | None, total: int | None) -> None:
        if etag and etag.startswith("W/"):
            etag = None  # weak validators are not allowed in If-Range
        self.etag = etag or self.etag
        self.total = total if total is not None else self.total
        tmp = self.meta.with_suffix(".tmp")
        with contextlib.suppress(OSError):
            tmp.write_text(json.dumps({"url": self.url, "etag": self.etag, "size": total}), encoding="utf-8")
            os.replace(tmp, self.meta)

    def discard(self) -> None:
        for p in (self.part, self.meta):
            with contextlib.suppress(OSError):
                p.unlink()
        self.etag = None
        self.total = None


class _Hasher:
    def __init__(self, algo: str):
        self.algo = algo
        self._h = hashlib.new(algo)
        self.count = 0
        self.valid = True

    def update(self, b: bytes) -> None:
        if self.valid:
            self._h.update(b)
        self.count += len(b)

    def reset(self) -> None:
        self._h = hashlib.new(self.algo)
        self.count = 0
        self.valid = True

    def invalidate(self) -> None:
        self.valid = False

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def _parse_checksum(checksum: str | None) -> tuple[str, str | None]:
    if not checksum:
        return "sha256", None
    c = checksum.strip().lower()
    if ":" in c:
        algo, _, hexd = c.partition(":")
        if algo not in ("sha256", "md5", "sha1"):
            raise DownloadError(f"Unsupported checksum algorithm '{algo}'")
        return algo, hexd
    return "sha256", c


def _hash_file(path: Path, algo: str, cancel: CancelToken | None) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK_SIZE * 4), b""):
            if cancel is not None:
                cancel.check()
            h.update(block)
    return h.hexdigest()


def _record_hash(key: str, sha: str, size: int, url: str, revision: str | None) -> None:
    path = get_settings().models_dir / RECORDED_HASHES
    with _record_lock:
        data = _read_recorded(path)
        data[key] = {
            "sha256": sha,
            "size": size,
            "url": url,
            "revision": revision,
            "recorded": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        _write_recorded(path, data)


def _forget_recorded(model_id: str) -> None:
    path = get_settings().models_dir / RECORDED_HASHES
    with _record_lock:
        data = _read_recorded(path)
        keep = {k: v for k, v in data.items() if not k.startswith(model_id + "/")}
        if len(keep) != len(data):
            _write_recorded(path, keep)


def recorded_hashes() -> dict[str, dict]:
    """Checksums recorded for downloads that had no pinned hash (``"<id>/<file>"`` keys)."""
    with _record_lock:
        return _read_recorded(get_settings().models_dir / RECORDED_HASHES)


def _read_recorded(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_recorded(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        log.warning("Cannot write %s: %s", path, e)


# --------------------------------------------------------------------------- progress / disk / locks
class _Aggregate:
    """Combines per-file byte progress into one throttled ProgressReporter stream."""

    def __init__(self, title: str, totals: dict[str, int | None], size_mb: float, rep: ProgressReporter):
        self.title = title
        self.totals = dict(totals)
        self.done: dict[str, int] = dict.fromkeys(totals, 0)
        self.fallback_total = int(size_mb * 1e6) if size_mb else None
        self.rep = rep
        self._last = 0.0
        self._t0 = time.monotonic()
        self._start_bytes: int | None = None

    def update(self, f: str, done: int, total: int | None) -> None:
        self.done[f] = done
        if total is not None:
            self.totals[f] = total
        now = time.monotonic()
        all_done = sum(self.done.values())
        if self._start_bytes is None:
            self._start_bytes = all_done
        known = [t for t in self.totals.values() if t]
        grand = sum(known) if len(known) == len(self.totals) else self.fallback_total
        finished = grand is not None and all_done >= grand
        if now - self._last < PROGRESS_INTERVAL_S and not finished:
            return
        self._last = now
        speed = (all_done - self._start_bytes) / max(1e-3, now - self._t0)
        if grand:
            msg = f"Downloading {self.title}: {all_done / 1e6:.0f} / {grand / 1e6:.0f} MB"
            frac = min(1.0, all_done / grand)
        else:
            msg = f"Downloading {self.title}: {all_done / 1e6:.0f} MB"
            frac = 0.0
        if speed > 0 and now - self._t0 > 1.0:
            msg += f" ({speed / 1e6:.1f} MB/s)"
        self.rep(frac, msg)


def _check_disk_space(spec: ModelSpec, todo: list[str]) -> None:
    need = 0
    for f in todo:
        expected = spec.sizes.get(f)
        if expected is None:
            expected = int(spec.size_mb * 1e6 / max(1, len(spec.files)))
        part = _local(spec, f)
        part = part.with_name(part.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        need += max(0, expected - have)
    d = model_dir(spec.id)
    try:
        free = shutil.disk_usage(d).free
    except OSError:
        return
    if free < need + DISK_MARGIN_BYTES:
        raise DownloadError(
            f"Not enough disk space to download {spec.title} ({need / 1e9:.2f} GB needed, "
            f"{free / 1e9:.2f} GB free).",
            hint=f"Free some space on the drive holding {d} and try again.",
        )


@contextlib.contextmanager
def _model_lock(model_id: str, cancel: CancelToken | None = None) -> Iterator[None]:
    """Per-model in-process lock; waiting for another thread's download honours ``cancel``."""
    with _locks_guard:
        lk = _model_locks.setdefault(model_id, threading.Lock())
    while not lk.acquire(timeout=0.2):
        if cancel is not None:
            cancel.check()
    try:
        yield
    finally:
        lk.release()


@contextlib.contextmanager
def _interprocess_lock(
    path: Path, cancel: CancelToken | None, on_wait: Callable[[], None] | None = None
) -> Iterator[None]:
    """Exclusive lock on ``path`` shared with other app processes (GUI + CLI).

    ``on_wait`` is called once if another process holds the lock (progress message).
    """
    try:
        fh = open(path, "a+b")  # noqa: SIM115 - closed in finally
    except OSError as e:
        raise _os_error(e, path.parent.name, path) from e
    locked = False
    warned = False
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError:
                if not warned:
                    log.info("Waiting for another process downloading into %s", path.parent)
                    warned = True
                    if on_wait is not None:
                        on_wait()
                _sleep(0.25, cancel)
        yield
    finally:
        if locked:
            with contextlib.suppress(OSError):
                if os.name == "nt":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


REPLACE_RETRIES = 20  # x 0.25 s: antivirus / Search indexer briefly lock fresh files on Windows


def _replace_retry(src: Path, dst: Path, retries: int | None = None, delay: float = 0.25) -> None:
    """``os.replace`` that retries transient Windows sharing violations.

    Windows Defender and the Search indexer open freshly written files for a moment, and
    a cached ONNX session keeps its file mapped; both make the rename fail with
    ``PermissionError`` (WinError 5/32).  Non-permission errors are raised immediately.
    """
    n = REPLACE_RETRIES if retries is None else retries
    for i in range(max(1, n)):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i >= n - 1:
                raise
            time.sleep(delay)


def _rmtree(d: Path) -> None:
    """shutil.rmtree that clears the read-only bit (Windows) and retries busy entries."""
    import stat

    def _retry(func: Callable, path: str, _exc: object) -> None:
        for i in range(8):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
                return
            except FileNotFoundError:
                return
            except PermissionError:
                if i == 7:
                    raise
                time.sleep(0.25)

    if sys.version_info >= (3, 12):
        shutil.rmtree(d, onexc=_retry)
    else:  # pragma: no cover - Python 3.11
        shutil.rmtree(d, onerror=_retry)


__all__ = [
    "model_dir",
    "is_downloaded",
    "ensure_model",
    "delete_model",
    "download_url",
    "env_override",
    "local_path",
    "variant_files",
    "downloaded_size_mb",
    "recorded_hashes",
]
