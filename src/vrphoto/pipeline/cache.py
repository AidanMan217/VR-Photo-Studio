"""On-disk cache of depth estimates (``.npz``), keyed by image content + model + size.

Depth estimation is the slowest step of most jobs and users often re-run the same photo
with different options (layout, strength, path...).  :class:`DepthCache` stores each
:class:`~vrphoto.core.types.DepthResult` as a compressed ``.npz`` in
``<cache_dir>/depth`` under a key derived from

* the SHA-256 of the source (the original file's bytes, or the pixel data of an array),
* the depth model id, and
* the processing size (the image size the model saw),

so a changed photo, model or "max processing size" never returns a stale result.  The
cache is size-capped: the least recently used entries (by file modification time, which
:meth:`DepthCache.get` refreshes) are deleted when the total exceeds ``max_bytes``.
Corrupt or unreadable entries are discarded silently.  All methods are thread-safe and
never raise for cache I/O problems (a failing cache just means recomputing).
"""

from __future__ import annotations

import hashlib
import io
import os
import secrets
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..core.log import get_logger
from ..core.types import DepthResult, Intrinsics

log = get_logger(__name__)

#: Default size cap of the depth cache.
DEFAULT_MAX_MB = 1024
#: Bump when the stored format or key derivation changes.
CACHE_VERSION = 1
_SUFFIX = ".npz"
_HASH_CHUNK = 1 << 20


def hash_bytes(data: bytes | bytearray | memoryview) -> str:
    """SHA-256 hex digest of ``data``."""
    return hashlib.sha256(bytes(data)).hexdigest()


def hash_file(path: str | os.PathLike[str]) -> str:
    """SHA-256 hex digest of a file's contents (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_array(arr: np.ndarray) -> str:
    """SHA-256 hex digest of an array's shape, dtype and pixel data."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(f"{a.shape}|{a.dtype.str}|".encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def content_hash(source: str | os.PathLike[str] | bytes | bytearray | memoryview | np.ndarray) -> str:
    """Hash of a cache source: a file path (its bytes), raw bytes or an image array."""
    if isinstance(source, np.ndarray):
        return hash_array(source)
    if isinstance(source, bytes | bytearray | memoryview):
        return hash_bytes(source)
    return hash_file(source)


def _size_str(size: int | tuple[int, int] | Iterable[int]) -> str:
    if isinstance(size, int):
        return str(size)
    return "x".join(str(int(v)) for v in size)


def make_key(
    source: str | os.PathLike[str] | bytes | np.ndarray,
    model_id: str,
    size: int | tuple[int, int],
    *,
    extra: str = "",
) -> str:
    """Cache key for a depth estimate.

    ``source``: the original file (hashed by content), raw bytes or the image array the
    model sees.  ``model_id``: the estimator id.  ``size``: processing size as ``(W, H)``
    (or the long side).  ``extra``: anything else that changes the result (e.g. pano face
    size).  The key is a 40-character hex string (safe as a file name).
    """
    digest = source if isinstance(source, str) and _looks_like_digest(source) else content_hash(source)
    text = f"v{CACHE_VERSION}|{digest}|{model_id}|{_size_str(size)}|{extra}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:40]


def _looks_like_digest(s: str) -> bool:
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s)


class DepthCache:
    """Size-capped LRU cache of :class:`DepthResult` objects in ``root``.

    Args:
        root: cache folder (created on demand), e.g. ``settings.cache_dir / "depth"``.
        max_bytes: total size cap; ``0`` disables storing (``get`` still works).
    """

    def __init__(self, root: str | os.PathLike[str], max_bytes: int = DEFAULT_MAX_MB * 1_000_000) -> None:
        self.root = Path(root)
        self.max_bytes = max(0, int(max_bytes))
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ helpers
    def path_for(self, key: str) -> Path:
        if not key or not all(c.isalnum() or c in "-_" for c in key):
            raise ValueError(f"Invalid cache key {key!r}")
        return self.root / f"{key}{_SUFFIX}"

    def _entries(self) -> list[tuple[float, int, Path]]:
        out: list[tuple[float, int, Path]] = []
        try:
            it = list(self.root.glob(f"*{_SUFFIX}"))
        except OSError:
            return out
        for p in it:
            try:
                st = p.stat()
            except OSError:
                continue
            out.append((st.st_mtime, st.st_size, p))
        return out

    # ------------------------------------------------------------------ API
    def get(self, key: str) -> DepthResult | None:
        """The cached result for ``key`` or None (refreshes the entry's LRU time)."""
        try:
            path = self.path_for(key)
        except ValueError:
            return None
        with self._lock:
            if not path.is_file():
                self.misses += 1
                return None
            try:
                with np.load(path, allow_pickle=False) as z:
                    res = _decode(z)
            except Exception as e:  # corrupt / truncated / foreign file
                log.warning("Discarding unreadable depth cache entry %s: %s", path.name, e)
                _unlink(path)
                self.misses += 1
                return None
            try:
                now = time.time()
                os.utime(path, (now, now))
            except OSError:
                pass
            self.hits += 1
        log.debug("Depth cache hit %s (%s)", key, res.model_id)
        return res

    def put(self, key: str, result: DepthResult) -> Path | None:
        """Store ``result`` under ``key`` (atomic write) and prune to the size cap.

        Returns the entry path, or None when storing is disabled or failed.
        """
        if self.max_bytes <= 0:
            return None
        try:
            path = self.path_for(key)
        except ValueError:
            return None
        try:
            payload = _encode(result)
        except Exception as e:
            log.warning("Depth result could not be cached: %s", e)
            return None
        if len(payload) > self.max_bytes:
            log.debug("Depth result (%d bytes) is larger than the cache cap; not cached", len(payload))
            return None
        with self._lock:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                tmp = self.root / f".{key}.{secrets.token_hex(4)}.tmp"
                try:
                    with open(tmp, "wb") as fh:
                        fh.write(payload)
                    os.replace(tmp, path)
                finally:
                    _unlink(tmp)
            except OSError as e:
                log.warning("Could not write the depth cache (%s): %s", self.root, e)
                return None
            self._prune_locked()
        return path

    def size_bytes(self) -> int:
        """Total size of all entries."""
        return sum(s for _, s, _ in self._entries())

    def __len__(self) -> int:
        return len(self._entries())

    def __contains__(self, key: object) -> bool:
        try:
            return isinstance(key, str) and self.path_for(key).is_file()
        except ValueError:
            return False

    def prune(self, max_bytes: int | None = None) -> int:
        """Delete least recently used entries until the total is <= the cap; returns bytes freed."""
        with self._lock:
            return self._prune_locked(max_bytes)

    def _prune_locked(self, max_bytes: int | None = None) -> int:
        cap = self.max_bytes if max_bytes is None else max(0, int(max_bytes))
        entries = sorted(self._entries())  # oldest first
        total = sum(s for _, s, _ in entries)
        freed = 0
        for _mtime, size, p in entries:
            if total <= cap:
                break
            if _unlink(p):
                total -= size
                freed += size
        if freed:
            log.debug("Depth cache pruned %.1f MB", freed / 1e6)
        return freed

    def clear(self) -> None:
        """Delete every entry (and stray temp files)."""
        with self._lock:
            for _, _, p in self._entries():
                _unlink(p)
            try:
                for p in self.root.glob(".*.tmp"):
                    _unlink(p)
            except OSError:
                pass


# --------------------------------------------------------------------------- encoding
def _encode(res: DepthResult) -> bytes:
    arrays: dict[str, np.ndarray] = {
        "disparity": np.asarray(res.disparity, dtype=np.float32),
        "model_id": np.array(str(res.model_id)),
        "version": np.array(CACHE_VERSION, dtype=np.int32),
    }
    if res.metric_m is not None:
        arrays["metric_m"] = np.asarray(res.metric_m, dtype=np.float32)
    if res.intrinsics is not None:
        k = res.intrinsics
        arrays["intrinsics"] = np.array([k.fx, k.fy, k.cx, k.cy, k.width, k.height], dtype=np.float64)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _decode(z: np.lib.npyio.NpzFile) -> DepthResult:
    if int(z["version"]) != CACHE_VERSION:
        raise ValueError("cache format version mismatch")
    disp = np.asarray(z["disparity"], dtype=np.float32)
    if disp.ndim != 2:
        raise ValueError("disparity is not a 2-D map")
    metric = np.asarray(z["metric_m"], dtype=np.float32) if "metric_m" in z.files else None
    K = None
    if "intrinsics" in z.files:
        fx, fy, cx, cy, w, h = (float(v) for v in z["intrinsics"])
        K = Intrinsics(fx, fy, cx, cy, int(w), int(h))
    return DepthResult(disparity=disp, model_id=str(z["model_id"]), metric_m=metric, intrinsics=K)


def _unlink(p: Path) -> bool:
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


__all__ = [
    "DEFAULT_MAX_MB",
    "CACHE_VERSION",
    "DepthCache",
    "make_key",
    "content_hash",
    "hash_bytes",
    "hash_file",
    "hash_array",
]
