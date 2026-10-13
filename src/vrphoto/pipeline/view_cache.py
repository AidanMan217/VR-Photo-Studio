"""On-disk cache of AI-generated back views (SHARP 360, design 3.5.3 "Cache").

MV-Adapter needs minutes on a GTX 10-series card for one set of six views, and the views
depend only on what the generator was given: the levelled input picture (made from the
photo, its cut-out and SHARP's result), the generation settings (steps, guidance, prompt,
seed) and the worker.  The seam angle, the back quality's mould grid, the colour match, the
relief and the Brush polish all work on the views AFTER they were made, so a re-run that
changes only those finds the views here and skips the generator.

**Layout.**  ``<cache_dir>/backviews/<key>/`` holds one set's files exactly as the worker
wrote them (``view_0.png`` ... ``view_5.png``, ``cameras.json`` whose ``file`` entries are
relative names) plus ``entry.json`` (``{"version", "key", "files", "bytes", "created",
"meta"}``).  An entry is complete once ``entry.json`` exists and lists files that all exist:
:meth:`ViewCache.put` copies into a hidden temp folder next to the entry and renames it into
place, so a reader never sees half a set (a crash leaves a temp folder that the next prune
removes).

**Keys** (:func:`make_key`): SHA-256 of the caller's input digest (SHARP 360: what the
levelled picture is made from - the photo, the cut-out, SHARP's camera and a coarse
fingerprint of its cloud - not its pixels, which differ slightly with the back quality's
mould grid), the source id, the
generation parameters (one key per seed, so a run with two attempts reuses the first
attempt's set of an earlier run) and the worker version (manifest version plus a hash of the
worker script, so an updated worker never returns stale views).  :meth:`ViewCache.discard`
deletes an entry whose files turned out to be damaged.

**Size.**  Least recently used entries (by ``entry.json``'s modification time, which
:meth:`ViewCache.get` refreshes) are deleted while the total is above ``max_bytes`` (2 GB by
default, about 800 sets of six 512 px views).  Cache problems never fail a job: unreadable or
incomplete entries are discarded, a failed write only means generating again; every method
logs and returns None / an empty result instead of raising for I/O errors.  Thread-safe.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..core.fsutil import atomic_write_bytes, replace_with_retry
from ..core.log import get_logger

log = get_logger(__name__)

#: Folder below ``Settings.cache_dir``.
SUBDIR = "backviews"
#: Default size cap (bytes).
DEFAULT_MAX_BYTES = 2_000_000_000
#: Bump when the entry layout or the key derivation changes.
CACHE_VERSION = 1
ENTRY_FILE = "entry.json"
_TMP_PREFIX = "."
_TMP_SUFFIX = ".tmp"
_STALE_TMP_S = 3600.0  # temp folders older than this are leftovers of a crash
_HASH_CHUNK = 1 << 20

_lock = threading.RLock()  # one lock for every ViewCache: they may share a folder


def _canonical(params: Mapping[str, Any]) -> str:
    return json.dumps(dict(params), sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)


def make_key(input_digest: str, *, source: str, params: Mapping[str, Any], worker: str) -> str:
    """The cache key of one generated set (40 hex characters, safe as a folder name).

    ``input_digest``: a digest of what the generator's input picture is made from (or of
    the picture itself);
    ``source``: the back source (``"mvadapter"``); ``params``: everything else the
    generator's output depends on (steps, guidance, prompt, seed, ...; JSON-able, order does
    not matter); ``worker``: the worker's version (:func:`worker_version`)."""
    text = f"v{CACHE_VERSION}|{input_digest}|{source}|{_canonical(params)}|{worker}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:40]


def worker_version(manifest: Any) -> str:
    """``"<manifest version>+<first 12 hex of the worker script's SHA-256>"`` (just the
    version when the script cannot be read) - part of every key, so a changed worker never
    returns views made by the old one."""
    version = str(getattr(manifest, "version", "") or "?")
    try:
        path = manifest.worker_path()
    except Exception:
        path = None
    if path is None:
        return version
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
                h.update(chunk)
        return f"{version}+{h.hexdigest()[:12]}"
    except OSError:
        return version


def _valid_key(key: Any) -> bool:
    return isinstance(key, str) and 0 < len(key) <= 128 and all(c.isalnum() or c in "-_" for c in key)


def _rmtree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)


class ViewCache:
    """Size-capped LRU cache of generated view sets in ``root`` (see the module docstring).

    Args:
        root: the cache folder (created on demand), e.g. ``settings.cache_dir / "backviews"``.
        max_bytes: total size cap; ``0`` disables storing (:meth:`get` still works).
    """

    def __init__(self, root: str | os.PathLike[str], max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.root = Path(root)
        self.max_bytes = max(0, int(max_bytes))
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ helpers
    def path_for(self, key: str) -> Path:
        """The folder of entry ``key`` (ValueError for a key that is not a safe name)."""
        if not _valid_key(key):
            raise ValueError(f"Invalid view cache key {key!r}")
        return self.root / key

    @staticmethod
    def _read_entry(folder: Path) -> dict[str, Any] | None:
        """The entry record of a complete entry folder, else None."""
        try:
            rec = json.loads((folder / ENTRY_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(rec, dict) or rec.get("version") != CACHE_VERSION:
            return None
        files = rec.get("files")
        if not isinstance(files, list) or not files or not all(isinstance(f, str) for f in files):
            return None
        if not all(Path(f).name == f and (folder / f).is_file() for f in files):
            return None
        return rec

    def _entries(self) -> list[tuple[float, int, Path]]:
        """``(last use, bytes, folder)`` of every entry folder (complete or not)."""
        out: list[tuple[float, int, Path]] = []
        try:
            folders = [p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(_TMP_PREFIX)]
        except OSError:
            return out
        for d in folders:
            size = 0
            try:
                for f in d.iterdir():
                    try:
                        size += f.stat().st_size
                    except OSError:
                        continue
                stamp = d / ENTRY_FILE
                mtime = stamp.stat().st_mtime if stamp.is_file() else d.stat().st_mtime
            except OSError:
                continue
            out.append((mtime, size, d))
        return out

    # ------------------------------------------------------------------ API
    def get(self, key: str) -> Path | None:
        """The folder of a complete cached set for ``key`` (its ``cameras.json`` names the
        pictures next to it), or None.  Refreshes the entry's LRU time; an incomplete or
        unreadable entry is deleted."""
        try:
            folder = self.path_for(key)
        except ValueError:
            return None
        with _lock:
            if not folder.is_dir():
                self.misses += 1
                return None
            if self._read_entry(folder) is None:
                log.warning("Discarding an incomplete cached view set %s", folder.name)
                _rmtree(folder)
                self.misses += 1
                return None
            try:
                now = time.time()
                os.utime(folder / ENTRY_FILE, (now, now))
            except OSError:
                pass
            self.hits += 1
        log.debug("View cache hit %s", key)
        return folder

    def put(
        self, key: str, files: Iterable[str | os.PathLike[str]], *, meta: Mapping[str, Any] | None = None
    ) -> Path | None:
        """Store copies of ``files`` (one set: its pictures and ``cameras.json``; names must
        differ) as entry ``key``, replacing an older entry, and prune to the size cap.
        Returns the entry folder, or None when storing is disabled or failed (the job then
        simply uses the worker's own files)."""
        if self.max_bytes <= 0:
            return None
        try:
            folder = self.path_for(key)
        except ValueError:
            log.warning("Not caching views under the invalid key %r", key)
            return None
        srcs = [Path(f) for f in files]
        names = [p.name for p in srcs]
        if not srcs or len(set(names)) != len(names) or ENTRY_FILE in names:
            log.warning("Not caching a view set with missing or clashing file names: %s", names)
            return None
        try:
            total = sum(p.stat().st_size for p in srcs)
        except OSError as e:
            log.warning("Not caching the view set %s: %s", key, e)
            return None
        if total > self.max_bytes:
            log.debug("View set %s (%d bytes) is larger than the cache cap; not cached", key, total)
            return None
        with _lock:
            tmp = self.root / f"{_TMP_PREFIX}{key}.{secrets.token_hex(4)}{_TMP_SUFFIX}"
            try:
                tmp.mkdir(parents=True)
                for p in srcs:
                    shutil.copyfile(p, tmp / p.name)
                rec = {
                    "version": CACHE_VERSION,
                    "key": key,
                    "files": names,
                    "bytes": int(total),
                    "created": time.time(),
                    "meta": json.loads(_canonical(meta or {})),
                }
                atomic_write_bytes(tmp / ENTRY_FILE, json.dumps(rec, indent=1).encode("utf-8"), fsync=False)
                if folder.exists():
                    _rmtree(folder)
                replace_with_retry(tmp, folder)
            except OSError as e:
                log.warning("Could not write the view cache (%s): %s", self.root, e)
                _rmtree(tmp)
                return None
            self._prune_locked()
        log.debug("Cached view set %s (%d files, %.1f MB)", key, len(names), total / 1e6)
        return folder if folder.is_dir() else None

    def discard(self, key: str) -> bool:
        """Delete entry ``key`` (a set whose files turned out to be damaged, e.g. a picture
        truncated by a crash after it was stored), so it is generated again.  True when an
        entry was removed; never raises."""
        try:
            folder = self.path_for(key)
        except ValueError:
            return False
        with _lock:
            if not folder.is_dir():
                return False
            log.warning("Discarding the damaged cached view set %s", folder.name)
            _rmtree(folder)
            return not folder.exists()

    def salvage(self, folder: str | os.PathLike[str], keys: Mapping[Any, str]) -> dict[Any, Path]:
        """Keep the finished sets a worker left in ``folder`` before it failed or was
        cancelled, so the next run does not generate them again.

        Every ``cameras.json`` below ``folder`` whose ``seed`` is a key of ``keys`` and whose
        listed pictures all exist is stored under ``keys[seed]`` (when not cached already).
        ``keys`` maps each seed of the request to its cache key (:func:`make_key`): a set's
        folder does not know the input picture it was made from.  Returns ``{seed: entry
        folder}`` of the sets stored; never raises."""
        out: dict[Any, Path] = {}
        want = {str(k): (k, v) for k, v in keys.items()}
        try:
            found = sorted(Path(folder).rglob("cameras.json"))
        except OSError:
            return out
        for cams in found:
            try:
                doc = json.loads(cams.read_text(encoding="utf-8"))
                seed = str(doc.get("seed"))
                views = [cams.parent / str(v["file"]) for v in doc.get("views", [])]
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
            if seed not in want or not views or not all(p.is_file() for p in views):
                continue
            seed_key, key = want[seed]
            if seed_key in out:
                continue
            existing = self.get(key)
            stored = existing or self.put(key, [cams, *views], meta={"salvaged": True})
            if stored is not None:
                out[seed_key] = stored
                if existing is None:
                    log.info("Kept the finished view set for seed %s from an interrupted run", seed)
        return out

    def size_bytes(self) -> int:
        """Total size of all entries."""
        return sum(s for _, s, _ in self._entries())

    def __len__(self) -> int:
        return len(self._entries())

    def __contains__(self, key: object) -> bool:
        try:
            return isinstance(key, str) and self._read_entry(self.path_for(key)) is not None
        except ValueError:
            return False

    def prune(self, max_bytes: int | None = None) -> int:
        """Delete least recently used entries until the total is <= the cap (and leftover
        temp folders of crashed writes); returns the bytes freed."""
        with _lock:
            return self._prune_locked(max_bytes)

    def _prune_locked(self, max_bytes: int | None = None) -> int:
        cap = self.max_bytes if max_bytes is None else max(0, int(max_bytes))
        now = time.time()
        try:
            for p in self.root.iterdir():
                if p.name.startswith(_TMP_PREFIX) and p.name.endswith(_TMP_SUFFIX):
                    try:
                        if now - p.stat().st_mtime > _STALE_TMP_S:
                            _rmtree(p)
                    except OSError:
                        continue
        except OSError:
            pass
        entries = sorted(self._entries())  # least recently used first
        total = sum(s for _, s, _ in entries)
        freed = 0
        for _t, size, p in entries:
            if total <= cap:
                break
            _rmtree(p)
            if not p.exists():
                total -= size
                freed += size
        if freed:
            log.info("View cache pruned %.1f MB", freed / 1e6)
        return freed

    def clear(self) -> None:
        """Delete every entry and temp folder."""
        with _lock:
            try:
                for p in self.root.iterdir():
                    if p.is_dir():
                        _rmtree(p)
            except OSError:
                pass


# --------------------------------------------------------------------------- module-level shortcuts
def default_root(settings: Any = None) -> Path:
    """``<settings.cache_dir>/backviews`` (the process settings when ``settings`` is None)."""
    if settings is None:
        from ..core.config import get_settings

        settings = get_settings()
    return Path(settings.cache_dir) / SUBDIR


def _cache(root: str | os.PathLike[str] | None) -> ViewCache:
    return ViewCache(root if root is not None else default_root())


def get(key: str, *, root: str | os.PathLike[str] | None = None) -> Path | None:
    """:meth:`ViewCache.get` on the default cache (or ``root``)."""
    return _cache(root).get(key)


def put(
    key: str,
    files: Iterable[str | os.PathLike[str]],
    *,
    root: str | os.PathLike[str] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Path | None:
    """:meth:`ViewCache.put` on the default cache (or ``root``)."""
    return _cache(root).put(key, files, meta=meta)


def salvage(
    folder: str | os.PathLike[str], keys: Mapping[Any, str], *, root: str | os.PathLike[str] | None = None
) -> dict[Any, Path]:
    """:meth:`ViewCache.salvage` on the default cache (or ``root``)."""
    return _cache(root).salvage(folder, keys)


__all__ = [
    "CACHE_VERSION",
    "DEFAULT_MAX_BYTES",
    "ENTRY_FILE",
    "SUBDIR",
    "ViewCache",
    "default_root",
    "get",
    "make_key",
    "put",
    "salvage",
    "worker_version",
]
