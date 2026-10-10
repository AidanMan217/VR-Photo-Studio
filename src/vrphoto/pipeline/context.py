"""Run context: settings plus lazily created, reused models and the depth cache.

A :class:`RunContext` is shared by all jobs of a :class:`~vrphoto.pipeline.queue.JobQueue`
(and by consecutive CLI jobs), so a depth / segmentation / inpainting model is loaded once
and reused.  Instances are cached per ``(name, device)``; creating one never downloads or
loads anything (that happens on first use inside the estimator, with progress).

``"auto"`` (or ``None``) model names resolve to the user's choice in Settings
(``depth_model``, ``inpaint_model``, ``segment_model``), which itself defaults to
``"auto"`` - the best model that is available, with a model-free fallback.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import Settings, get_settings
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, DepthResult, ProgressReporter
from .cache import DEFAULT_MAX_MB, DepthCache, make_key

log = get_logger(__name__)

#: Estimator ids that are too cheap to be worth caching.
_UNCACHED_ESTIMATORS = frozenset({"heuristic"})


def _norm(value: str | None, default: str) -> str:
    v = str(value).strip() if value is not None else ""
    return v or default


class RunContext:
    """Settings + cached model instances + depth cache for running jobs.

    Args:
        settings: application settings (default: the process-wide :func:`get_settings`).
        cache: depth cache to use; default ``DepthCache(settings.cache_dir / "depth")``
            capped at ``settings.depth_cache_mb`` (if that setting exists) or
            :data:`~vrphoto.pipeline.cache.DEFAULT_MAX_MB`.
        use_cache: False disables the depth cache entirely.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache: DepthCache | None = None,
        use_cache: bool = True,
    ) -> None:
        self.settings = settings if settings is not None else get_settings()
        self._lock = threading.RLock()
        self._estimators: dict[tuple[str, str], Any] = {}
        self._segmenters: dict[tuple[str, str], Any] = {}
        self._inpainters: dict[tuple[str, str], Any] = {}
        self._cache: DepthCache | None = cache
        self._use_cache = bool(use_cache)

    # ------------------------------------------------------------------ cache
    @property
    def cache(self) -> DepthCache | None:
        """The depth cache (created on first access), or None when disabled."""
        if not self._use_cache:
            return None
        with self._lock:
            if self._cache is None:
                mb = getattr(self.settings, "depth_cache_mb", DEFAULT_MAX_MB)
                try:
                    mb = float(mb)
                except (TypeError, ValueError):
                    mb = DEFAULT_MAX_MB
                root = Path(self.settings.cache_dir) / "depth"
                self._cache = DepthCache(root, max_bytes=int(max(0.0, mb) * 1_000_000))
            return self._cache

    # ------------------------------------------------------------------ resolution
    def resolve_device(self, device: str | None = None) -> str:
        """``device`` or, for ``None``/``"auto"``, the Settings device (default "auto")."""
        d = _norm(device, "auto")
        if d.lower() == "auto":
            d = _norm(getattr(self.settings, "device", "auto"), "auto")
        return d

    def resolve_name(self, kind: str, name: str | None = None) -> str:
        """Model name for ``kind`` ("depth" | "segment" | "inpaint"); "auto" -> Settings."""
        n = _norm(name, "auto")
        if n.lower() == "auto":
            n = _norm(getattr(self.settings, f"{kind}_model", "auto"), "auto")
        return n

    def _get(
        self,
        store: dict[tuple[str, str], Any],
        kind: str,
        name: str | None,
        device: str | None,
        factory: Callable[[str, str], Any],
    ) -> Any:
        key = (self.resolve_name(kind, name).lower(), self.resolve_device(device).lower())
        with self._lock:
            inst = store.get(key)
            if inst is None:
                inst = factory(key[0], key[1])
                store[key] = inst
                log.debug("Created %s model %s on %s", kind, key[0], key[1])
            return inst

    # ------------------------------------------------------------------ models
    def estimator(self, name: str | None = None, device: str | None = None) -> Any:
        """Depth estimator (``depth.get_estimator``), cached per (name, device)."""
        from ..depth import get_estimator

        return self._get(self._estimators, "depth", name, device, lambda n, d: get_estimator(n, d))

    def segmenter(self, name: str | None = None, device: str | None = None) -> Any:
        """Subject segmenter (``segment.get_segmenter``), cached per (name, device)."""
        from ..segment import get_segmenter

        return self._get(self._segmenters, "segment", name, device, lambda n, d: get_segmenter(n, d))

    def inpainter(self, name: str | None = None, device: str | None = None) -> Any:
        """Inpainter (``inpaint.get_inpainter``), cached per (name, device)."""
        from ..inpaint import get_inpainter

        return self._get(self._inpainters, "inpaint", name, device, lambda n, d: get_inpainter(n, d))

    # ------------------------------------------------------------------ depth with cache
    def depth(
        self,
        img: RGB,
        *,
        estimator: Any | None = None,
        name: str | None = None,
        device: str | None = None,
        source: Any | None = None,
        extra: str = "",
        compute: Callable[[Any], DepthResult] | None = None,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> DepthResult:
        """Estimate depth for ``img``, using the depth cache when possible.

        ``estimator`` (default ``self.estimator(name, device)``) does the work.
        ``source`` identifies the content for the cache key: the original file path (its
        bytes are hashed) or None to hash ``img`` itself.  ``extra`` distinguishes
        different computations of the same image (e.g. ``"pano:518"``).  ``compute``
        (``est -> DepthResult``) replaces ``est.estimate(img)`` (used for 360 panoramas).
        Results of the instant heuristic estimator are not cached.  The cache is keyed
        by the estimator's id at lookup time and the result is stored under the id of
        the model that actually produced it (so an "auto" estimator that fell back to
        the heuristic never poisons the cache for the real model).
        """
        est = estimator if estimator is not None else self.estimator(name, device)
        h, w = img.shape[:2]
        cache = self.cache
        est_id = str(getattr(est, "id", type(est).__name__))
        digest: str | None = None
        if cache is not None and est_id not in _UNCACHED_ESTIMATORS:
            try:
                from .cache import content_hash

                digest = content_hash(source if source is not None else img)
            except OSError as e:
                log.debug("Cannot hash %s for the depth cache: %s", source, e)
                digest = None
            if digest is not None:
                hit = cache.get(make_key(digest, est_id, (w, h), extra=extra))
                if hit is not None and hit.disparity.shape == (h, w):
                    if progress is not None:
                        progress(1.0, "Depth loaded from cache")
                    return hit
        if compute is not None:
            res = compute(est)
        else:
            res = est.estimate(img, progress=progress, cancel=cancel)
        if cache is not None and digest is not None:
            used = str(res.model_id or est_id)
            used_id = used.split(":", 1)[1] if used.startswith("pano:") and extra.startswith("pano") else used
            if used_id not in _UNCACHED_ESTIMATORS and np.asarray(res.disparity).shape == (h, w):
                cache.put(make_key(digest, used_id, (w, h), extra=extra), res)
        return res

    # ------------------------------------------------------------------ housekeeping
    def clear(self) -> None:
        """Forget all model instances (e.g. after the user changed the device)."""
        with self._lock:
            self._estimators.clear()
            self._segmenters.clear()
            self._inpainters.clear()

    def __repr__(self) -> str:
        return (
            f"RunContext(estimators={list(self._estimators)}, segmenters={list(self._segmenters)}, "
            f"inpainters={list(self._inpainters)})"
        )


__all__ = ["RunContext"]
