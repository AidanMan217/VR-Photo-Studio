"""Depth estimator protocol, factory and shared helpers.

Every estimator returns a :class:`~vrphoto.core.types.DepthResult` at the INPUT
resolution whose ``disparity`` is float32 in [0, 1] with 1 = nearest.  Model-backed
estimators load (and on first use download) their ONNX model lazily inside
:meth:`estimate`, so construction is instant and download progress flows through the
``progress`` reporter of the first call.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.errors import DownloadError, InferenceError, InputError, ModelUnavailable
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, DepthResult, ProgressReporter

log = get_logger(__name__)


@runtime_checkable
class DepthEstimator(Protocol):
    """Anything that turns an RGB image into a :class:`DepthResult`."""

    id: str

    def estimate(
        self, img: RGB, *, progress: ProgressReporter | None = None, cancel: CancelToken | None = None
    ) -> DepthResult: ...


# name -> (title, factory key).  Order is the order shown in the GUI.
_ESTIMATORS: dict[str, str] = {
    "auto": "Automatic (Depth Anything V2 Small, falls back to heuristic)",
    "da2-small": "Depth Anything V2 Small (fast, Apache-2.0)",
    "da2-base": "Depth Anything V2 Base (non-commercial)",
    "da2-large": "Depth Anything V2 Large (best detail, non-commercial)",
    "moge2": "MoGe-2 ViT-L (metric depth + field of view)",
    "da3": "Depth Anything 3 Mono Large (experimental)",
    "heuristic": "Heuristic (no AI model, instant)",
}
_ALIASES = {
    "moge2-vitl": "moge2",
    "moge": "moge2",
    "da3-mono-large": "da3",
    "da2": "da2-small",
    "none": "heuristic",
}


def list_estimators() -> list[tuple[str, str]]:
    """``(id, title)`` of every estimator name accepted by :func:`get_estimator`."""
    return list(_ESTIMATORS.items())


def get_estimator(name: str = "auto", device: str = "auto") -> DepthEstimator:
    """Create a depth estimator by name (see :func:`list_estimators`).

    Construction never touches the network or loads a model; that happens on the first
    :meth:`DepthEstimator.estimate` call.  Raises :class:`InputError` for unknown names.
    """
    key = (name or "auto").strip().lower()
    key = _ALIASES.get(key, key)
    if key == "auto":
        return AutoEstimator(device=device)
    if key == "heuristic":
        from .heuristic import HeuristicEstimator

        return HeuristicEstimator()
    if key in ("da2-small", "da2-base", "da2-large"):
        from .da2 import DA2Estimator

        return DA2Estimator(key, device=device)
    if key == "moge2":
        from .moge import MoGeEstimator

        return MoGeEstimator("moge2-vitl", device=device)
    if key == "da3":
        from .da3 import DA3Estimator

        return DA3Estimator("da3-mono-large", device=device)
    names = ", ".join(_ESTIMATORS)
    raise InputError(f"Unknown depth model '{name}'", hint=f"Choose one of: {names}")


RETRY_AFTER_S = 600.0  # an "auto" estimator that fell back retries the model after this


class AutoEstimator:
    """Depth Anything V2 Small when it can be loaded (downloading it on first use with
    progress), otherwise the no-model heuristic.

    The fallback is sticky so a batch job does not retry a failing download for every
    image, but not forever (the estimator lives as long as the app's run context): the
    model is tried again after ``retry_after_s`` seconds, or immediately once its files
    appear on disk (e.g. the user downloaded it from Settings > Models after a failed
    download)."""

    def __init__(
        self, device: str = "auto", model_id: str = "da2-small", *, retry_after_s: float = RETRY_AFTER_S
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.retry_after_s = float(retry_after_s)
        self._primary: DepthEstimator | None = None
        self._fallback: DepthEstimator | None = None
        self._failed_at = 0.0
        self._failed_download = False
        self._lock = threading.Lock()
        self.fallback_reason: str | None = None

    @property
    def id(self) -> str:  # type: ignore[override]
        return "heuristic" if self._fallback is not None else self.model_id

    def _retry_due(self) -> bool:
        if time.monotonic() - self._failed_at >= self.retry_after_s:
            return True
        if self._failed_download:  # the model was missing: retry as soon as it is on disk
            try:
                from ..models.download import is_downloaded

                return bool(is_downloaded(self.model_id))
            except Exception:  # pragma: no cover - defensive
                return False
        return False

    def _get(self) -> tuple[DepthEstimator, bool]:
        with self._lock:
            if self._fallback is not None:
                if not self._retry_due():
                    return self._fallback, True
                log.info("Trying depth model %s again", self.model_id)
                self._fallback = None
                self.fallback_reason = None
            if self._primary is None:
                from .da2 import DA2Estimator

                self._primary = DA2Estimator(self.model_id, device=self.device)
            return self._primary, False

    def estimate(
        self, img: RGB, *, progress: ProgressReporter | None = None, cancel: CancelToken | None = None
    ) -> DepthResult:
        """Estimate depth with DA2-small, or the heuristic if the model is unavailable."""
        est, is_fallback = self._get()
        if not is_fallback:
            try:
                return est.estimate(img, progress=progress, cancel=cancel)
            except ModelUnavailable as e:  # includes DownloadError; Cancelled propagates
                msg = str(e) + (f" ({e.hint})" if getattr(e, "hint", None) else "")
                log.warning(
                    "Depth model %s unavailable, using the heuristic estimator: %s", self.model_id, msg
                )
                from .heuristic import HeuristicEstimator

                with self._lock:
                    self._fallback = HeuristicEstimator()
                    self.fallback_reason = str(e)
                    self._failed_at = time.monotonic()
                    self._failed_download = isinstance(e, DownloadError)
                    est = self._fallback
            except InferenceError as e:  # runtime failure: fall back for this image only
                log.warning("Depth model %s failed on this image, using the heuristic: %s", self.model_id, e)
                from .heuristic import HeuristicEstimator

                est = HeuristicEstimator()
        return est.estimate(img, progress=progress, cancel=cancel)


def as_rgb(img: np.ndarray) -> RGB:
    """Validate/convert an input image to contiguous HxWx3 uint8 RGB.

    Accepts grey (HxW / HxWx1), RGBA (alpha dropped) and float images in [0, 1].
    Raises :class:`InputError` for anything else.
    """
    if not isinstance(img, np.ndarray):
        raise InputError("Depth estimation needs an image array")
    a = img
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4) or a.shape[0] < 2 or a.shape[1] < 2:
        raise InputError(f"Unsupported image shape {img.shape} for depth estimation")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = (np.clip(np.nan_to_num(a), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        elif a.dtype == np.uint16:
            a = (a >> 8).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def reporter(progress: ProgressReporter | None) -> ProgressReporter:
    """``progress`` or a no-op reporter."""
    return progress if progress is not None else ProgressReporter.null()


def check(cancel: CancelToken | None) -> None:
    """``cancel.check()`` if a token was given."""
    if cancel is not None:
        cancel.check()


__all__ = [
    "DepthEstimator",
    "AutoEstimator",
    "get_estimator",
    "list_estimators",
    "as_rgb",
]
