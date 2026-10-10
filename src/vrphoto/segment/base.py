"""Segmenter protocol and factory.

``get_segmenter(name)`` names: ``"auto"`` (BiRefNet-lite, falling back to GrabCut when the
model cannot be downloaded/loaded), ``"birefnet-lite"``, ``"birefnet-portrait"``,
``"isnet"``, ``"u2net"``, ``"grabcut"``.  Construction is cheap: models load (and download)
on first use, so the GUI can create segmenters freely.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.errors import Cancelled, InferenceError, InputError, ModelUnavailable
from ..core.log import get_logger
from ..core.types import CancelToken, FloatMap, ProgressReporter
from . import _runtime as rt

log = get_logger(__name__)

__all__ = ["Segmenter", "AutoSegmenter", "get_segmenter", "list_segmenters", "SEGMENTER_NAMES"]

Rect = tuple[int, int, int, int]


@runtime_checkable
class Segmenter(Protocol):
    """Subject segmentation: ``segment(img) -> float32 HxW alpha`` (1 = subject).

    ``rect`` is ``(x, y, width, height)`` in pixels; when given, the subject lies inside it
    and everything outside is background."""

    id: str

    def segment(
        self,
        img: np.ndarray,
        *,
        rect: Rect | None = None,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> FloatMap: ...


#: (name, title, model id or None)
_CATALOG: list[tuple[str, str, str | None]] = [
    ("auto", "Automatic (best available)", None),
    ("birefnet-lite", "BiRefNet lite (general, AI)", "birefnet-lite"),
    ("birefnet-portrait", "BiRefNet portrait (people, AI)", "birefnet-portrait"),
    ("isnet", "IS-Net general (AI)", "isnet-general"),
    ("u2net", "U2-Net (fast, AI)", "u2net"),
    ("grabcut", "GrabCut (no download)", None),
]
SEGMENTER_NAMES: tuple[str, ...] = tuple(n for n, _, _ in _CATALOG)
_ALIASES = {"isnet-general": "isnet", "birefnet": "birefnet-lite", "u2-net": "u2net", "opencv": "grabcut"}


def list_segmenters() -> list[tuple[str, str]]:
    """``(name, title)`` pairs for the GUI, in display order."""
    return [(n, t) for n, t, _ in _CATALOG]


def _canonical(name: str) -> str:
    key = (name or "auto").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in SEGMENTER_NAMES:
        raise InputError(
            f"Unknown segmentation method '{name}'.",
            hint="Choose one of: " + ", ".join(SEGMENTER_NAMES),
        )
    return key


def _make(name: str, device: str | int) -> Segmenter:
    if name in ("birefnet-lite", "birefnet-portrait"):
        from .birefnet import BiRefNetSegmenter

        return BiRefNetSegmenter(name, device)
    if name == "isnet":
        from .isnet import ISNetSegmenter

        return ISNetSegmenter("isnet-general", device)
    if name == "u2net":
        from .isnet import U2NetSegmenter

        return U2NetSegmenter("u2net", device)
    if name == "grabcut":
        from .grabcut import GrabCutSegmenter

        return GrabCutSegmenter()
    raise InputError(f"Unknown segmentation method '{name}'.")  # pragma: no cover


class AutoSegmenter:
    """Tries each candidate in order and falls back when a model is unavailable.

    Candidates keep their priority order on every call.  A model candidate is skipped
    while it is neither available locally nor downloadable, or while a recent download of
    it failed (see :func:`_runtime.auto_candidate_ok`).  Model download/load failures and
    inference failures fall through to the next candidate (logged) and that candidate is
    avoided for a while (:class:`_runtime.FailureMemo`; a model downloaded meanwhile is
    retried at once).  :class:`Cancelled` and bad input propagate.  The last candidate
    (GrabCut) needs no model, so ``segment`` only raises for cancellation or invalid
    input.  ``resolved`` is the backend used by the latest call."""

    def __init__(self, candidates: list[str] | None = None, device: str | int = "auto") -> None:
        names = candidates or ["birefnet-lite", "grabcut"]
        self._candidates: list[str] = []
        for c in (_canonical(n) for n in names):
            if c != "auto" and c not in self._candidates:
                self._candidates.append(c)
        if "grabcut" not in self._candidates:
            self._candidates.append("grabcut")
        self.device = device
        self._lock = threading.Lock()
        self._instances: dict[str, Segmenter] = {}
        self._failed = rt.FailureMemo()
        self._resolved: str | None = None

    @property
    def id(self) -> str:
        """Id of the backend in use (the preferred one before first use)."""
        return self._resolved or self.preferred

    @property
    def preferred(self) -> str:
        """The backend that will be tried first."""
        return self._order()[0]

    @property
    def candidates(self) -> list[str]:
        return list(self._candidates)

    def needs_download(self) -> bool:
        """True if the preferred backend is an AI model that is not downloaded yet."""
        model_id = _model_id(self.preferred)
        return model_id is not None and not rt.is_available(model_id)

    @property
    def resolved(self) -> str | None:
        return self._resolved

    def _get(self, name: str) -> Segmenter:
        with self._lock:
            inst = self._instances.get(name)
            if inst is None:
                inst = _make(name, self.device)
                self._instances[name] = inst
            return inst

    def segment(
        self,
        img: np.ndarray,
        *,
        rect: Rect | None = None,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> FloatMap:
        """Segment with the first candidate that works (see class docstring)."""
        rgb = rt.as_rgb(img)
        last_err: Exception | None = None
        for name in self._order():
            try:
                alpha = self._get(name).segment(rgb, rect=rect, progress=progress, cancel=cancel)
            except Cancelled:
                raise
            except (ModelUnavailable, InferenceError) as e:
                last_err = e
                self._failed.add(name, _model_id(name))
                log.warning("Segmentation with %s unavailable (%s); falling back", name, e)
                continue
            if self._resolved != name:
                log.info("Auto segmentation uses %s", name)
            self._resolved = name
            return alpha
        log.error("All segmentation methods failed (%s); using GrabCut", last_err)
        self._resolved = "grabcut"
        return _make("grabcut", self.device).segment(rgb, rect=rect, progress=progress, cancel=cancel)

    def _order(self) -> list[str]:
        order = [
            n for n in self._candidates if not self._failed.blocked(n) and rt.auto_candidate_ok(_model_id(n))
        ]
        return order or ["grabcut"]


def _model_id(name: str) -> str | None:
    for n, _t, mid in _CATALOG:
        if n == name:
            return mid
    return None


def get_segmenter(name: str = "auto", device: str | int = "auto") -> Segmenter:
    """Create a segmenter by name (see module docstring).  Never downloads anything;
    raises :class:`InputError` for unknown names only."""
    key = _canonical(name)
    if key == "auto":
        return AutoSegmenter(device=device)
    return _make(key, device)
