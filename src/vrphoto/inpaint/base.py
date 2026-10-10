"""Inpainter protocol and factory.

``get_inpainter(name)`` names: ``"auto"`` (LaMa, else MI-GAN, else OpenCV Telea; model
download/load failures fall back automatically and are logged), ``"lama"``, ``"migan"``,
``"opencv"`` (Telea), ``"opencv-ns"`` (Navier-Stokes), ``"pushpull"``.  Construction is
cheap and never touches the network: models load (and download) on first use.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.errors import Cancelled, InferenceError, InputError, ModelUnavailable
from ..core.log import get_logger
from ..core.types import RGB, CancelToken, ProgressReporter
from ..segment import _runtime as rt

log = get_logger(__name__)

__all__ = ["Inpainter", "AutoInpainter", "get_inpainter", "list_inpainters", "INPAINTER_NAMES"]


@runtime_checkable
class Inpainter(Protocol):
    """``inpaint(img, mask) -> RGB``: fill ``mask`` (bool HxW, True = fill).

    The output has the input's size and its unmasked pixels are unchanged."""

    id: str

    def inpaint(
        self,
        img: RGB,
        mask: np.ndarray,
        *,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> RGB: ...


#: (name, title, model id or None)
_CATALOG: list[tuple[str, str, str | None]] = [
    ("auto", "Automatic (best available)", None),
    ("lama", "LaMa (best quality, AI)", "lama"),
    ("migan", "MI-GAN (fast, AI)", "migan"),
    ("opencv", "OpenCV Telea (no download)", None),
    ("opencv-ns", "OpenCV Navier-Stokes (no download)", None),
    ("pushpull", "Smooth fill (instant, blurry)", None),
]
INPAINTER_NAMES: tuple[str, ...] = tuple(n for n, _, _ in _CATALOG)
_ALIASES = {
    "telea": "opencv",
    "ns": "opencv-ns",
    "push-pull": "pushpull",
    "mi-gan": "migan",
    "big-lama": "lama",
}


def list_inpainters() -> list[tuple[str, str]]:
    """``(name, title)`` pairs for the GUI, in display order."""
    return [(n, t) for n, t, _ in _CATALOG]


def _canonical(name: str) -> str:
    key = (name or "auto").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in INPAINTER_NAMES:
        raise InputError(
            f"Unknown inpainting method '{name}'.",
            hint="Choose one of: " + ", ".join(INPAINTER_NAMES),
        )
    return key


def _model_id(name: str) -> str | None:
    for n, _t, mid in _CATALOG:
        if n == name:
            return mid
    return None


def _make(name: str, device: str | int) -> Inpainter:
    if name == "lama":
        from .lama import LamaInpainter

        return LamaInpainter(device)
    if name == "migan":
        from .migan import MiganInpainter

        return MiganInpainter(device)
    from .opencv import OpenCVInpainter

    if name == "opencv":
        return OpenCVInpainter("telea")
    if name == "opencv-ns":
        return OpenCVInpainter("ns")
    if name == "pushpull":
        return OpenCVInpainter("pushpull")
    raise InputError(f"Unknown inpainting method '{name}'.")  # pragma: no cover


class AutoInpainter:
    """Tries LaMa, then MI-GAN, then OpenCV Telea (always in this priority order).

    A model candidate is skipped while it is neither available locally nor downloadable,
    or while a recent download of it failed (:func:`_runtime.auto_candidate_ok`);
    otherwise it is tried (downloading on first use) and download / load / inference
    failures fall through to the next candidate with a logged warning.  A failed
    candidate is avoided for a while (:class:`_runtime.FailureMemo`), but a model the user
    downloads meanwhile is retried at once.  ``resolved`` is the backend used by the
    latest call.  Only :class:`Cancelled` and invalid input (:class:`InputError`)
    propagate."""

    def __init__(self, candidates: list[str] | None = None, device: str | int = "auto") -> None:
        names = candidates or ["lama", "migan", "opencv"]
        self._candidates: list[str] = []
        for c in (_canonical(n) for n in names):
            if c != "auto" and c not in self._candidates:
                self._candidates.append(c)
        if not any(_model_id(c) is None for c in self._candidates):
            self._candidates.append("opencv")  # always end with a model-free method
        self.device = device
        self._lock = threading.Lock()
        self._instances: dict[str, Inpainter] = {}
        self._failed = rt.FailureMemo()
        self._resolved: str | None = None

    @property
    def id(self) -> str:
        """Id of the backend in use (the one that will be tried first before first use)."""
        return self._resolved or self.preferred

    @property
    def preferred(self) -> str:
        order = self._order()
        return order[0] if order else "opencv"

    @property
    def resolved(self) -> str | None:
        return self._resolved

    @property
    def candidates(self) -> list[str]:
        return list(self._candidates)

    def needs_download(self) -> bool:
        """True if the preferred backend is a model that is not downloaded yet."""
        model_id = _model_id(self.preferred)
        return model_id is not None and not rt.is_available(model_id)

    def _order(self) -> list[str]:
        return [
            n for n in self._candidates if not self._failed.blocked(n) and rt.auto_candidate_ok(_model_id(n))
        ]

    def _get(self, name: str) -> Inpainter:
        with self._lock:
            inst = self._instances.get(name)
            if inst is None:
                inst = _make(name, self.device)
                self._instances[name] = inst
            return inst

    def inpaint(
        self,
        img: RGB,
        mask: np.ndarray,
        *,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> RGB:
        """Inpaint with the first backend that works (see class docstring)."""
        rgb = rt.as_rgb(img)
        m = rt.as_bool_mask(mask, rgb.shape[:2])
        if not m.any():
            return rgb.copy()
        order = self._order() or ["opencv"]
        last_err: Exception | None = None
        for name in order:
            try:
                out = self._get(name).inpaint(rgb, m, progress=progress, cancel=cancel)
            except Cancelled:
                raise
            except (ModelUnavailable, InferenceError) as e:
                last_err = e
                self._failed.add(name, _model_id(name))
                log.warning("Inpainting with %s unavailable (%s); falling back", name, e)
                continue
            if self._resolved != name:
                log.info("Auto inpainting uses %s", name)
            self._resolved = name
            return out
        log.error("All inpainting methods failed (%s); using push-pull fill", last_err)
        self._resolved = "pushpull"
        return _make("pushpull", self.device).inpaint(rgb, m, progress=progress, cancel=cancel)


def get_inpainter(name: str = "auto", device: str | int = "auto") -> Inpainter:
    """Create an inpainter by name (see module docstring).  Never downloads anything and
    never raises for ``"auto"``; unknown names raise :class:`InputError`."""
    key = _canonical(name)
    if key == "auto":
        return AutoInpainter(device=device)
    return _make(key, device)
