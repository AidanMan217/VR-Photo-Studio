"""Private helpers shared by the ONNX-backed segmenters and inpainters.

Everything model-related goes through the ``vrphoto.models`` contract
(``models.ort.load_model`` / ``OrtModel``, ``models.download.is_downloaded``,
``models.registry.get_spec``); those imports are lazy so this module (and the
``segment`` / ``inpaint`` packages) import without onnxruntime or network access.

Wrappers accept an injected ``model`` object (anything implementing
:class:`SessionLike`) so they can be unit-tested with fake sessions.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from ..core.errors import (
    UI_MODELS,
    Cancelled,
    DownloadError,
    InferenceError,
    InputError,
    ModelUnavailable,
    VRPhotoError,
)
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter

log = get_logger(__name__)

_MODEL_HINT = (
    "Check your internet connection (AI models are downloaded once, on first use), "
    f"or delete the model in the {UI_MODELS} so it is downloaded again."
)


@runtime_checkable
class SessionLike(Protocol):
    """The subset of ``vrphoto.models.ort.OrtModel`` the wrappers rely on."""

    def input_names(self) -> list[str]: ...

    def input_shapes(self) -> list[list[int | str | None]]: ...

    def run(
        self, feeds: dict[str, np.ndarray], output_names: list[str] | None = None
    ) -> dict[str, np.ndarray]: ...


# ----------------------------------------------------------------------------- registry
def env_var_for(model_id: str) -> str:
    """Name of the ``VRPHOTO_MODEL_<ID>`` override variable for ``model_id``."""
    return "VRPHOTO_MODEL_" + model_id.upper().replace("-", "_").replace(".", "_")


def env_override(model_id: str) -> Path | None:
    """Path given by the ``VRPHOTO_MODEL_<ID>`` environment override, if it exists."""
    val = os.environ.get(env_var_for(model_id), "").strip()
    if not val:
        return None
    p = Path(val).expanduser()
    return p if p.is_file() else None


def _spec(model_id: str) -> Any | None:
    try:
        from ..models.registry import get_spec  # lazy: other owner's module
    except Exception:  # ImportError or a broken module: behave as "unknown model"
        return None
    try:
        return get_spec(model_id)
    except Exception:
        return None


def model_title(model_id: str) -> str:
    """Human readable model title (registry title, else the id)."""
    spec = _spec(model_id)
    title = getattr(spec, "title", "") if spec is not None else ""
    return str(title) or model_id


#: ``dml_ok`` fallbacks used when the registry is unavailable (LaMa crashes on DirectML).
_DML_DEFAULTS: dict[str, bool] = {"lama": False}


def spec_dml_ok(model_id: str, default: bool | None = None) -> bool:
    """Registry ``dml_ok`` flag (False forces CPU on DirectML)."""
    if default is None:
        default = _DML_DEFAULTS.get(model_id, True)
    spec = _spec(model_id)
    return bool(getattr(spec, "dml_ok", default)) if spec is not None else default


def spec_input_size(model_id: str) -> tuple[int, int] | None:
    spec = _spec(model_id)
    size = getattr(spec, "input_size", None) if spec is not None else None
    if size and len(size) == 2 and all(isinstance(v, int) and v > 0 for v in size):
        return int(size[0]), int(size[1])
    return None


def is_available(model_id: str) -> bool:
    """True if the model can be used without a download (downloaded or env override).

    Never raises."""
    if env_override(model_id) is not None:
        return True
    try:
        from ..models.download import is_downloaded
    except Exception:
        return False
    try:
        return bool(is_downloaded(model_id))
    except Exception as e:  # pragma: no cover - defensive
        log.debug("is_downloaded(%s) failed: %s", model_id, e)
        return False


def is_registered(model_id: str) -> bool:
    """True if the model registry knows ``model_id`` (i.e. it can be downloaded)."""
    return _spec(model_id) is not None


# ----------------------------------------------------------------------------- failure memory
#: After a failed download, "auto" methods skip that model for this long (seconds) unless it
#: has become available locally (e.g. downloaded from Settings).  Offline, every attempt
#: costs ~15-60 s of retries, which would otherwise repeat for every job.
DOWNLOAD_RETRY_S = 300.0
#: How long an Auto* instance avoids a backend that failed to load or run (seconds).
FAILURE_TTL_S = 600.0

_dl_lock = threading.Lock()
_dl_failures: dict[str, float] = {}


def _clock() -> float:
    """Monotonic seconds (indirection so tests can fake the clock)."""
    return time.monotonic()


def note_download_failure(model_id: str) -> None:
    """Remember (process-wide) that downloading ``model_id`` just failed."""
    with _dl_lock:
        _dl_failures[model_id] = _clock()


def clear_download_failure(model_id: str) -> None:
    with _dl_lock:
        _dl_failures.pop(model_id, None)


def reset_download_failures() -> None:
    """Forget all remembered download failures (tests, "retry" buttons)."""
    with _dl_lock:
        _dl_failures.clear()


def recent_download_failure(model_id: str) -> bool:
    """True if a download of ``model_id`` failed less than :data:`DOWNLOAD_RETRY_S` ago."""
    with _dl_lock:
        t = _dl_failures.get(model_id)
        if t is None:
            return False
        if _clock() - t > DOWNLOAD_RETRY_S:
            del _dl_failures[model_id]
            return False
        return True


def auto_candidate_ok(model_id: str | None) -> bool:
    """Whether an "auto" chain should try ``model_id`` (None = model-free method).

    Usable when available locally, or when it can be downloaded and no download of it
    failed recently.  Never raises."""
    if model_id is None or is_available(model_id):
        return True
    return is_registered(model_id) and not recent_download_failure(model_id)


class FailureMemo:
    """Thread-safe memory of the backends an Auto* chain should skip for now.

    An entry expires after ``ttl`` seconds, or as soon as a model that was not available
    locally when it failed (i.e. its download failed) becomes available, so a model the
    user downloads later in Settings is picked up without restarting the app."""

    def __init__(self, ttl: float | None = None) -> None:
        self.ttl = FAILURE_TTL_S if ttl is None else float(ttl)
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, str | None, bool]] = {}

    def add(self, name: str, model_id: str | None) -> None:
        was_available = model_id is not None and is_available(model_id)
        with self._lock:
            self._entries[name] = (_clock(), model_id, was_available)

    def discard(self, name: str) -> None:
        with self._lock:
            self._entries.pop(name, None)

    def blocked(self, name: str) -> bool:
        with self._lock:
            entry = self._entries.get(name)
        if entry is None:
            return False
        t, model_id, was_available = entry
        expired = _clock() - t > self.ttl
        if not expired and model_id is not None and not was_available:
            expired = is_available(model_id)  # downloaded since the failure
        if expired:
            self.discard(name)
            return False
        return True

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.blocked(name)


# ----------------------------------------------------------------------------- loading
def open_model(
    model_id: str,
    *,
    path: str | Path | None = None,
    device: str | int = "auto",
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> SessionLike:
    """Load ``model_id`` through the models layer (downloading it on first use).

    With an explicit ``path`` the ONNX file is opened directly (``OrtModel``) honouring
    the registry's ``dml_ok`` flag.  Raises :class:`ModelUnavailable` with a readable
    message on any failure; :class:`Cancelled` propagates.
    """
    title = model_title(model_id)
    try:
        if path is not None:
            p = Path(path)
            if not p.is_file():
                raise ModelUnavailable(f"Model file not found: {p}", hint=_MODEL_HINT)
            from ..models.ort import OrtModel

            return OrtModel(p, device, dml_ok=spec_dml_ok(model_id))
        from ..models.ort import load_model

        try:
            model = load_model(model_id, device=device, progress=progress, cancel=cancel)
        except DownloadError:
            note_download_failure(model_id)
            raise
        clear_download_failure(model_id)
        return model
    except VRPhotoError:
        raise
    except ImportError as e:
        raise ModelUnavailable(
            f"AI model support is not available in this build ({e}).",
            hint="Reinstall VR Photo Studio, or choose a non-AI method.",
        ) from e
    except Exception as e:
        raise ModelUnavailable(f"Could not load the {title} model: {e}", hint=_MODEL_HINT) from e


class LazyModel:
    """Thread-safe, lazily loaded model handle.

    * ``model``: an already-open session-like object (tests, advanced callers);
    * ``model_path``: an ONNX file opened once and owned by this handle;
    * neither: the registry model via ``models.ort.load_model`` on every :meth:`get`.
      That call is cheap once loaded (the models layer caches sessions per device) and
      it never hands out a session the models layer has since closed (e.g. after the
      user deleted / re-downloaded the model in Settings).
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | int = "auto",
        model: SessionLike | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = model
        self._path = Path(model_path) if model_path is not None else None
        self._lock = threading.Lock()
        self._logged = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def get(
        self, *, progress: ProgressReporter | None = None, cancel: CancelToken | None = None
    ) -> SessionLike:
        """Return the session, loading (and downloading) it if needed.

        While another thread holds the handle (e.g. downloading the model) the wait stays
        responsive to ``cancel``."""
        if self._model is not None:
            return self._model
        while not self._lock.acquire(timeout=0.2):
            if cancel is not None:
                cancel.check()
        try:
            if self._model is not None:
                return self._model
            if cancel is not None:
                cancel.check()
            model = open_model(
                self.model_id, path=self._path, device=self.device, progress=progress, cancel=cancel
            )
            if self._path is not None:
                self._model = model
            if not self._logged:
                log.info("Using model %s (%s)", self.model_id, getattr(model, "provider", "?"))
                self._logged = True
            return model
        finally:
            self._lock.release()


_RUN_HINT = "Try Device = CPU in Settings, or choose another method."
_OOM_MARKERS = ("bad allocation", "failed to allocate", "out of memory", "e_outofmemory", "8007000e")


def _is_oom(e: BaseException) -> bool:
    if isinstance(e, MemoryError):
        return True
    text = str(e).lower()
    return any(k in text for k in _OOM_MARKERS)


def run_model(
    model: SessionLike, feeds: dict[str, np.ndarray], *, model_id: str, cancel: CancelToken | None = None
) -> dict[str, np.ndarray]:
    """Run ``model`` and translate unexpected exceptions into :class:`InferenceError`."""
    if cancel is not None:
        cancel.check()
    title = model_title(model_id)
    try:
        outs = model.run(feeds)
    except Cancelled:
        raise
    except VRPhotoError as e:
        if isinstance(e, InferenceError) and _is_oom(e):
            raise InferenceError(
                f"The {title} model ran out of memory.",
                hint="Close other programs, use a smaller photo, or set Device = CPU in Settings.",
            ) from e
        if getattr(e, "hint", None) is None:
            e.hint = _RUN_HINT
        raise
    except Exception as e:
        if _is_oom(e):
            raise InferenceError(
                f"The {title} model ran out of memory.",
                hint="Close other programs, use a smaller photo, or set Device = CPU in Settings.",
            ) from e
        raise InferenceError(f"The {title} model failed: {e}", hint=_RUN_HINT) from e
    if not outs:
        raise InferenceError(f"The {title} model returned no output.", hint=_RUN_HINT)
    if cancel is not None:
        cancel.check()
    return outs


# ----------------------------------------------------------------------------- tensors
def static_hw(model: SessionLike, index: int, default: tuple[int, int]) -> tuple[int, int]:
    """Static (H, W) of input ``index`` if the model declares one, else ``default``."""
    try:
        shape = model.input_shapes()[index]
    except Exception:
        return default
    if len(shape) == 4 and all(isinstance(v, int) and v > 0 for v in shape[2:]):
        return int(shape[2]), int(shape[3])
    return default


def _channels(shape: Sequence[Any]) -> int | None:
    if len(shape) == 4 and isinstance(shape[1], int):
        return int(shape[1])
    return None


def image_mask_inputs(model: SessionLike) -> tuple[str, str]:
    """Identify (image_input_name, mask_input_name) of an inpainting model.

    Uses the declared channel count (3 vs 1) first, then names containing
    ``mask`` / ``image``, then input order (image first)."""
    names = list(model.input_names())
    if len(names) < 2:
        raise InferenceError(f"Inpainting model has {len(names)} input(s); expected image and mask.")
    try:
        shapes = list(model.input_shapes())
    except Exception:
        shapes = []
    if len(shapes) == len(names):
        ch = [_channels(s) for s in shapes]
        if ch.count(3) == 1 and ch.count(1) == 1:
            return names[ch.index(3)], names[ch.index(1)]
    lowered = [n.lower() for n in names]
    mask_idx = [i for i, n in enumerate(lowered) if "mask" in n]
    if len(mask_idx) == 1:
        others = [i for i in range(len(names)) if i != mask_idx[0]]
        img_idx = [i for i in others if "image" in lowered[i] or "img" in lowered[i]] or others
        return names[img_idx[0]], names[mask_idx[0]]
    return names[0], names[1]


def pick_output(outs: dict[str, np.ndarray], preferred: Sequence[str] = ()) -> np.ndarray:
    """Output tensor by preferred name, else the first one."""
    for name in preferred:
        if name in outs:
            return np.asarray(outs[name])
    return np.asarray(next(iter(outs.values())))


def as_rgb(img: np.ndarray, what: str = "image") -> np.ndarray:
    """Validate/convert to ``HxWx3 uint8`` (grey and RGBA are accepted)."""
    if not isinstance(img, np.ndarray):
        raise InputError(f"The {what} must be a numpy array, got {type(img).__name__}.")
    if img.dtype != np.uint8:
        raise InputError(f"The {what} must be 8-bit (uint8), got {img.dtype}.")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3:
        raise InputError(f"The {what} must be an RGB image (HxWx3), got shape {img.shape}.")
    if img.shape[0] < 1 or img.shape[1] < 1:
        raise InputError(f"The {what} is empty.")
    return np.ascontiguousarray(img)


def as_bool_mask(mask: np.ndarray, shape: tuple[int, int], what: str = "mask") -> np.ndarray:
    """Validate/convert a mask to ``HxW bool`` (non-zero = True)."""
    if not isinstance(mask, np.ndarray):
        raise InputError(f"The {what} must be a numpy array, got {type(mask).__name__}.")
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2 or mask.shape != tuple(shape):
        raise InputError(f"The {what} shape {mask.shape} does not match the image ({shape[0]}x{shape[1]}).")
    if mask.dtype == np.bool_:
        return mask
    if np.issubdtype(mask.dtype, np.floating):
        return np.nan_to_num(mask, nan=0.0) > 0.5
    return mask != 0


def resize_to(img: np.ndarray, w: int, h: int, *, smooth: bool = False) -> np.ndarray:
    """Resize with area filtering when shrinking and cubic (or linear when ``smooth``)
    when enlarging.  Preserves dtype (uint8 results are saturated)."""
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    if w <= iw and h <= ih:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR if smooth else cv2.INTER_CUBIC
    out = cv2.resize(img, (int(w), int(h)), interpolation=interp)
    if img.ndim == 3 and out.ndim == 2:
        out = out[..., None]
    return out


def sub_progress(progress: ProgressReporter | None, a: float, b: float) -> ProgressReporter | None:
    return progress.sub(a, b) if progress is not None else None


def report(progress: ProgressReporter | None, frac: float, msg: str = "") -> None:
    if progress is not None:
        progress(frac, msg)
