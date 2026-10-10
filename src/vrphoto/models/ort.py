"""ONNX Runtime wrapper: provider selection, transparent CPU fallback, fp16 handling.

Provider order: ``CUDAExecutionProvider`` (only present with onnxruntime-gpu) or
``DmlExecutionProvider`` (onnxruntime-directml, any DirectX 12 GPU) followed by
``CPUExecutionProvider``.  DirectML sessions use ``enable_mem_pattern=False`` and
sequential execution as required by the DML EP.  If creating the GPU session or running
it fails, the model is transparently re-created on the CPU and that choice is *sticky*
for the model file for the rest of the process (a driver that crashed once will crash
again).  Models whose registry entry says ``dml_ok=False`` never use DirectML.

``onnxruntime`` is imported lazily so importing this module never requires it.
"""

from __future__ import annotations

import contextlib
import io
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import get_settings
from ..core.errors import Cancelled, DownloadError, InferenceError, ModelUnavailable
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter

log = get_logger(__name__)


_stdout_lock = threading.Lock()


def quiet_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` while capturing anything it prints to stdout (onnxruntime prints
    "EP Error ... Falling back" banners with ``print()``), logging it instead, so the
    CLI's ``--json`` output stays machine-readable."""
    buf = io.StringIO()
    with _stdout_lock, contextlib.redirect_stdout(buf):
        try:
            return fn(*args, **kwargs)
        finally:
            text = buf.getvalue().strip()
            if text:
                log.info("onnxruntime: %s", " ".join(text.split()))

CPU = "CPUExecutionProvider"
DML = "DmlExecutionProvider"
CUDA = "CUDAExecutionProvider"
GPU_PROVIDERS = (CUDA, DML)

ProviderChain = list[tuple[str, dict[str, str]]]

_ORT_TYPES: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(bool)": np.bool_,
}

_sticky_lock = threading.Lock()
_cpu_sticky: set[str] = set()
_best_adapter: list[int | None] = []  # memoised DXGI choice ([] = not computed)


def _import_ort() -> Any:
    try:
        import onnxruntime as ort  # noqa: PLC0415 - heavy optional import
    except Exception as e:  # ImportError, DLL load failure on Windows, ...
        raise ModelUnavailable(
            f"ONNX Runtime could not be loaded: {e}",
            hint="Reinstall VR Photo Studio; on Windows also install the latest Visual C++ runtime.",
        ) from e
    return ort


def available_providers() -> list[str]:
    """Execution providers compiled into the installed onnxruntime ([] if unavailable)."""
    try:
        ort = _import_ort()
        return list(ort.get_available_providers())
    except Exception as e:
        log.warning("onnxruntime unavailable: %s", e)
        return []


# --------------------------------------------------------------------------- device parsing
def parse_device(device: str | int | None = "auto") -> tuple[str, int | None]:
    """Normalise a device spec to ``(kind, index)``.

    ``kind`` is ``"cpu"``, ``"gpu"`` or ``"auto"``; ``index`` is the GPU adapter index
    (the DirectML ``device_id`` / CUDA ordinal) or None for "best available".  Accepts
    ints, ``"auto"``, ``"cpu"``, ``"gpu"``, ``"dml"``, ``"cuda"``, ``"1"``, ``"dml:1"``,
    ``"cuda:0"``, ``"gpu:1"``.  ``"auto"`` consults ``Settings.device``.
    """
    if device is None:
        device = "auto"
    if isinstance(device, bool):
        raise TypeError("device must be a str or int")
    if isinstance(device, int):
        return ("gpu", device) if device >= 0 else ("cpu", None)
    d = str(device).strip().lower()
    if d in ("", "auto"):
        try:
            pref = str(get_settings().device).strip().lower()
        except Exception:  # pragma: no cover - settings must never break inference
            pref = "auto"
        if pref in ("", "auto"):
            return "auto", None
        return parse_device(pref)
    if d == "cpu":
        return "cpu", None
    if d in ("gpu", "dml", "cuda", "directml"):
        return "gpu", None
    kind, sep, idx = d.partition(":")
    if sep and kind in ("gpu", "dml", "cuda", "directml") and idx.isdigit():
        return "gpu", int(idx)
    if d.isdigit():
        return "gpu", int(d)
    log.warning("Unknown device '%s'; using automatic selection", device)
    return "auto", None


def _best_dml_adapter() -> int | None:
    """DXGI index of the hardware adapter with the most dedicated VRAM (None if unknown)."""
    if _best_adapter:
        return _best_adapter[0]
    choice: int | None = None
    try:
        from .gpu import list_devices

        devs = [d for d in list_devices() if d.vendor != "cpu"]
        if devs:
            choice = max(devs, key=lambda d: (d.vram_mb or 0, -d.index)).index
    except Exception as e:  # pragma: no cover - defensive
        log.debug("Adapter enumeration failed: %s", e)
    _best_adapter.append(choice)
    return choice


def _cuda_ordinal(idx: int | None) -> int:
    """CUDA device ordinal for a device index chosen in Settings.

    Settings store the DXGI adapter index (what DirectML uses).  On a machine with an
    integrated GPU listed first, the NVIDIA card is DXGI adapter 1 but CUDA device 0, so
    the index is mapped through the adapter list (NVIDIA adapters in DXGI order).  Without
    adapter information (Linux) the index is used as is.
    """
    if idx is None:
        return 0
    try:
        from .gpu import list_devices

        devs = list_devices()
    except Exception:  # pragma: no cover - defensive
        devs = []
    if not devs:
        return idx
    nvidia = [d.index for d in devs if d.vendor == "nvidia"]
    if idx in nvidia:
        return nvidia.index(idx)
    log.warning("Selected adapter %d is not an NVIDIA GPU; CUDA will use GPU 0", idx)
    return 0


def select_providers(
    device: str | int | None = "auto", *, dml_ok: bool = True, available: list[str] | None = None
) -> ProviderChain:
    """Ordered ``[(provider, options), ...]`` to try for ``device`` (always ends with CPU)."""
    avail = available_providers() if available is None else list(available)
    kind, idx = parse_device(device)
    if kind == "cpu":
        return [(CPU, {})]
    chain: ProviderChain = []
    if CUDA in avail:
        chain.append((CUDA, {"device_id": str(_cuda_ordinal(idx))}))
    elif DML in avail and dml_ok:
        if idx is None:
            idx = _best_dml_adapter()
        if idx is not None:
            chain.append((DML, {"device_id": str(idx)}))
        else:  # let DirectML (DXCore) pick the high-performance GPU itself
            chain.append((DML, {"performance_preference": "high_performance", "device_filter": "gpu"}))
    chain.append((CPU, {}))
    return chain


def _key(path: Path) -> str:
    try:
        return str(path.resolve()).lower()
    except OSError:  # pragma: no cover
        return str(path).lower()


def _mark_cpu(path: Path) -> None:
    with _sticky_lock:
        _cpu_sticky.add(_key(path))


def _unmark_cpu(path: Path) -> None:
    with _sticky_lock:
        _cpu_sticky.discard(_key(path))


def _is_cpu_sticky(path: Path) -> bool:
    with _sticky_lock:
        return _key(path) in _cpu_sticky


def reset_cpu_fallbacks() -> None:
    """Forget sticky CPU fallbacks (e.g. after the user changes GPU in Settings)."""
    with _sticky_lock:
        _cpu_sticky.clear()
    _best_adapter.clear()


# --------------------------------------------------------------------------- OrtModel
class OrtModel:
    """A thread-safe ONNX Runtime session with automatic GPU -> CPU fallback.

    ``device``: see :func:`parse_device`.  ``fp16``: the file is a half-precision
    variant (outputs are checked for NaN/inf and, if found, the model is re-run with
    ``fallback_path`` (the fp32 file) and/or on the CPU).  Inputs are cast to the dtype
    the graph expects, so callers always feed float32; float16 outputs come back as
    float32.  ``dml_ok=False`` keeps the model off DirectML.
    """

    def __init__(
        self,
        path: str | Path,
        device: str | int = "auto",
        *,
        fp16: bool = False,
        dml_ok: bool = True,
        fallback_path: str | Path | Callable[[], Path] | None = None,
        title: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.title = title or self.path.name
        if not self.path.is_file():
            raise ModelUnavailable(
                f"Model file not found: {self.path}",
                hint="Download the model again from Settings > Models.",
            )
        self.device = device
        self.fp16 = fp16
        self.dml_ok = dml_ok
        self._fallback_path = fallback_path
        self._nan_inherent = False  # NaN/inf also on CPU fp32 -> stop treating it as a GPU fault
        self._lock = threading.RLock()
        self._sess: Any = None
        self.provider: str = CPU
        self._inputs: list[Any] = []
        self._outputs: list[Any] = []
        self._create()

    # ------------------------------------------------------------------ session handling
    def _new_session(self, chain: ProviderChain) -> Any:
        ort = _import_ort()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if chain[0][0] == DML:
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = quiet_call(
            ort.InferenceSession,
            str(self.path),
            sess_options=so,
            providers=[p for p, _ in chain],
            provider_options=[o for _, o in chain],
        )
        disable = getattr(sess, "disable_fallback", None)
        if callable(disable):
            disable()  # failures must reach us so we can log + make the CPU choice sticky
        return sess

    def _install(self, sess: Any) -> None:
        self._sess = sess
        provs = list(sess.get_providers()) or [CPU]
        self.provider = provs[0]
        self._inputs = list(sess.get_inputs())
        self._outputs = list(sess.get_outputs())

    def _create(self, force_cpu: bool = False) -> None:
        chain = (
            [(CPU, {})]
            if force_cpu or _is_cpu_sticky(self.path)
            else select_providers(self.device, dml_ok=self.dml_ok)
        )
        if chain[0][0] != CPU:
            try:
                sess = self._new_session(chain)
                self._install(sess)
                if self.provider == CPU:
                    log.warning("%s: %s unavailable, running on CPU", self.path.name, chain[0][0])
                    _mark_cpu(self.path)
                else:
                    log.info("%s: using %s %s", self.path.name, self.provider, chain[0][1])
                return
            except Exception as e:
                log.warning(
                    "%s: creating a %s session failed (%s); falling back to CPU",
                    self.path.name,
                    chain[0][0],
                    e,
                )
                _mark_cpu(self.path)
        try:
            sess = self._new_session([(CPU, {})])
        except ModelUnavailable:
            raise
        except Exception as e:
            raise ModelUnavailable(
                f"Cannot load the {self.title} model ({self.path.name}): {e}",
                hint="The file may be corrupt or incomplete; delete the model in Settings > Models "
                "so it is downloaded again.",
            ) from e
        self._install(sess)
        log.info("%s: using CPU", self.path.name)

    def _switch_to_cpu(self, reason: str) -> None:
        log.warning("%s: inference failed on %s (%s); retrying on CPU", self.path.name, self.provider, reason)
        _mark_cpu(self.path)
        if self.fp16 and self._fallback_path is not None:
            # half precision is a GPU optimisation: on the CPU use the fp32 file instead
            fb, self._fallback_path = self._fallback_path, None
            try:
                fb_path = Path(fb() if callable(fb) else fb)
                if not fb_path.is_file():
                    raise FileNotFoundError(fb_path)
                self.path, self.fp16 = fb_path, False
            except Exception as e:
                log.warning("fp32 model unavailable (%s); running the fp16 model on the CPU", e)
        self._create(force_cpu=True)

    # ------------------------------------------------------------------ introspection
    def input_names(self) -> list[str]:
        """Graph input names in declaration order."""
        return [i.name for i in self._inputs]

    def input_shapes(self) -> list[list[int | str | None]]:
        """Input shapes; symbolic dims are strings (or None when unnamed)."""
        return [list(i.shape) if i.shape is not None else [] for i in self._inputs]

    def input_types(self) -> list[str]:
        """ONNX type strings of the inputs, e.g. ``"tensor(float)"``."""
        return [str(i.type) for i in self._inputs]

    def output_names(self) -> list[str]:
        """Graph output names in declaration order."""
        return [o.name for o in self._outputs]

    def output_shapes(self) -> list[list[int | str | None]]:
        return [list(o.shape) if o.shape is not None else [] for o in self._outputs]

    @property
    def has_fp16_io(self) -> bool:
        return any("float16" in str(t.type) for t in (*self._inputs, *self._outputs))

    # ------------------------------------------------------------------ inference
    def _prepare(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        types = {i.name: str(i.type) for i in self._inputs}
        unknown = set(feeds) - set(types)
        if unknown:
            raise ValueError(
                f"Unknown input(s) {sorted(unknown)} for {self.path.name}; expected {sorted(types)}"
            )
        out: dict[str, np.ndarray] = {}
        for name, t in types.items():
            if name not in feeds:
                raise ValueError(f"Missing input '{name}' for {self.path.name}")
            arr = np.asarray(feeds[name])
            want = _ORT_TYPES.get(t)
            if want is not None and arr.dtype != want:
                arr = arr.astype(want)
            out[name] = np.ascontiguousarray(arr)
        return out

    def _run_raw(
        self,
        feeds: dict[str, np.ndarray],
        output_names: list[str] | None,
        cancel: CancelToken | None = None,
    ) -> dict[str, np.ndarray]:
        names = list(output_names) if output_names else self.output_names()
        if cancel is None:
            res = self._sess.run(names, feeds)
        else:
            res = self._run_cancellable(names, feeds, cancel)
        out: dict[str, np.ndarray] = {}
        for n, v in zip(names, res):
            a = np.asarray(v)
            if a.dtype == np.float16:
                a = a.astype(np.float32)
            out[n] = a
        return out

    def _run_cancellable(self, names: list[str], feeds: dict[str, np.ndarray], cancel: CancelToken) -> Any:
        """``session.run`` that stops between graph nodes when ``cancel`` fires.

        ORT checks ``RunOptions.terminate`` before each node, so a long CPU inference
        (seconds to a minute for the large models) stops promptly instead of finishing.
        """
        cancel.check()
        ro_cls = getattr(_import_ort(), "RunOptions", None)
        if ro_cls is None:  # pragma: no cover - very old / stub onnxruntime
            return self._sess.run(names, feeds)
        ro = ro_cls()
        done = threading.Event()

        def watch() -> None:
            while not done.wait(0.1):
                if cancel.cancelled:
                    ro.terminate = True
                    return

        threading.Thread(target=watch, name="vrphoto-ort-cancel", daemon=True).start()
        try:
            res = self._sess.run(names, feeds, ro)
        except Exception:
            if cancel.cancelled:
                raise Cancelled("Operation cancelled") from None
            raise
        finally:
            done.set()
        cancel.check()
        return res

    @property
    def closed(self) -> bool:
        """True after :meth:`close` (e.g. the model was unloaded or deleted in Settings)."""
        return self._sess is None

    def run(
        self,
        feeds: dict[str, np.ndarray],
        output_names: list[str] | None = None,
        *,
        cancel: CancelToken | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the model.  Thread-safe; retries once on CPU if the GPU provider fails.

        ``cancel`` (optional extension of the contract) aborts a running inference
        between graph nodes and raises :class:`Cancelled`.  Raises
        :class:`InferenceError` if the model fails on every provider and
        :class:`ModelUnavailable` if the model was closed (load it again with
        :func:`load_model`).
        """
        with self._lock:
            if self._sess is None:
                raise ModelUnavailable(
                    f"The {self.title} model was unloaded",
                    hint="Run the job again; the model is reloaded automatically.",
                )
            prepared = self._prepare(feeds)
            try:
                outs = self._run_raw(prepared, output_names, cancel)
            except Cancelled:
                raise
            except Exception as e:
                if self.provider == CPU:
                    raise _inference_error(self.title, e) from e
                self._switch_to_cpu(str(e))
                prepared = self._prepare(feeds)
                try:
                    outs = self._run_raw(prepared, output_names, cancel)
                except Cancelled:
                    raise
                except Exception as e2:
                    raise _inference_error(self.title, e2, both=True) from e2
            suspicious = self.fp16 or self.provider != CPU or self.has_fp16_io
            if suspicious and not self._nan_inherent and not _all_finite(outs):
                outs = self._recover_nonfinite(feeds, output_names, outs)
            return outs

    def _recover_nonfinite(
        self, feeds: dict[str, np.ndarray], output_names: list[str] | None, outs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """NaN/inf outputs: try the fp32 variant, then the CPU; if the CPU (fp32) output is
        non-finite too, the values are inherent to the model, so the GPU is restored."""
        if self._fallback_path is not None:
            fb = self._fallback_path
            self._fallback_path = None
            old_path, old_fp16 = self.path, self.fp16
            try:
                fb_path = Path(fb() if callable(fb) else fb)
                log.warning(
                    "%s produced NaN/inf; switching to the fp32 model %s", self.path.name, fb_path.name
                )
                self.path, self.fp16 = fb_path, False
                self._create()
                outs = self._run_raw(self._prepare(feeds), output_names)
                if _all_finite(outs):
                    return outs
            except Exception as e:
                log.warning("fp32 fallback for %s failed: %s", old_path.name, e)
                self.path, self.fp16 = old_path, old_fp16
                self._create()
        if self.provider != CPU:
            gpu = self.provider
            self._switch_to_cpu("non-finite outputs")
            try:
                cpu_outs = self._run_raw(self._prepare(feeds), output_names)
            except Exception as e:
                raise _inference_error(self.title, e) from e
            if _all_finite(cpu_outs):
                return cpu_outs
            log.info("%s outputs NaN/inf on CPU too (inherent); returning to %s", self.path.name, gpu)
            _unmark_cpu(self.path)
            self._create()
            outs = cpu_outs
        else:
            log.warning("%s produces NaN/inf values on %s", self.path.name, self.provider)
        self._nan_inherent = True
        return outs

    def close(self) -> None:
        """Release the session (the object must not be used afterwards)."""
        with self._lock:
            self._sess = None

    def __repr__(self) -> str:
        return f"OrtModel({self.path.name!r}, provider={self.provider!r})"


_OOM_MARKERS = ("bad allocation", "failed to allocate", "out of memory", "e_outofmemory", "8007000e")


def _inference_error(title: str, e: BaseException, *, both: bool = False) -> InferenceError:
    """User-facing InferenceError (with a hint for the common out-of-memory case)."""
    text = str(e).strip()
    detail = " ".join(text.split())[:1000] if text else type(e).__name__
    where = " on the GPU and on the CPU" if both else ""
    low = text.lower()
    if isinstance(e, MemoryError) or any(k in low for k in _OOM_MARKERS):
        hint = (
            "The computer ran out of memory. Close other programs, or lower the maximum "
            "processing size in Settings, and try again."
        )
    else:
        hint = "Try again; if it keeps failing, choose CPU as the device in Settings or delete and re-download the model."
    return InferenceError(f"The {title} model failed{where}: {detail}", hint=hint)


def _all_finite(outs: dict[str, np.ndarray]) -> bool:
    for a in outs.values():
        if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all():
            return False
    return True


# --------------------------------------------------------------------------- cache
_cache_lock = threading.Lock()
_cache: dict[tuple[str, str, str], OrtModel] = {}
_key_locks: dict[tuple[str, str], threading.Lock] = {}


def _will_use_gpu(device: str | int, dml_ok: bool) -> bool:
    return select_providers(device, dml_ok=dml_ok)[0][0] != CPU


def load_model(
    model_id: str,
    *,
    device: str | int = "auto",
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> OrtModel:
    """Download (if needed) and open a registry model; sessions are cached per device.

    With ``Settings.use_fp16`` and a GPU provider, the model's fp16 variant is used when
    the registry lists one (falling back to fp32 automatically on NaN output).
    """
    from .download import ensure_model, env_override
    from .registry import get_spec

    spec = get_spec(model_id)
    try:
        want_fp16 = bool(get_settings().use_fp16)
    except Exception:  # pragma: no cover
        want_fp16 = False
    use_fp16 = (
        want_fp16
        and spec.fp16_file is not None
        and env_override(model_id) is None
        and _will_use_gpu(device, spec.dml_ok)
    )
    variant = "fp16" if use_fp16 else "default"
    dev_key = str(device).lower()
    if parse_device(device)[0] == "auto" or dev_key == "auto":
        dev_key += f"|{get_settings().device}"
    with _cache_lock:
        klock = _key_locks.setdefault((model_id, dev_key), threading.Lock())
    with klock:
        try:
            path = ensure_model(model_id, progress=progress, cancel=cancel, variant=variant)
        except DownloadError as e:
            if not use_fp16:
                raise
            # the optional half-precision file is missing upstream / failed: use fp32
            log.warning("fp16 variant of %s unavailable (%s); using the fp32 model", model_id, e)
            use_fp16 = False
            path = ensure_model(model_id, progress=progress, cancel=cancel)
        ckey = (model_id, dev_key, _key(path))
        with _cache_lock:
            m = _cache.get(ckey)
        if m is not None:
            return m
        if cancel is not None:
            cancel.check()
        if progress is not None:
            progress(1.0, f"Loading {spec.title}")
        fallback: Callable[[], Path] | None = (lambda: ensure_model(model_id)) if use_fp16 else None
        m = OrtModel(
            path, device, fp16=use_fp16, dml_ok=spec.dml_ok, fallback_path=fallback, title=spec.title
        )
        with _cache_lock:
            _cache[ckey] = m
        return m


def unload_model(model_id: str | None = None) -> None:
    """Drop cached sessions for ``model_id`` (or all models) to free memory / unlock files."""
    with _cache_lock:
        keys = [k for k in _cache if model_id is None or k[0] == model_id]
        dropped = [_cache.pop(k) for k in keys]
    for m in dropped:  # close outside the cache lock: close() waits for a running inference
        m.close()


def cached_models() -> list[tuple[str, str, str]]:
    """Keys ``(model_id, device, provider)`` of the currently cached sessions."""
    with _cache_lock:
        return [(k[0], k[1], m.provider) for k, m in _cache.items()]


__all__ = [
    "OrtModel",
    "available_providers",
    "select_providers",
    "parse_device",
    "load_model",
    "unload_model",
    "cached_models",
    "reset_cpu_fallbacks",
    "CPU",
    "DML",
    "CUDA",
]
