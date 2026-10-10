"""Depth Anything 3 (mono) through a community ONNX export - experimental.

The export's exact I/O is not standardised, so everything is read from the session:

* input: the first graph input, rank 4 ``[B,3,H,W]`` or rank 5 ``[B,N,3,H,W]`` (DA3's
  multi-view layout, N = 1); fixed spatial size is honoured, otherwise the long side is
  resized to 504 (DA3's ``upper_bound_resize``) rounded to multiples of 14; float16
  inputs are handled by :class:`~vrphoto.models.ort.OrtModel`.  ImageNet normalisation.
* output: the first output whose name contains ``depth`` / ``disp`` (else the first
  output).  Unlike DA2, DA3-Mono predicts *depth* (README: "Unlike disparity-based
  models (e.g. Depth Anything 2), it directly predicts depth"), so by default the output
  is inverted; outputs named ``*disp*`` are used as disparity directly.  An optional
  ``sky`` output marks pixels as far.
"""

from __future__ import annotations

import threading
from typing import Any, Literal

import numpy as np

from ..core.log import get_logger
from ..core.types import RGB, CancelToken, DepthResult, ProgressReporter
from .base import as_rgb, check, reporter
from .da2 import PATCH, preprocess, round_multiple, squeeze_map, static_hw
from .post import finalize_disparity

log = get_logger(__name__)

PROCESS_LONG = 504
OutputKind = Literal["auto", "depth", "disparity"]


def da3_input_size(h: int, w: int, long_side: int = PROCESS_LONG) -> tuple[int, int]:
    """Long side -> ``long_side`` keeping aspect, both sides multiples of 14."""
    s = long_side / max(h, w)
    return round_multiple(h * s, PATCH), round_multiple(w * s, PATCH)


SKY_THRESHOLD = 0.3  # DA3 ``compute_sky_mask``: non-sky = sky_prediction < 0.3
MIN_SKY_PIXELS = 10


def sky_mask(sky_pred: np.ndarray) -> np.ndarray | None:
    """Boolean sky mask from DA3's sky output, as upstream ``_process_mono_sky_estimation``.

    Returns None (no sky handling) when either sky or non-sky covers at most
    :data:`MIN_SKY_PIXELS` pixels, like the reference implementation.
    """
    s = np.nan_to_num(np.asarray(sky_pred, np.float32), nan=0.0)
    sky = s >= SKY_THRESHOLD
    n_sky = int(sky.sum())
    if n_sky <= MIN_SKY_PIXELS or sky.size - n_sky <= MIN_SKY_PIXELS:
        return None
    return sky


def depth_to_disparity(depth: np.ndarray, sky: np.ndarray | None = None) -> np.ndarray:
    """Invert a relative depth map robustly (non-positive / sky pixels become far)."""
    d = np.asarray(depth, np.float32)
    ok = np.isfinite(d) & (d > 0)
    if sky is not None:
        ok &= ~sky
    if not ok.any():
        return np.zeros_like(d)
    lo = float(np.percentile(d[ok], 0.5))
    hi = float(np.percentile(d[ok], 99.5))
    eps = max(lo * 1e-3, 1e-6)
    inv = np.where(ok, 1.0 / np.clip(d, max(lo, eps), max(hi, lo + eps)), 0.0)
    far = 1.0 / max(hi, lo + eps)
    return np.where(ok, inv, far * 0.5).astype(np.float32)


class DA3Estimator:
    """Generic Depth Anything 3 mono estimator (see module docstring)."""

    def __init__(
        self,
        model_id: str = "da3-mono-large",
        device: str | int = "auto",
        *,
        output_kind: OutputKind = "auto",
        process_long: int = PROCESS_LONG,
    ) -> None:
        self.id = model_id
        self.device = device
        self.output_kind = output_kind
        self.process_long = int(process_long)
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self, progress: ProgressReporter, cancel: CancelToken | None) -> Any:
        with self._lock:
            # reload when the cached session was closed (model deleted / re-downloaded or
            # sessions unloaded in Settings); load_model re-uses its own session cache
            if self._model is None or getattr(self._model, "closed", False) is True:
                from ..models.ort import load_model

                self._model = load_model(self.id, device=self.device, progress=progress, cancel=cancel)
            return self._model

    def net_size(self, h: int, w: int, model: Any) -> tuple[int, int]:
        fixed = static_hw(model.input_shapes()[0])
        if fixed is not None:
            return fixed
        return da3_input_size(h, w, self.process_long)

    @staticmethod
    def pick_output(names: list[str]) -> str:
        for key in ("disp", "depth"):
            for n in names:
                if key in n.lower():
                    return n
        return names[0]

    def kind_for(self, output_name: str) -> str:
        if self.output_kind != "auto":
            return self.output_kind
        return "disparity" if "disp" in output_name.lower() else "depth"

    def estimate(
        self, img: RGB, *, progress: ProgressReporter | None = None, cancel: CancelToken | None = None
    ) -> DepthResult:
        """Relative disparity in [0, 1] (1 = near) at the input resolution."""
        rep = reporter(progress)
        rgb = as_rgb(img)
        h, w = rgb.shape[:2]
        model = self._load(rep.sub(0.0, 0.5), cancel)
        check(cancel)
        rep(0.5, "Estimating depth (DA3)")
        th, tw = self.net_size(h, w, model)
        x = preprocess(rgb, (th, tw))
        shape = model.input_shapes()[0]
        if len(shape) == 5:
            x = x[:, None]  # [B, N=1, 3, H, W]
        names = model.output_names()
        out_name = self.pick_output(names)
        wanted = [out_name] + [n for n in names if "sky" in n.lower() and n != out_name]
        outs = model.run({model.input_names()[0]: x}, wanted, cancel=cancel)
        pred = squeeze_map(outs[out_name])
        sky = None
        for n in wanted[1:]:
            s = squeeze_map(outs[n])
            if s.shape == pred.shape:
                sky = sky_mask(s)
        kind = self.kind_for(out_name)
        rel = pred if kind == "disparity" else depth_to_disparity(pred, sky)
        if kind == "disparity" and sky is not None:
            rel = np.where(sky, float(np.nanmin(rel)), rel)
        check(cancel)
        rep(0.85, "Refining depth")
        disp, raw = finalize_disparity(rel, rgb)
        rep(1.0, "Depth ready")
        return DepthResult(disparity=disp, model_id=self.id, raw=raw)


__all__ = ["DA3Estimator", "da3_input_size", "depth_to_disparity"]
