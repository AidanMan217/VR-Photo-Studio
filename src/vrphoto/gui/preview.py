"""Preview computations (no Qt): depth, quick stereo/anaglyph, wiggle and parallax views.

Everything here runs in worker threads and works on small images (about 640 px) so a
preview takes well under a second once the depth map is known.  Depth maps are cached
per (file, modification time, depth model); the layered image used for the parallax
and stereo previews is cached per (depth key, look-behind).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter

log = get_logger(__name__)

PREVIEW_LONG = 640  # long side of the stereo / wiggle / parallax previews
DEPTH_LONG = 1024  # long side of the image the depth preview is computed on
SOURCE_LONG = 1600  # long side of the "Original" preview
PARALLAX_BASELINE_M = 0.08  # look-behind band prepared for the parallax preview


class LRUCache:
    """Tiny thread-safe LRU cache."""

    def __init__(self, maxsize: int = 16) -> None:
        self.maxsize = maxsize
        self._d: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any:
        with self._lock:
            if key not in self._d:
                return None
            self._d.move_to_end(key)
            return self._d[key]

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._d

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)


def file_key(path: str | Path) -> tuple[str, float]:
    p = Path(path)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (str(p.resolve()), mtime)


@dataclass
class DepthPreview:
    image: np.ndarray  # RGB at <= DEPTH_LONG
    disparity: np.ndarray  # float32 [0,1], same size
    model_id: str
    hfov_deg: float | None  # from the model (MoGe) or EXIF, else None


EstimatorFactory = Callable[[str, str], Any]


def _default_estimator_factory(name: str, device: str) -> Any:
    from ..depth.base import get_estimator

    return get_estimator(name, device)


class PreviewEngine:
    """Computes and caches previews.  Thread-safe; call from worker threads."""

    def __init__(self, estimator_factory: EstimatorFactory | None = None) -> None:
        self.estimator_factory = estimator_factory or _default_estimator_factory
        self.sources = LRUCache(8)
        self.depths = LRUCache(16)
        self.layers = LRUCache(8)
        self._est_lock = threading.Lock()
        self._estimators: dict[tuple[str, str], Any] = {}

    # ---------------------------------------------------------------- sources
    def source(self, path: str | Path, *, cancel: CancelToken | None = None) -> np.ndarray:
        """The input as RGB, long side <= SOURCE_LONG (photo or first video frame)."""
        key = file_key(path)
        img = self.sources.get(key)
        if img is None:
            from .util import load_preview_source

            img = load_preview_source(path, SOURCE_LONG)
            self.sources.put(key, img)
        if cancel is not None:
            cancel.check()
        return img

    # ---------------------------------------------------------------- depth
    def _estimator(self, name: str, device: str) -> Any:
        k = (name, device)
        with self._est_lock:
            est = self._estimators.get(k)
            if est is None:
                est = self.estimator_factory(name, device)
                self._estimators[k] = est
            return est

    def depth_key(self, path: str | Path, model: str) -> tuple:
        return (*file_key(path), model or "auto")

    def cached_depth(self, path: str | Path, model: str) -> DepthPreview | None:
        return self.depths.get(self.depth_key(path, model))

    def depth(
        self,
        path: str | Path,
        model: str = "auto",
        device: str = "auto",
        *,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> DepthPreview:
        """Depth preview for ``path`` with depth model ``model`` (cached)."""
        key = self.depth_key(path, model)
        hit = self.depths.get(key)
        if hit is not None:
            return hit
        rep = progress or ProgressReporter.null()
        rep(0.0, "Loading photo")
        from ..core.imaging import limit_long

        src = self.source(path, cancel=cancel)
        img = np.ascontiguousarray(limit_long(src, DEPTH_LONG)[0])
        if cancel is not None:
            cancel.check()
        est = self._estimator(model or "auto", device or "auto")
        rep(0.05, "Estimating depth")
        res = est.estimate(img, progress=rep.sub(0.05, 1.0), cancel=cancel)
        disp = np.clip(np.nan_to_num(res.disparity.astype(np.float32)), 0.0, 1.0)
        if disp.shape[:2] != img.shape[:2]:
            import cv2

            disp = cv2.resize(disp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        hfov = None
        if getattr(res, "intrinsics", None) is not None:
            hfov = float(res.intrinsics.hfov_deg)
        else:
            try:
                from ..core.imaging import exif_hfov

                hfov = exif_hfov(path)
            except Exception:
                hfov = None
        dp = DepthPreview(img, disp, str(getattr(res, "model_id", model)), hfov)
        self.depths.put(key, dp)
        rep(1.0, "Depth ready")
        return dp

    @staticmethod
    def colorized(dp: DepthPreview) -> np.ndarray:
        from ..core.imaging import colorize_depth

        return colorize_depth(dp.disparity)

    # ---------------------------------------------------------------- layered image
    def layered(
        self,
        dp: DepthPreview,
        *,
        near_m: float = 1.2,
        far_m: float = 15.0,
        hfov_deg: float | None = None,
        look_behind: bool = True,
        cancel: CancelToken | None = None,
    ) -> Any:
        """Low-res :class:`LayeredImage` (long side PREVIEW_LONG) for stereo/parallax previews."""
        hf = hfov_deg if hfov_deg else dp.hfov_deg
        key = (id(dp), round(near_m, 3), round(far_m, 3), hf, look_behind)
        hit = self.layers.get(key)
        if hit is not None and hit[0] is dp:
            return hit[1]
        import cv2

        from ..core.imaging import limit_long
        from ..core.types import DepthScale
        from ..geometry.camera import estimate_intrinsics
        from ..geometry.layers import build_layers

        img, s = limit_long(dp.image, PREVIEW_LONG)
        img = np.ascontiguousarray(img)
        disp = dp.disparity
        if s != 1.0:
            disp = cv2.resize(disp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        near = max(0.05, float(near_m))
        far = max(near * 1.01, float(far_m))
        inv_z = DepthScale(near, far).to_inv_z(disp)
        K = estimate_intrinsics(img.shape, hfov_deg=hf if hf and 10 < hf < 175 else None)
        if cancel is not None:
            cancel.check()
        inpainter = None
        mode = "none"
        if look_behind:
            try:
                from ..inpaint.base import get_inpainter

                inpainter = get_inpainter("pushpull")
                mode = "band"
            except Exception as e:  # inpainting module missing -> single layer
                log.debug("Preview look-behind unavailable: %s", e)
        li = build_layers(
            img, inv_z, K, max_baseline_m=PARALLAX_BASELINE_M, inpainter=inpainter, mode=mode, cancel=cancel
        )
        self.layers.put(key, (dp, li))
        return li

    # ---------------------------------------------------------------- renders
    @staticmethod
    def stereo_params(opts: dict[str, Any] | None) -> Any:
        """StereoParams from job options (``ipd_m``, ``strength``, ...), defaults otherwise."""
        from ..core.types import StereoParams

        o = opts or {}

        def num(name: str, default: float) -> float:
            v = o.get(name)
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default

        conv = o.get("convergence_m")
        return StereoParams(
            ipd_m=num("ipd_m", 0.063),
            strength=num("strength", 1.0),
            convergence_m=float(conv) if isinstance(conv, (int, float)) and conv else None,
            max_parallax_pct=num("max_parallax_pct", 3.0),
            mode="flat",
            swap_eyes=bool(o.get("swap_eyes", False)),
        )

    @staticmethod
    def anaglyph(li: Any, params: Any, *, cancel: CancelToken | None = None) -> np.ndarray:
        from ..geometry.stereo import compose, stereo_pair

        left, right = stereo_pair(li, params, cancel=cancel)
        return compose(left, right, "anaglyph")

    @staticmethod
    def wiggle(li: Any, params: Any, n: int = 8, *, cancel: CancelToken | None = None) -> list[np.ndarray]:
        from ..geometry.stereo import wiggle_frames

        return wiggle_frames(li, params, n=n, cancel=cancel)

    @staticmethod
    def parallax_frame(
        li: Any, dx: float, dy: float, amplitude_m: float = 0.04, *, cancel: CancelToken | None = None
    ) -> np.ndarray:
        """Render the view from a camera moved by (dx, dy) in [-1, 1] x ``amplitude_m``."""
        from ..core.types import Pose
        from ..geometry.warp import fill_holes, render_view

        dx = float(np.clip(dx, -1.0, 1.0))
        dy = float(np.clip(dy, -1.0, 1.0))
        pose = Pose.translate(dx * amplitude_m, dy * amplitude_m * 0.6, 0.0)
        rr = render_view(li, pose, iters=3)
        if cancel is not None:
            cancel.check()
        return fill_holes(rr)
