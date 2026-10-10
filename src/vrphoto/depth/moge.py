"""MoGe-2 (Microsoft, MIT) monocular geometry: metric depth + camera field of view.

The ONNX export only contains the network forward pass (see MoGe ``docs/onnx.md``):
inputs ``image`` (float32 ``[B,3,H,W]`` RGB in [0,1]; ImageNet normalisation happens
inside the graph) and ``num_tokens`` (int64 scalar, 1200-2500); outputs ``points``
(affine-invariant point map ``[B,H,W,3]``), ``normal``, ``mask`` (sigmoid probability)
and ``metric_scale``.  The post-processing of ``MoGeModel.infer`` is ported here:

* :func:`recover_focal_shift` - numpy port of ``moge.utils.geometry_torch.recover_focal_shift``
  (nearest-downsample to 64x64, then solve ``min |f * xy / (z + shift) - uv|`` for the z
  shift with the focal in closed form).  MoGe uses SciPy's Levenberg-Marquardt; SciPy is
  not a dependency, so a small 1-D LM with numeric derivatives is used instead.
* metric depth ``z = (points_z + shift) * metric_scale``; intrinsics from the focal, which
  is relative to half the image diagonal (``f_px = focal * diag / 2``).

The image is resized to the encoder's token grid (``14 * round(sqrt(N / a))`` x
``14 * round(sqrt(N * a))``) before inference: the exported graph resizes without
antialiasing, so feeding it the exact grid size avoids aliasing and wasted work.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from ..core.log import get_logger
from ..core.types import RGB, CancelToken, DepthResult, Intrinsics, ProgressReporter
from .base import as_rgb, check, reporter
from .da2 import static_hw
from .post import finalize_disparity

log = get_logger(__name__)

PATCH = 14
DEFAULT_TOKENS = 1800
FALLBACK_FOV_DEG = 60.0  # used when the recovered focal is implausible
FOV_RANGE_DEG = (5.0, 170.0)
MIN_VALID_FRAC = 0.01  # below this fraction of valid pixels the camera/scale are not trusted


# --------------------------------------------------------------------------- geometry port
def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float | None = None) -> np.ndarray:
    """MoGe's normalised image-plane coordinates, ``HxWx2`` float32.

    Pixel centres span ``(-w/diag, -h/diag)`` .. ``(w/diag, h/diag)`` (left-top to
    right-bottom), i.e. pixel ``i`` maps to ``(i + 0.5 - W/2) / (diag / 2)``.
    """
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=np.float32)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.stack([uu, vv], axis=-1)


def _nearest_idx(n_src: int, n_dst: int) -> np.ndarray:
    """Source indices of PyTorch ``F.interpolate(mode='nearest')``."""
    return np.minimum((np.arange(n_dst) * (n_src / n_dst)).astype(np.int64), n_src - 1)


def _lm_shift(
    residual: Callable[[float], np.ndarray],
    z: np.ndarray,
    x0: float = 0.0,
    ftol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Minimise ``|residual(s)|^2`` over the scalar shift ``s`` (1-D Levenberg-Marquardt)."""
    scale = float(max(np.median(np.abs(z)), 1e-6))
    s = float(x0)
    r = residual(s)
    cost = float(r @ r)
    if not math.isfinite(cost):
        return s
    lam = 1e-3
    for _ in range(max_iter):
        h = 1e-6 * (abs(s) + scale)
        jac = (residual(s + h) - residual(s - h)) / (2 * h)
        g = float(jac @ r)
        hess = float(jac @ jac)
        if not (math.isfinite(g) and math.isfinite(hess)) or hess <= 0:
            break
        if math.sqrt(hess) * scale < 1e-9 * max(math.sqrt(cost), 1e-12):
            break  # cost is flat in s (e.g. fronto-parallel plane): keep the current shift
        accepted = False
        while lam < 1e12:
            step = -g / (hess * (1.0 + lam))
            s_new = s + step
            if np.median(z + s_new) > 0:  # never flip the scene behind the camera
                r_new = residual(s_new)
                c_new = float(r_new @ r_new)
                if math.isfinite(c_new) and c_new < cost * (1.0 - 1e-12):
                    accepted = True
                    break
            lam *= 10.0
        if not accepted:
            break
        rel = (cost - c_new) / max(cost, 1e-300)
        s, r, cost = s_new, r_new, c_new
        lam = max(lam / 10.0, 1e-12)
        if rel < ftol or abs(step) < 1e-12 * (abs(s) + scale):
            break
    return s


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray) -> tuple[float, float]:
    """Solve ``min |f * xy / (z + shift) - uv|`` for (shift, focal).  Returns ``(shift, focal)``."""
    uv = uv.reshape(-1, 2).astype(np.float64)
    xy = xyz[..., :2].reshape(-1, 2).astype(np.float64)
    z = xyz[..., 2].reshape(-1).astype(np.float64)

    def proj(shift: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return xy / (z + shift)[:, None]

    def residual(shift: float) -> np.ndarray:
        p = proj(shift)
        den = float(np.square(p).sum())
        f = float((p * uv).sum()) / den if den > 0 else 0.0
        return (f * p - uv).ravel()

    shift = _lm_shift(residual, z)
    p = proj(shift)
    den = float(np.square(p).sum())
    focal = float((p * uv).sum()) / den if den > 0 else 1.0
    return shift, focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float) -> float:
    """Solve ``min |focal * xy / (z + shift) - uv|`` for the shift with a known focal."""
    uv = uv.reshape(-1, 2).astype(np.float64)
    xy = xyz[..., :2].reshape(-1, 2).astype(np.float64)
    z = xyz[..., 2].reshape(-1).astype(np.float64)

    def residual(shift: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return (focal * xy / (z + shift)[:, None] - uv).ravel()

    return _lm_shift(residual, z)


def recover_focal_shift(
    points: np.ndarray,
    mask: np.ndarray | None = None,
    focal: float | None = None,
    downsample_size: tuple[int, int] = (64, 64),
) -> tuple[float, float]:
    """Recover ``(focal, shift)`` from an affine-invariant point map (port of MoGe).

    ``points``: ``HxWx3`` (a leading batch dim of 1 is accepted).  ``mask``: optional
    bool ``HxW`` of valid pixels.  ``focal`` (relative to half the image diagonal) may be
    given to solve for the shift only.  Assumes the principal point is the image centre
    and square pixels.  Returns ``(1.0, 0.0)`` if fewer than 2 valid points remain.
    """
    pts = np.asarray(points)
    if pts.ndim == 4 and pts.shape[0] == 1:
        pts = pts[0]
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"points must be HxWx3, got {pts.shape}")
    h, w = pts.shape[:2]
    uv = normalized_view_plane_uv(w, h)
    ri = _nearest_idx(h, downsample_size[0])
    ci = _nearest_idx(w, downsample_size[1])
    p_lr = pts[ri][:, ci].reshape(-1, 3)
    uv_lr = uv[ri][:, ci].reshape(-1, 2)
    keep = np.isfinite(p_lr).all(axis=1)
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        keep &= m.astype(bool)[ri][:, ci].reshape(-1)
    p_lr, uv_lr = p_lr[keep], uv_lr[keep]
    if p_lr.shape[0] < 2:
        return 1.0, 0.0
    if focal is None:
        shift, f = solve_optimal_focal_shift(uv_lr, p_lr)
        return float(f), float(shift)
    return float(focal), float(solve_optimal_shift(uv_lr, p_lr, float(focal)))


def fov_to_focal(fov_x_deg: float, width: int, height: int) -> float:
    """MoGe focal (relative to half diagonal) for a horizontal FOV."""
    a = width / height
    return a / (1 + a**2) ** 0.5 / math.tan(math.radians(fov_x_deg) / 2)


def focal_to_fov(focal: float, width: int, height: int) -> float:
    """Horizontal FOV in degrees for a MoGe focal."""
    diag = math.hypot(width, height)
    return math.degrees(2 * math.atan(width / diag / max(focal, 1e-9)))


def focal_to_intrinsics(focal: float, width: int, height: int) -> Intrinsics:
    """Pinhole intrinsics (pixels) of a ``width`` x ``height`` image for a MoGe focal.

    The principal point follows :meth:`Intrinsics.from_hfov` (``W/2, H/2``).
    """
    f_px = focal * math.hypot(width, height) / 2.0
    return Intrinsics(f_px, f_px, width / 2.0, height / 2.0, int(width), int(height))


def points_to_depth(
    points: np.ndarray,
    mask: np.ndarray | None = None,
    metric_scale: float | None = None,
    fov_x: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """MoGe ``infer`` post-processing for one point map.

    Returns ``(depth, valid, focal, shift)``: ``depth`` = ``(points_z + shift) *
    metric_scale`` (float32 HxW), ``valid`` bool HxW (mask > 0.5, finite, z > 0).  With
    ``fov_x`` (degrees) only the shift is solved.  An implausible recovered focal (FOV
    outside 5-170 degrees) falls back to a 60 degree FOV.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 4:
        pts = pts[0]
    h, w = pts.shape[:2]
    mask_b = None
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[0]
        mask_b = m > 0.5 if m.dtype != np.bool_ else m
    if fov_x is not None:
        focal, shift = recover_focal_shift(pts, mask_b, focal=fov_to_focal(fov_x, w, h))
    else:
        focal, shift = recover_focal_shift(pts, mask_b)
        fov = focal_to_fov(focal, w, h) if focal > 0 else float("nan")
        if not (FOV_RANGE_DEG[0] <= fov <= FOV_RANGE_DEG[1]):
            log.warning("MoGe recovered an implausible FOV (%.1f deg); assuming %.0f", fov, FALLBACK_FOV_DEG)
            focal, shift = recover_focal_shift(pts, mask_b, focal=fov_to_focal(FALLBACK_FOV_DEG, w, h))
    z = pts[..., 2] + np.float32(shift)
    valid = np.isfinite(z) & (z > 0)
    if mask_b is not None:
        valid &= mask_b
    if metric_scale is not None and math.isfinite(metric_scale) and metric_scale > 0:
        z = z * np.float32(metric_scale)
    return z.astype(np.float32), valid, float(focal), float(shift)


# --------------------------------------------------------------------------- estimator
def token_grid(h: int, w: int, num_tokens: int) -> tuple[int, int]:
    """Encoder token grid ``(rows, cols)`` as computed by ``MoGeModel.forward``."""
    a = w / h
    return max(1, round((num_tokens / a) ** 0.5)), max(1, round((num_tokens * a) ** 0.5))


class MoGeEstimator:
    """MoGe-2 estimator: relative disparity + metric depth (metres) + intrinsics."""

    def __init__(
        self,
        model_id: str = "moge2-vitl",
        device: str | int = "auto",
        *,
        num_tokens: int = DEFAULT_TOKENS,
        fov_x: float | None = None,
    ) -> None:
        self.id = model_id
        self.device = device
        self.num_tokens = int(min(max(num_tokens, 1200), 3600))
        self.fov_x = fov_x
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

    @staticmethod
    def _image_input(model: Any) -> int:
        """Index of the image input (the one that is not the token count)."""
        names = model.input_names()
        for i, n in enumerate(names):
            if "token" not in n.lower():
                return i
        return 0

    def net_size(self, h: int, w: int, model: Any) -> tuple[int, int]:
        """Network input ``(H, W)`` for an image of ``h`` x ``w``."""
        fixed = static_hw(model.input_shapes()[self._image_input(model)])
        if fixed is not None:
            return fixed
        rows, cols = token_grid(h, w, self.num_tokens)
        return rows * PATCH, cols * PATCH

    def _feeds(self, model: Any, x: np.ndarray) -> dict[str, np.ndarray]:
        names = model.input_names()
        shapes = model.input_shapes()
        img_i = self._image_input(model)
        feeds = {names[img_i]: x}
        for i, (n, shp) in enumerate(zip(names, shapes)):
            if i != img_i and "token" in n.lower():
                tok = np.array(self.num_tokens, dtype=np.int64)
                feeds[n] = tok.reshape([1] * len(shp)) if shp else tok
        return feeds

    def estimate(
        self, img: RGB, *, progress: ProgressReporter | None = None, cancel: CancelToken | None = None
    ) -> DepthResult:
        """Disparity [0,1] (1 = near), metric depth ``metric_m`` (m) and ``intrinsics``."""
        rep = reporter(progress)
        rgb = as_rgb(img)
        h, w = rgb.shape[:2]
        model = self._load(rep.sub(0.0, 0.5), cancel)
        check(cancel)
        rep(0.5, "Estimating geometry (MoGe-2)")
        th, tw = self.net_size(h, w, model)
        interp = cv2.INTER_AREA if (th < h and tw < w) else cv2.INTER_CUBIC
        x = cv2.resize(rgb, (tw, th), interpolation=interp) if (th, tw) != (h, w) else rgb
        x = np.ascontiguousarray((x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        outs = model.run(self._feeds(model, x), cancel=cancel)
        if "points" not in outs:
            from ..core.errors import InferenceError

            raise InferenceError(f"{self.id}: model has no 'points' output (got {sorted(outs)})")
        points = np.asarray(outs["points"], np.float32)
        points = points[0] if points.ndim == 4 else points
        mask = outs.get("mask")
        scale_arr = outs.get("metric_scale")
        metric_scale = float(np.ravel(scale_arr)[0]) if scale_arr is not None and np.size(scale_arr) else None
        check(cancel)
        rep(0.8, "Recovering camera")
        ph, pw = points.shape[:2]
        z, valid, focal, shift = points_to_depth(points, mask, metric_scale, fov_x=self.fov_x)

        if valid.any():
            zv = z[valid]
            far = float(np.percentile(zv, 99.0)) * 2.0
            near = float(max(zv.min(), 1e-3))
        else:  # nothing valid: flat scene at a nominal distance
            far, near = 10.0, 10.0
        zf = np.where(valid, np.clip(z, near, far), far).astype(np.float32)
        inv = (1.0 / zf).astype(np.float32)
        disp, raw = finalize_disparity(inv, rgb)
        metric = None
        K: Intrinsics | None = None
        frac_valid = float(valid.mean())
        if frac_valid < MIN_VALID_FRAC:
            # e.g. an all-sky / blank image: the recovered camera and scale are meaningless,
            # so let the caller fall back to EXIF / default FOV and relative depth
            log.warning(
                "MoGe found almost no valid geometry (%.2f%% of pixels); ignoring its camera and scale",
                100.0 * frac_valid,
            )
        else:
            if metric_scale is not None:
                inv_up = np.clip(raw, 1.0 / far, 1.0 / near)
                metric = (1.0 / inv_up).astype(np.float32)
            K = focal_to_intrinsics(focal, pw, ph).resized(w, h)
            log.info("MoGe: hfov %.1f deg, shift %.3f, metric scale %s", K.hfov_deg, shift, metric_scale)
        rep(1.0, "Depth ready")
        return DepthResult(disparity=disp, model_id=self.id, metric_m=metric, intrinsics=K, raw=raw)


__all__ = [
    "MoGeEstimator",
    "recover_focal_shift",
    "solve_optimal_focal_shift",
    "solve_optimal_shift",
    "normalized_view_plane_uv",
    "points_to_depth",
    "focal_to_intrinsics",
    "focal_to_fov",
    "fov_to_focal",
    "token_grid",
]
