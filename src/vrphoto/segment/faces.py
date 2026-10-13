"""Face detection with OpenCV's YuNet (``cv2.FaceDetectorYN``).

'SHARP 360' uses it to reject an AI back view that shows a face (design 3.5.6 / M6: a
frontal face scoring above 0.8 in the head region of the 135 / 180 / 225 deg views means the
generator painted the front onto the back of the head) and as the V11 "face" penalty.

The model is the registry's ``yunet`` (MIT, 0.23 MB), downloaded on first use like every
other model (``models.download.ensure_model``; ``VRPHOTO_MODEL_YUNET`` points at a local
file).  It is optional: when OpenCV has no ``FaceDetectorYN``, the model cannot be
downloaded (offline) or fails to load, :func:`find_faces` returns ``[]`` and
:func:`frontal_face_score` 0.0 - the check is skipped, never the job.  A failed download is
not retried for a few minutes (``segment._runtime``'s download memory).

Coordinates are pixels of the picture passed in: a face is ``(x, y, w, h, score)`` with
``(x, y)`` its box's top-left corner.  Pictures are RGB (``HxWx3``; grey and RGBA accepted,
uint8 / uint16 / float in [0, 1]).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from ..core.errors import Cancelled, InputError
from ..core.log import get_logger
from ..core.types import CancelToken
from . import _runtime as rt

log = get_logger(__name__)

MODEL_ID = "yunet"
#: Design 3.5.6: an AI back view with a frontal face above this score in its head region is rejected.
REJECT_SCORE = 0.8
#: :func:`frontal_face_score` reports faces from this score up (a continuous QC penalty);
#: compare its result with :data:`REJECT_SCORE` to reject.
MIN_SCORE = 0.5
NMS_THRESHOLD = 0.3
TOP_K = 50
#: The 2023mar graph's static input size: used when OpenCV refuses the picture's own size
#: (OpenCV 5's ONNX Runtime engine needs a fixed size for that graph).
FIXED_SIZE = 640
#: A face counts for the head region when this much of its box is on the subject's matte.
MATTE_COVER = 0.5

Face = tuple[float, float, float, float, float]


class Detector(Protocol):
    """What :func:`find_faces` needs: ``detect(bgr, score_threshold)`` -> ``(N, 5 | 15)``
    float rows ``x, y, w, h[, 5 landmarks (x, y): right eye, left eye, nose tip, right and
    left mouth corner], score`` in the pixels of ``bgr`` (uint8 ``HxWx3``)."""

    def detect(self, bgr: np.ndarray, score_threshold: float) -> np.ndarray: ...


class YuNet:
    """``cv2.FaceDetectorYN`` for pictures of any size (thread-safe: one call at a time)."""

    def __init__(self, model_path: str | Path):
        self.path = Path(model_path)
        self._lock = threading.Lock()
        self._det = cv2.FaceDetectorYN.create(str(self.path), "", (320, 320), REJECT_SCORE, NMS_THRESHOLD, TOP_K)
        self._fixed = False  # OpenCV refused a free input size once: always letterbox

    def detect(self, bgr: np.ndarray, score_threshold: float) -> np.ndarray:
        h, w = bgr.shape[:2]
        with self._lock:
            self._det.setScoreThreshold(float(score_threshold))
            if not self._fixed:
                try:
                    return self._run(bgr, (w, h))
                except cv2.error as e:
                    log.info("YuNet at %dx%d failed (%s); using its fixed %d px input", w, h,
                             str(e).strip().splitlines()[-1][:120], FIXED_SIZE)
                    self._fixed = True
            s = min(1.0, FIXED_SIZE / max(h, w))
            img = bgr if s == 1.0 else cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                                                  interpolation=cv2.INTER_AREA)
            pad = np.full((FIXED_SIZE, FIXED_SIZE, 3), 127, np.uint8)  # bottom / right only: same coordinates
            pad[: img.shape[0], : img.shape[1]] = img
            rows = self._run(pad, (FIXED_SIZE, FIXED_SIZE))
            if s < 1.0:
                rows[:, :14] /= s
            return rows

    def _run(self, img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        self._det.setInputSize(size)
        _, faces = self._det.detect(np.ascontiguousarray(img))
        return np.zeros((0, 15), np.float32) if faces is None else np.asarray(faces, np.float32).reshape(-1, 15)


# ----------------------------------------------------------------------------- the model
_lock = threading.Lock()
_detectors: dict[str, YuNet] = {}
_bad_paths: set[str] = set()
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)
    else:
        log.debug(msg, *args)


def supported() -> bool:
    """True when this OpenCV build has ``cv2.FaceDetectorYN`` (OpenCV >= 4.5.4)."""
    return hasattr(cv2, "FaceDetectorYN")


def available() -> bool:
    """True when faces can be detected without a download (never raises)."""
    return supported() and rt.is_available(MODEL_ID)


def model_path(*, download: bool = True, cancel: CancelToken | None = None) -> Path | None:
    """The YuNet ONNX file (downloaded on first use when ``download``), or None when it is not
    on this PC and cannot be fetched now (logged once).  :class:`Cancelled` propagates."""
    try:
        from ..models import download as dl
    except Exception as e:  # a broken build: the check is optional
        _warn_once("import", "Face check skipped: the model downloader is unavailable (%s)", e)
        return None
    try:
        if dl.is_downloaded(MODEL_ID):
            return dl.local_path(MODEL_ID)
        if not download:
            return None
        if rt.recent_download_failure(MODEL_ID):
            return None
        path = dl.ensure_model(MODEL_ID, cancel=cancel)
        rt.clear_download_failure(MODEL_ID)
        return path
    except Cancelled:
        raise
    except Exception as e:  # offline, a bad env override, disk full ...: no face check this time
        rt.note_download_failure(MODEL_ID)
        _warn_once("download", "Face check skipped: the YuNet face model is not available (%s)", e)
        return None


def get_detector(*, download: bool = True, cancel: CancelToken | None = None) -> Detector | None:
    """The shared YuNet detector, or None when OpenCV or the model is unavailable."""
    if not supported():
        _warn_once("cv2", "Face check skipped: this OpenCV build (%s) has no FaceDetectorYN", cv2.__version__)
        return None
    path = model_path(download=download, cancel=cancel)
    if path is None:
        return None
    key = str(path)
    with _lock:
        det = _detectors.get(key)
        if det is not None:
            return det
        if key in _bad_paths:
            return None
        try:
            det = YuNet(path)
        except (cv2.error, OSError, ValueError) as e:
            _bad_paths.add(key)
            _warn_once("load", "Face check skipped: OpenCV could not load the YuNet model %s (%s)", path, e)
            return None
        _detectors[key] = det
        return det


def reset() -> None:
    """Forget the loaded detector and remembered failures (tests, a re-downloaded model)."""
    with _lock:
        _detectors.clear()
        _bad_paths.clear()
        _warned.clear()


# ----------------------------------------------------------------------------- detection
def _to_bgr_u8(img: Any) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4) or a.shape[0] < 1 or a.shape[1] < 1:
        raise InputError(f"Expected an RGB picture (H x W x 3), got shape {a.shape}.")
    if a.dtype == np.uint8:
        u8 = a
    elif a.dtype == np.uint16:
        u8 = (a // 257).astype(np.uint8)
    else:
        f = np.nan_to_num(a.astype(np.float32), nan=0.0)
        scale = 255.0 if f.size and float(f.max()) <= 1.0 + 1e-6 else 1.0
        u8 = np.clip(np.rint(f * scale), 0, 255).astype(np.uint8)
    if u8.shape[2] == 1:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(u8[..., 2::-1])  # RGB(A) -> BGR


def detect_faces(
    rgb: Any,
    *,
    score_threshold: float = REJECT_SCORE,
    max_long: int = 640,
    detector: Detector | None = None,
    download: bool = True,
    cancel: CancelToken | None = None,
) -> np.ndarray:
    """All faces as ``(N, 5 | 15)`` float32 rows ``x, y, w, h[, landmarks], score`` in the
    pixels of ``rgb`` (the score is the last column), best first.  The picture is shrunk to
    a long side of ``max_long`` first (YuNet finds faces of about 10-300 px).  Empty when the
    detector is unavailable or fails (logged)."""
    bgr = _to_bgr_u8(rgb)
    if detector is None:
        detector = get_detector(download=download, cancel=cancel)
        if detector is None:
            return np.zeros((0, 15), np.float32)
    h, w = bgr.shape[:2]
    s = min(1.0, float(max_long) / max(h, w)) if max_long and max_long > 0 else 1.0
    if s < 1.0:
        bgr = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    if min(bgr.shape[:2]) < 10:
        return np.zeros((0, 15), np.float32)
    try:
        raw = np.asarray(detector.detect(bgr, float(score_threshold)), np.float32)
    except Cancelled:
        raise
    except Exception as e:  # cv2.error and friends: no face check for this picture
        _warn_once("detect", "Face detection failed (%s: %s); the face check is skipped", type(e).__name__, e)
        return np.zeros((0, 15), np.float32)
    if raw.size == 0:
        return np.zeros((0, raw.shape[1] if raw.ndim == 2 and raw.shape[1] >= 5 else 15), np.float32)
    raw = raw.reshape(len(raw), -1)
    if raw.shape[1] < 5:
        raise InputError(f"A face detector returned {raw.shape[1]} values per face (expected 5 or 15).")
    raw = raw[np.isfinite(raw).all(axis=1) & (raw[:, -1] >= float(score_threshold))]
    if s < 1.0:  # back to the caller's pixels (the score column stays)
        raw = raw.copy()
        raw[:, :-1] /= s
    return raw[np.argsort(-raw[:, -1], kind="stable")]


def find_faces(
    rgb: Any,
    *,
    score_threshold: float = REJECT_SCORE,
    max_long: int = 640,
    detector: Detector | None = None,
    download: bool = True,
    cancel: CancelToken | None = None,
) -> list[Face]:
    """Faces ``(x, y, w, h, score)`` scoring at least ``score_threshold`` (pixels of ``rgb``,
    best first); ``[]`` when there is none or the detector is unavailable."""
    raw = detect_faces(rgb, score_threshold=score_threshold, max_long=max_long, detector=detector,
                       download=download, cancel=cancel)
    return [(float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[-1])) for r in raw]


def head_region(
    alpha: Any, head_frac: float = 0.3, *, threshold: float = 0.5
) -> tuple[float, float, float, float] | None:
    """``(x0, y0, x1, y1)`` (pixel edges): the top ``head_frac`` of the bounding box of
    ``alpha >= threshold`` - where an upright subject's head is; None for an empty matte."""
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    on = a >= threshold
    rows = np.flatnonzero(on.any(axis=1))
    if not len(rows):
        return None
    cols = np.flatnonzero(on.any(axis=0))
    y0, y1 = float(rows[0]), float(rows[-1] + 1)
    frac = min(max(float(head_frac), 0.0), 1.0)
    return float(cols[0]), y0, float(cols[-1] + 1), y0 + frac * (y1 - y0)


def is_frontal(face: Any) -> bool:
    """True unless the face's landmarks say it is turned away (the nose tip outside the span
    of the eyes, the eyes not above the nose, or eyes closer than a quarter of the box width:
    a profile).  Rows without landmarks count as frontal."""
    r = np.asarray(face, dtype=np.float64).ravel()
    if len(r) < 15:
        return True
    w = r[2]
    rex, rey, lex, ley, nx, ny = r[4], r[5], r[6], r[7], r[8], r[9]
    d = abs(lex - rex)
    lo, hi = min(rex, lex) - 0.15 * d, max(rex, lex) + 0.15 * d
    return bool(d >= 0.25 * w and lo <= nx <= hi and max(rey, ley) < ny)


def _matte_cover(a: np.ndarray, face: np.ndarray) -> float:
    H, W = a.shape
    x0, y0 = int(np.clip(np.floor(face[0]), 0, W)), int(np.clip(np.floor(face[1]), 0, H))
    x1, y1 = int(np.clip(np.ceil(face[0] + face[2]), 0, W)), int(np.clip(np.ceil(face[1] + face[3]), 0, H))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((a[y0:y1, x0:x1] >= 0.5).mean())


def frontal_face_score(
    rgb: Any,
    alpha: Any = None,
    *,
    head_frac: float = 0.3,
    min_score: float = MIN_SCORE,
    frontal_only: bool = True,
    max_long: int = 640,
    detector: Detector | None = None,
    download: bool = True,
    cancel: CancelToken | None = None,
) -> float:
    """The best score of a face in the subject's head region, 0.0 when there is none or the
    detector is unavailable.

    With ``alpha`` (the subject's matte, ``HxW`` - resized to the picture when it differs) a
    face counts when its centre lies in the top ``head_frac`` of the matte's bounding box
    (:func:`head_region`) and at least half of its box is on the matte; without it anywhere in
    the picture.  ``frontal_only`` ignores faces whose landmarks say they are turned away
    (:func:`is_frontal`).  Faces scoring below ``min_score`` are not reported; reject a back
    view when the result is above :data:`REJECT_SCORE` (0.8)."""
    raw = detect_faces(rgb, score_threshold=min_score, max_long=max_long, detector=detector, download=download,
                       cancel=cancel)
    if not len(raw):
        return 0.0
    H, W = np.asarray(rgb).shape[:2]
    a = None
    region = (0.0, 0.0, float(W), float(H))
    if alpha is not None:
        a = np.asarray(alpha, dtype=np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        if a.shape != (H, W):
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
        if a.size and float(np.nanmax(a)) > 1.0 + 1e-6:  # a uint8-style matte
            a = a / 255.0
        reg = head_region(a, head_frac)
        if reg is None:
            return 0.0
        region = reg
    best = 0.0
    for r in raw:
        cx, cy = r[0] + 0.5 * r[2], r[1] + 0.5 * r[3]
        if not (region[0] <= cx < region[2] and region[1] <= cy < region[3]):
            continue
        if a is not None and _matte_cover(a, r) < MATTE_COVER:
            continue
        if frontal_only and not is_frontal(r):
            continue
        best = max(best, float(r[-1]))
    return best


__all__ = [
    "Detector",
    "FIXED_SIZE",
    "Face",
    "MIN_SCORE",
    "MODEL_ID",
    "REJECT_SCORE",
    "YuNet",
    "available",
    "detect_faces",
    "find_faces",
    "frontal_face_score",
    "get_detector",
    "head_region",
    "is_frontal",
    "model_path",
    "reset",
    "supported",
]
