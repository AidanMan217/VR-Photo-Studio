"""Capture to Gaussian splat: many photos or a phone video -> COLMAP poses -> Brush training
-> cleaned ``.splat`` / ``.ply`` ready for the WebXR viewer.

Pipeline (:func:`capture_to_splat`):

1. Videos: extract frames at ``video_fps`` with ffmpeg (2x oversampled), keep the sharpest
   frame of each pair and drop blurry ones (variance of the Laplacian), cap to
   ``max_images`` by uniform subsampling.  Photos are copied (EXIF orientation applied).
2. COLMAP: ``feature_extractor`` (one shared OPENCV camera, GPU when an NVIDIA GPU is
   present), ``sequential_matcher`` for video frames / ``exhaustive_matcher`` for up to 100
   photos, ``global_mapper`` when available (COLMAP >= 4) falling back to the incremental
   ``mapper``, ``image_undistorter`` -> ``images/`` + ``sparse/`` (the layout Brush reads).
3. Brush: headless training (``--total-steps``, single export at the end).
4. Post-processing with :mod:`vrphoto.geometry.splat`: crop floaters, orient so that the
   average camera "up" becomes +y with the floor at y = 0, give the splat a real-world
   (metric) scale, decimate, write ``.splat`` and ``.ply`` plus ``<name>.capture.json``.

Scale: photos alone cannot tell COLMAP how big the scene is, so its units are arbitrary
(about 2 units per metre for a typical walk-around).  The splat is rescaled so the median
registered camera is ``camera_height_m`` (1.45 m: a phone held at chest/eye height) above
the floor; if the camera heights are unusable, so that the cameras orbit at about
``ORBIT_RADIUS_M``.  With ``subject_height_m > 0`` the subject near the centre is made that
tall instead.  ``<name>.capture.json`` records ``{"scale", "height_m", "method"}`` (plus
``camera_height_m`` for diagnostics, and ``subject_height_m`` when the subject height was
given) where ``height_m`` is the height of the written splat's bounding box (splat centres)
in metres - exactly what the viewer's ``object.height_m`` fits, so the splat keeps its real
size; it is NOT the subject's height.

The tools come from the ``capture`` add-on (``manifests/capture.toml``); ``VRPHOTO_COLMAP`` /
``VRPHOTO_BRUSH`` override their paths (a ``.py`` path runs with the current Python -
used by the tests with fake tools).

Stage 3 is reusable for datasets with known cameras (SHARP 360, docs/design/SHARP360.md
§3.11 and §4.5): :func:`write_colmap_text` writes the COLMAP text model Brush reads,
:func:`train_splat` runs Brush on any prepared folder, and :func:`brush_selftest`
(``vrphoto-cli addons brush-test``) checks once on the user's PC that Brush trains there,
whether SH degree 0 works and whether masked and transparent pictures can be mixed.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import AddonError, AddonNotInstalled, InputError
from ..core.fsutil import atomic_write_bytes
from ..core.imaging import IMAGE_EXTS, VIDEO_EXTS, is_image, is_video
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter
from .env import clean_env, native_tool_env
from .runner import LogSink, ProcResult, run_process

log = get_logger(__name__)

CAPTURE_ID = "capture"
MIN_IMAGES = 3
EXHAUSTIVE_MAX = 100
UNDISTORT_MAX_PX = 1600
CAMERA_HEIGHT_M = 1.45  # typical height of a hand-held phone above the floor
ORBIT_RADIUS_M = 2.0  # typical walk-around radius (User Guide: 1.5-3 m)


# --------------------------------------------------------------------------- tools
def _tool_cmd(path: str) -> list[str]:
    from .manager import tool_command

    return tool_command(path)


def find_colmap() -> list[str]:
    """Command prefix for COLMAP (raises :class:`AddonNotInstalled`)."""
    return _tool_cmd(_resolve("colmap"))


def find_brush() -> list[str]:
    """Command prefix for Brush (raises :class:`AddonNotInstalled`)."""
    return _tool_cmd(_resolve("brush"))


def _resolve(name: str) -> str:
    from . import manager  # lazy: manager imports are cheap but keep capture importable standalone

    try:
        m = manager.get_manifest(CAPTURE_ID)
    except AddonError:
        env_var = {"colmap": "VRPHOTO_COLMAP", "brush": "VRPHOTO_BRUSH"}[name]
        p = os.environ.get(env_var) or shutil.which("colmap" if name == "colmap" else "brush_app")
        if p:
            return p
        raise AddonNotInstalled(f"{name} was not found.", hint="Install the capture add-on.") from None
    return manager.resolve_binary(m, name)


def _tool_env(cmd: Sequence[str]) -> dict[str, str]:
    """Environment for COLMAP / Brush: their own bin / lib folders first on PATH, COLMAP's
    Qt plugins (see :func:`vrphoto.addons.env.native_tool_env`), Brush logging on."""
    exe = cmd[1] if len(cmd) > 1 and str(cmd[1]).lower().endswith(".py") else cmd[0]
    return native_tool_env(exe, {"RUST_LOG": "info", "RUST_BACKTRACE": "1"})


def _has_nvidia_gpu() -> bool:
    try:
        from ..models.gpu import nvidia_vram_mb  # optional sibling module

        return nvidia_vram_mb() is not None
    except Exception:
        return shutil.which("nvidia-smi") is not None


@dataclass
class ColmapInfo:
    """Capabilities of the COLMAP build in use."""

    version: str = ""
    has_global_mapper: bool = False
    new_option_names: bool = True  # FeatureExtraction.* (>= 3.12) vs SiftExtraction.*
    has_log_target: bool = False


_info_cache: dict[str, ColmapInfo] = {}


def colmap_info(cmd: Sequence[str]) -> ColmapInfo:
    """Probe COLMAP (``help`` and ``feature_extractor -h``); cached per executable."""
    key = " ".join(cmd)
    if key in _info_cache:
        return _info_cache[key]
    env = _tool_env(cmd)
    info = ColmapInfo()

    def output(args: list[str]) -> str:
        # the help texts are longer than ProcResult.tail - collect every line
        lines: list[str] = []
        run_process([*cmd, *args], env=env, timeout=60, title="COLMAP", on_line=lines.append)
        return "\n".join(lines)

    try:
        text = output(["help"])
        m = re.search(r"COLMAP\s+(\d+\.\d+(?:\.\d+)?)", text)
        info.version = m.group(1) if m else ""
        info.has_global_mapper = re.search(r"^\s*global_mapper\s*$", text, re.M) is not None
        text = output(["feature_extractor", "-h"])
        if "FeatureExtraction.use_gpu" in text:
            info.new_option_names = True
        elif "SiftExtraction.use_gpu" in text:
            info.new_option_names = False
        else:  # unknown help format: decide by version (FeatureExtraction.* arrived with 3.12)
            info.new_option_names = _version_tuple(info.version) >= (3, 12) if info.version else True
        info.has_log_target = "--log_target" in text
    except AddonError as e:
        log.warning("Could not probe COLMAP: %s", e)
    _info_cache[key] = info
    return info


def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return ()


# --------------------------------------------------------------------------- frames
def sharpness(img_or_path: np.ndarray | str | os.PathLike[str], max_long: int = 640) -> float:
    """Variance of the Laplacian of the (downscaled) grayscale image; higher = sharper."""
    import cv2

    if isinstance(img_or_path, np.ndarray):
        img = img_or_path
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    else:
        data = np.fromfile(os.fspath(img_or_path), dtype=np.uint8)
        gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return 0.0
    h, w = gray.shape[:2]
    s = max_long / max(h, w)
    if s < 1.0:
        gray = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharp(
    frames: Sequence[Path],
    *,
    group: int = 2,
    rel_thresh: float = 0.35,
    max_images: int | None = None,
    scores: Sequence[float] | None = None,
) -> list[Path]:
    """Keep the sharpest frame of each run of ``group`` consecutive frames, drop frames
    whose sharpness is below ``rel_thresh`` x median, then subsample uniformly to
    ``max_images``.  Order is preserved."""
    frames = list(frames)
    if not frames:
        return []
    sc = list(scores) if scores is not None else [sharpness(f) for f in frames]
    group = max(1, int(group))
    picked: list[int] = []
    for start in range(0, len(frames), group):
        idx = range(start, min(start + group, len(frames)))
        picked.append(max(idx, key=lambda i: sc[i]))
    if len(picked) > MIN_IMAGES:
        med = float(np.median([sc[i] for i in picked]))
        good = [i for i in picked if sc[i] >= rel_thresh * med]
        if len(good) >= MIN_IMAGES:
            picked = good
    if max_images and len(picked) > max_images:
        sel = np.linspace(0, len(picked) - 1, int(max_images)).round().astype(int)
        picked = [picked[i] for i in sorted(set(sel.tolist()))]
    return [frames[i] for i in picked]


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise AddonError("ffmpeg was not found.", hint="Reinstall VR Photo Studio.") from e


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_TIME_RE = re.compile(r"time=\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def extract_frames(
    video: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    fps: float = 2.0,
    max_long: int = 1920,
    prefix: str = "frame",
    cancel: CancelToken | None = None,
    progress: ProgressReporter | None = None,
    log_sink: LogSink | None = None,
) -> list[Path]:
    """Extract JPEG frames from ``video`` at ``fps`` (long side limited to ``max_long``)."""
    video = Path(video)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if not video.is_file():
        raise InputError(f"Video not found: {video}")
    progress = progress or ProgressReporter.null()
    duration = [0.0]

    def on_line(line: str) -> None:
        m = _DUR_RE.search(line)
        if m and not duration[0]:
            duration[0] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        m = _TIME_RE.search(line)
        if m and duration[0]:
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            progress(min(1.0, t / duration[0]), f"Extracting frames from {video.name}")

    scale = (
        f"scale='if(gt(iw,ih),min({max_long},iw),-2)':'if(gt(iw,ih),-2,min({max_long},ih))'"
    )
    pattern = dst / f"{prefix}_%05d.jpg"
    cmd = [
        _ffmpeg(), "-hide_banner", "-nostdin", "-y", "-i", str(video),
        "-vf", f"fps={fps:g},{scale}", "-q:v", "2", "-start_number", "0", str(pattern),
    ]
    res = run_process(cmd, env=clean_env(), log_sink=log_sink, on_line=on_line, cancel=cancel,
                      title="Frame extraction")
    frames = sorted(dst.glob(f"{prefix}_*.jpg"))
    if not res.ok or not frames:
        raise InputError(
            f"Could not read frames from {video.name}.",
            hint="Use a common video format (MP4/MOV H.264 or HEVC). " + (res.tail[-1] if res.tail else ""),
        )
    progress(1.0, f"{len(frames)} frames extracted")
    return frames


# --------------------------------------------------------------------------- paths
def _ascii_name(text: str, fallback: str = "img") -> str:
    """File-name-safe ASCII version of ``text`` (COLMAP on Windows reads paths in the ANSI
    code page, so non-ASCII names may not open)."""
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return out[:60] or fallback


def _short_path(path: Path) -> Path:
    """Windows 8.3 short form of an existing path (ASCII when 8.3 names are enabled)."""
    if sys.platform != "win32":
        return path
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))  # type: ignore[attr-defined]
        if 0 < n < len(buf):
            return Path(buf.value)
    except Exception as e:  # pragma: no cover - Windows only, best effort
        log.debug("GetShortPathNameW failed: %s", e)
    return path


def _work_dir(out_dir: Path, stem: str) -> Path:
    """Scratch folder for COLMAP / Brush: inside ``out_dir`` normally; on Windows a path
    COLMAP can open even when the user name / folder contains non-ASCII characters
    (8.3 short path, else the public profile folder)."""
    name = f".capture_work_{_ascii_name(stem, 'capture')}"
    if sys.platform != "win32" or str(out_dir).isascii():
        return out_dir / name
    short = _short_path(out_dir)
    if str(short).isascii():
        return short / name
    public = Path(os.environ.get("PUBLIC") or r"C:\Users\Public")
    base = public / "VRPhotoStudio" / "capture"
    log.info("Output folder %s is not ASCII; COLMAP works in %s", out_dir, base)
    return base / f"{name}_{os.getpid()}_{int(time.time())}"


# --------------------------------------------------------------------------- inputs
def _expand_inputs(inputs: Sequence[str | os.PathLike[str]]) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    videos: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    images.append(f)
                elif f.suffix.lower() in VIDEO_EXTS:
                    videos.append(f)
        elif not p.exists():
            raise InputError(f"Input not found: {p}")
        elif is_video(p):
            videos.append(p)
        elif is_image(p):
            images.append(p)
        else:
            raise InputError(f"Unsupported input for capture: {p.name}", hint="Use photos (JPG/PNG/HEIC) or videos (MP4/MOV).")
    return images, videos


def _photo_sort_key(p: Path) -> tuple[str, str]:
    """Capture time (EXIF DateTimeOriginal) then file name: sequential order of a walk-around."""
    stamp = ""
    try:
        from PIL import Image

        with Image.open(p) as im:
            exif = im.getexif()
            stamp = str(exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132) or "")
    except Exception:
        pass
    return (stamp, p.name.lower())


def _camera_key(p: Path) -> tuple:
    """Photos sharing camera model, focal length and size share one COLMAP camera."""
    try:
        from PIL import Image

        with Image.open(p) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769)
            size = im.size
            orient = int(exif.get(0x0112, 1) or 1)
            if orient in (5, 6, 7, 8):
                size = (size[1], size[0])
            return (str(exif.get(0x010F, "")), str(exif.get(0x0110, "")), str(sub.get(0x920A, "")), size)
    except Exception:
        return ("?", p.suffix.lower())


def _import_photo(src: Path, dst: Path) -> None:
    """Copy a photo for COLMAP; re-encode when it is rotated by EXIF or not JPEG/PNG."""
    orient = 1
    try:
        from PIL import Image

        with Image.open(src) as im:
            orient = int(im.getexif().get(0x0112, 1) or 1)
    except Exception:
        orient = 1
    if src.suffix.lower() in (".jpg", ".jpeg", ".png") and orient == 1:
        shutil.copy2(src, dst.with_suffix(src.suffix.lower()))
        return
    from ..core.imaging import load_image, save_image

    save_image(dst.with_suffix(".jpg"), load_image(src), quality=95)


# --------------------------------------------------------------------------- COLMAP model io
def read_images_bin(path: str | os.PathLike[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Parse COLMAP ``images.bin`` -> ``[(name, qvec_wxyz, tvec)]`` (cam_from_world)."""
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n):
            data = fh.read(4 + 8 * 7 + 4)
            vals = struct.unpack("<I7dI", data)
            q = np.array(vals[1:5], dtype=np.float64)
            t = np.array(vals[5:8], dtype=np.float64)
            name = bytearray()
            while True:
                c = fh.read(1)
                if not c or c == b"\0":
                    break
                name += c
            (npts,) = struct.unpack("<Q", fh.read(8))
            fh.seek(npts * 24, os.SEEK_CUR)
            out.append((name.decode("utf-8", errors="replace"), q, t))
    return out


def count_registered(model_dir: str | os.PathLike[str]) -> int:
    """Number of registered images of a COLMAP model folder (0 if unreadable)."""
    p = Path(model_dir) / "images.bin"
    try:
        with open(p, "rb") as fh:
            return int(struct.unpack("<Q", fh.read(8))[0])
    except (OSError, struct.error):
        return 0


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / (np.linalg.norm(q) or 1.0)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def camera_up_vector(images_bin: str | os.PathLike[str]) -> np.ndarray | None:
    """Average world-space "up" of all cameras (COLMAP cameras look down +z with y down,
    so up is ``R^T @ (0, -1, 0)``).  People hold phones upright, which makes this a robust
    gravity estimate.  None when the model is empty."""
    ups = []
    for _name, q, _t in read_images_bin(images_bin):
        R = _quat_to_R(q)
        ups.append(R.T @ np.array([0.0, -1.0, 0.0]))
    if not ups:
        return None
    v = np.mean(ups, axis=0)
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-6 else None


def camera_centers(images_bin: str | os.PathLike[str]) -> np.ndarray:
    """World positions (``Nx3``) of the registered cameras: ``C = -R^T t``."""
    cs = [-_quat_to_R(q).T @ t for _name, q, t in read_images_bin(images_bin)]
    return np.asarray(cs, dtype=np.float64).reshape(-1, 3)


def _best_model(sparse: Path) -> tuple[Path | None, int]:
    best, best_n = None, 0
    if sparse.is_dir():
        for sub in sorted(sparse.iterdir()):
            if sub.is_dir():
                n = count_registered(sub)
                if n > best_n:
                    best, best_n = sub, n
    return best, best_n


# --------------------------------------------------------------------------- PLY xyz probe
def ply_xyz(path: str | os.PathLike[str], max_n: int = 2000) -> np.ndarray:
    """First ``max_n`` vertex positions of a binary little-endian PLY (float properties)."""
    with open(path, "rb") as fh:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = fh.readline()
            if not line:
                raise AddonError(f"Invalid PLY file: {Path(path).name}")
            header += line
        lines = header.decode("ascii", errors="replace").splitlines()
        if not any(ln.startswith("format binary_little_endian") for ln in lines):
            raise AddonError("Only binary little-endian PLY files are supported.")
        n = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        types = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
                 "int": "<i4", "int32": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2", "char": "i1"}
        for ln in lines:
            parts = ln.split()
            if parts[:2] == ["element", "vertex"]:
                n, in_vertex = int(parts[2]), True
            elif parts and parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts and parts[0] == "property":
                props.append((parts[2], types.get(parts[1], "<f4")))
        dt = np.dtype(props)
        k = min(n, max_n)
        arr = np.frombuffer(fh.read(dt.itemsize * k), dtype=dt, count=k)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def _map_vector_to_cloud(raw_xyz: np.ndarray, cloud_means: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Express a direction given in the PLY file frame in the frame ``read_splats`` returns
    (least-squares affine fit between raw and loaded positions; identity if they match)."""
    n = min(len(raw_xyz), len(cloud_means))
    if n < 4:
        return v
    A = np.hstack([raw_xyz[:n], np.ones((n, 1))])
    M, *_ = np.linalg.lstsq(A, cloud_means[:n].astype(np.float64), rcond=None)
    out = M[:3].T @ v
    nrm = np.linalg.norm(out)
    return out / nrm if nrm > 1e-9 else v


def _fit_affine(src: np.ndarray, dst: np.ndarray, max_n: int = 4096) -> np.ndarray | None:
    """4x3 matrix ``M`` with ``dst ~= [src, 1] @ M`` for corresponding rows, or None when the
    rows do not correspond or the fit is ambiguous (degenerate point set)."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = min(len(src), len(dst))
    if n < 4:
        return None
    idx = np.arange(n)
    if n > max_n:
        idx = np.random.default_rng(0).choice(n, max_n, replace=False)
    a, b = src[idx], dst[idx]
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[ok], b[ok]
    if len(a) < 4:
        return None
    extent = float(np.ptp(b, axis=0).max()) if len(b) else 0.0
    tol = 1e-4 * max(extent, 1e-9) + 1e-6
    if np.abs(a - b).max() <= tol:
        return np.vstack([np.eye(3), np.zeros((1, 3))])
    A = np.hstack([a, np.ones((len(a), 1))])
    M, _res, rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    if rank < 4 or not np.isfinite(M).all() or np.abs(A @ M - b).max() > tol * 10:
        return None
    return M


def _apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return np.hstack([pts, np.ones((len(pts), 1))]) @ M


def _orbit_radius(cams: np.ndarray) -> float:
    """Median horizontal distance of the cameras from the subject (the origin after orientation)."""
    return float(np.median(np.linalg.norm(cams[:, [0, 2]], axis=1)))


def _subject_height(means: np.ndarray, cams: np.ndarray | None) -> float:
    """Robust (2..98 percentile) height of the points in a vertical cylinder around the
    centre: ``0.3 x`` the camera orbit radius, or the central 30 % of the points."""
    xz = np.linalg.norm(means[:, [0, 2]], axis=1)
    central = np.zeros(len(means), bool)
    if cams is not None and len(cams):
        central = xz <= 0.3 * _orbit_radius(cams)
    if central.sum() < 20:
        central = xz <= np.percentile(xz, 30)
    if central.sum() < 2:
        return 0.0
    lo, hi = np.percentile(means[central, 1], [2, 98])
    return float(hi - lo)


def _valid_scale(s: float | None) -> bool:
    return s is not None and math.isfinite(s) and 1e-4 < s < 1e4


def metric_scale(
    means: np.ndarray,
    cams: np.ndarray | None,
    *,
    subject_height_m: float = 0.0,
    camera_height_m: float = CAMERA_HEIGHT_M,
) -> tuple[float, str]:
    """Scale factor that makes an oriented capture (y up, floor at y = 0, subject on the
    y axis) metric, and the method used: ``"subject"`` (subject made ``subject_height_m``
    tall), ``"camera"`` (median camera ``camera_height_m`` above the floor), ``"radius"``
    (cameras orbit at :data:`ORBIT_RADIUS_M`) or ``"none"`` (scale 1)."""
    means = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    means = means[np.isfinite(means).all(axis=1)]
    if cams is not None:
        cams = np.asarray(cams, dtype=np.float64).reshape(-1, 3)
        cams = cams[np.isfinite(cams).all(axis=1)]
        if len(cams) < 2:
            cams = None
    if subject_height_m and subject_height_m > 0 and len(means) >= 2:
        h = _subject_height(means, cams)
        s = subject_height_m / h if h > 1e-9 else None
        if _valid_scale(s):
            return float(s), "subject"
        log.warning("Could not measure the subject height (%.4g); using the camera height instead", h)
    if cams is not None:
        heights = cams[:, 1]
        med_h = float(np.median(heights))
        r = _orbit_radius(cams)
        # usable when (nearly) all cameras are above the floor and not implausibly low
        if med_h > 0 and float(np.mean(heights > 0)) >= 0.8 and med_h >= 0.1 * r:
            s = camera_height_m / med_h
            if _valid_scale(s):
                return float(s), "camera"
        if r > 1e-9:
            s = ORBIT_RADIUS_M / r
            if _valid_scale(s):
                return float(s), "radius"
    return 1.0, "none"


def _scale_cloud(sc: Any, s: float) -> Any:
    if s == 1.0:
        return sc
    return replace(
        sc,
        means=(np.asarray(sc.means, dtype=np.float64) * s).astype(np.float32),
        scales=(np.asarray(sc.scales, dtype=np.float64) * s).astype(np.float32),
    )


def _bbox_height(sc: Any) -> float:
    y = np.asarray(sc.means, dtype=np.float64)[:, 1] if len(sc) else np.zeros(1)
    y = y[np.isfinite(y)]
    return float(y.max() - y.min()) if len(y) else 0.0


def _splat_api() -> Any:
    """Lazy import of :mod:`vrphoto.geometry.splat` (tests monkeypatch this)."""
    try:
        from ..geometry import splat
    except ImportError as e:
        raise AddonError("Splat processing is unavailable in this build.", hint=str(e)) from e
    return splat


# --------------------------------------------------------------------------- the pipeline
_REG_RE = re.compile(r"Registering image #\d+ \((?:num_reg_frames=)?(\d+)\)")
_PROC_RE = re.compile(r"Process(?:ed file|ing image) \[(\d+)/(\d+)\]")
_BLOCK_RE = re.compile(r"Processing block \[(\d+)/(\d+)(?:, (\d+)/(\d+))?\]")
_BRUSH_ITER_RE = re.compile(r"(?:Refine|Eval) iter (\d+)")


class _Runner:
    def __init__(self, sink: LogSink, cancel: CancelToken | None, timeout: float | None):
        self.sink = sink
        self.cancel = cancel
        self.deadline = (time.monotonic() + timeout) if timeout else None

    def remaining(self) -> float | None:
        """Seconds left before the deadline (at least 1), or None without a timeout."""
        if self.deadline is None:
            return None
        return max(1.0, self.deadline - time.monotonic())

    def __call__(self, cmd: list[str], title: str, on_line: Callable[[str], None] | None = None,
                 cwd: Path | None = None) -> ProcResult:
        return run_process(cmd, env=_tool_env(cmd), cwd=cwd, log_sink=self.sink, on_line=on_line,
                           cancel=self.cancel, timeout=self.remaining(), title=title)


def colmap_feature_args(db: Path, images: Path, *, use_gpu: bool, info: ColmapInfo) -> list[str]:
    """Arguments for ``colmap feature_extractor``."""
    gpu_opt = "FeatureExtraction.use_gpu" if info.new_option_names else "SiftExtraction.use_gpu"
    args = [
        "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images),
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.camera_model", "OPENCV",
        f"--{gpu_opt}", "1" if use_gpu else "0",
    ]
    return args + (["--log_target", "stderr"] if info.has_log_target else [])


def colmap_matcher_args(db: Path, *, sequential: bool, use_gpu: bool, info: ColmapInfo, overlap: int = 10) -> list[str]:
    """Arguments for ``sequential_matcher`` (video) or ``exhaustive_matcher`` (photos)."""
    gpu_opt = "FeatureMatching.use_gpu" if info.new_option_names else "SiftMatching.use_gpu"
    if sequential:
        args = ["sequential_matcher", "--database_path", str(db),
                "--SequentialMatching.overlap", str(overlap), "--SequentialMatching.quadratic_overlap", "1"]
    else:
        args = ["exhaustive_matcher", "--database_path", str(db)]
    args += [f"--{gpu_opt}", "1" if use_gpu else "0"]
    return args + (["--log_target", "stderr"] if info.has_log_target else [])


def colmap_mapper_args(db: Path, images: Path, sparse: Path, *, global_mapper: bool, info: ColmapInfo) -> list[str]:
    """Arguments for ``global_mapper`` (COLMAP >= 4) or the incremental ``mapper``."""
    args = ["global_mapper" if global_mapper else "mapper",
            "--database_path", str(db), "--image_path", str(images), "--output_path", str(sparse)]
    return args + (["--log_target", "stderr"] if info.has_log_target else [])


def colmap_undistort_args(images: Path, model: Path, out: Path, *, info: ColmapInfo,
                          max_size: int = UNDISTORT_MAX_PX) -> list[str]:
    """Arguments for ``image_undistorter`` (COLMAP output layout: images/ + sparse/)."""
    args = ["image_undistorter", "--image_path", str(images), "--input_path", str(model),
            "--output_path", str(out), "--output_type", "COLMAP", "--max_image_size", str(max_size)]
    return args + (["--log_target", "stderr"] if info.has_log_target else [])


def brush_args(dataset: Path, export_dir: Path, *, steps: int, max_splats: int, export_name: str = "splat.ply",
               seed: int = 42, extra: Sequence[str] = ()) -> list[str]:
    """Arguments for a headless Brush training run with a single export at the end.

    ``extra`` (e.g. ``["--sh-degree", "0"]``) is appended as given."""
    return [
        str(dataset),
        "--total-steps", str(int(steps)),
        "--export-every", str(int(steps)),
        "--eval-every", str(int(steps) + 1),
        "--export-path", str(export_dir),
        "--export-name", export_name,
        "--max-splats", str(int(max_splats)),
        "--seed", str(int(seed)),
        *(str(a) for a in extra),
    ]


def _fail_colmap(stage: str, res: ProcResult) -> AddonError:
    tail = res.tail_text(6)
    if re.search(r"CUDA|cuda|SiftGPU|OpenGL", tail):
        hint = "COLMAP's GPU code failed. Update your NVIDIA graphics driver; details are in the capture log."
    else:
        hint = "See the capture log for details. Make sure the photos overlap well and are sharp."
    return AddonError(f"COLMAP {stage} failed (exit code {res.returncode}).", hint=hint)


# --------------------------------------------------------------------------- known-camera datasets
def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion ``(w, x, y, z)`` with ``w >= 0`` (Shepperd's method,
    stable for every rotation; the inverse of :func:`_quat_to_R`)."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0:
        s = 2.0 * math.sqrt(1.0 + tr)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-300))
        v = np.zeros(3)
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = np.array([(R[k, j] - R[j, k]) / s, *v])
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def _check_camera(name: str, K: Any, R: Any, t: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if not name or name != name.strip() or any(c.isspace() for c in name) or "/" in name or "\\" in name:
        # COLMAP's text format splits on spaces and Brush looks the name up in images/
        raise InputError(f"Invalid image name for the training data: {name!r}.",
                         hint="Use a plain file name without spaces or folders.")
    if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,):
        raise InputError(f"Camera of {name}: K and R must be 3x3 and t a 3-vector.")
    if not (np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(t).all()):
        raise InputError(f"Camera of {name} contains NaN or infinite values.")
    pinhole = K[0, 0] > 0 and K[1, 1] > 0 and abs(K[0, 1]) <= 1e-9 * K[0, 0]
    if not pinhole or np.abs(K[2] - [0, 0, 1]).max() > 1e-12:
        raise InputError(f"Camera of {name} is not a plain pinhole camera (positive focal lengths, no skew).")
    if np.abs(R @ R.T - np.eye(3)).max() > 1e-4 or np.linalg.det(R) <= 0:
        raise InputError(f"Camera of {name}: R is not a rotation matrix.")
    return K, R, t


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except (OSError, ValueError) as e:
        raise InputError(
            f"Cannot read the size of the training image {path.name}.",
            hint="Write the images first or give each camera its (width, height).",
        ) from e


def write_colmap_text(
    dataset_dir: str | os.PathLike[str],
    cams: Sequence[Sequence[Any]],
    points: np.ndarray | None = None,
    colors: np.ndarray | None = None,
) -> Path:
    """Write a COLMAP text model ``<dataset_dir>/sparse/0/{cameras,images,points3D}.txt``.

    This is the known-camera layout Brush reads (with ``images/``, optional ``masks/`` and an
    ``init.ply`` next to ``sparse/``).

    ``cams``: one ``(name, K, R, t)`` per image, optionally ``(name, K, R, t, (width, height))``.
    ``K`` is the 3x3 pinhole matrix; ``R``, ``t`` map world to camera in the OpenCV convention
    (``x_cam = R @ x_world + t``; x right, y down, z forward), exactly COLMAP's
    ``cam_from_world``.  ``name`` is the file name inside ``<dataset_dir>/images``; without an
    explicit size it is read from that file.  Every image gets its own ``PINHOLE`` camera
    (``W H fx fy cx cy``); the second line of each image (its 2D points) is left empty.
    ``points`` (N x 3, world) and ``colors`` (N x 3, uint8 or floats in [0, 1]) become
    ``points3D.txt`` with empty tracks.  Cameras are written with 17 significant digits, so a
    point projects through the written camera as through the given one (to rounding error).

    Returns the model folder (``sparse/0``).
    """
    ds = Path(dataset_dir)
    if not cams:
        raise InputError("No cameras were given for the training data.")
    cam_lines: list[str] = []
    img_lines: list[str] = []
    seen: set[str] = set()
    for i, cam in enumerate(cams, start=1):
        if len(cam) not in (4, 5):
            raise InputError("Each camera must be (name, K, R, t) or (name, K, R, t, (width, height)).")
        name = str(cam[0])
        K, R, t = _check_camera(name, cam[1], cam[2], cam[3])
        if name in seen:
            raise InputError(f"The image name {name} is used twice; every view needs its own file.")
        seen.add(name)
        w, h = (int(cam[4][0]), int(cam[4][1])) if len(cam) == 5 else _image_size(ds / "images" / name)
        if w <= 0 or h <= 0:
            raise InputError(f"Invalid image size {w}x{h} for {name}.")
        g = "{:.17g}".format
        cam_lines.append(f"{i} PINHOLE {w} {h} {g(K[0, 0])} {g(K[1, 1])} {g(K[0, 2])} {g(K[1, 2])}")
        q = _rot_to_quat(R)
        img_lines.append(" ".join([str(i), *map(g, q), *map(g, t), str(i), name]))
        img_lines.append("")
    pts = np.zeros((0, 3)) if points is None else np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if colors is None:
        rgb = np.full((len(pts), 3), 128, dtype=np.int64)
    else:
        c = np.asarray(colors)
        if c.shape != pts.shape:
            raise InputError(f"points ({pts.shape}) and colors ({c.shape}) must have the same shape.")
        rgb = np.clip(c if c.dtype.kind in "iu" else np.rint(c * 255.0), 0, 255).astype(np.int64)
    ok = np.isfinite(pts).all(axis=1)
    pts, rgb = pts[ok], rgb[ok]

    model = ds / "sparse" / "0"
    model.mkdir(parents=True, exist_ok=True)
    head = "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
    text = head + f"# Number of cameras: {len(cam_lines)}\n" + "\n".join(cam_lines) + "\n"
    atomic_write_bytes(model / "cameras.txt", text.encode())
    head = ("# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    text = head + f"# Number of images: {len(cams)}\n" + "\n".join(img_lines) + "\n"
    atomic_write_bytes(model / "images.txt", text.encode())
    table = np.column_stack([np.arange(1, len(pts) + 1), pts, rgb, np.zeros(len(pts))])
    buf = io.StringIO()
    np.savetxt(buf, table, fmt=["%d", "%.9g", "%.9g", "%.9g", "%d", "%d", "%d", "%.1f"])
    head = ("# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            f"# Number of points: {len(pts)}\n")
    atomic_write_bytes(model / "points3D.txt", (head + buf.getvalue()).encode())
    return model


# --------------------------------------------------------------------------- Brush training
def train_splat(
    dataset: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    steps: int,
    max_splats: int,
    extra_args: Sequence[str] = (),
    export_name: str = "splat.ply",
    seed: int = 0,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    timeout: float | None = None,
    log_name: str = "brush.log",
    log_sink: LogSink | None = None,
    cwd: str | os.PathLike[str] | None = None,
    label: str = "Training the splat",
) -> Path:
    """Train a Gaussian splat with Brush on a prepared dataset; returns the exported ``.ply``.

    ``dataset`` is a folder Brush reads (COLMAP ``sparse/`` + ``images/``, optionally
    ``masks/`` and ``init.ply``, see :func:`write_colmap_text`); ``out_dir`` receives the
    single export ``export_name`` and should lie outside the dataset folder, so an export
    is never read back as an input.  ``extra_args`` are appended to the command line
    (:func:`brush_args`).  Progress messages read ``"<label> (step i/steps)"``.

    The log goes to ``out_dir/log_name`` unless ``log_sink`` is given (the capture pipeline
    passes its own); Brush runs in ``cwd`` (default ``out_dir``).  Honours ``cancel``
    (kills Brush) and ``timeout`` (seconds).  Raises :class:`AddonNotInstalled` without
    Brush and :class:`AddonError` (with an out-of-memory hint when that is the cause) when
    training fails or nothing is exported.
    """
    progress = progress or ProgressReporter.null()
    dataset, out_dir = Path(dataset), Path(out_dir)
    steps, max_splats = int(steps), int(max_splats)
    if not dataset.is_dir():
        raise InputError(f"The training data folder does not exist: {dataset}")
    if steps < 1 or max_splats < 1:
        raise InputError(f"Training steps and max splats must be positive (got {steps}, {max_splats}).")
    brush = find_brush()
    out_dir.mkdir(parents=True, exist_ok=True)
    ply = out_dir / export_name
    with contextlib.suppress(OSError):
        ply.unlink()  # a stale export must never pass for this run's result
    sink = log_sink if log_sink is not None else LogSink(out_dir / log_name)
    try:
        if log_sink is None:
            sink.header(f"Brush training: {steps} steps, {dataset}")
        runner = _Runner(sink, cancel, timeout)

        def brush_line(line: str) -> None:
            m = _BRUSH_ITER_RE.search(line)
            if m:
                progress(int(m.group(1)) / steps, f"{label} (step {m.group(1)}/{steps})")

        t0 = time.time()
        args = brush_args(dataset, out_dir, steps=steps, max_splats=max_splats, export_name=export_name,
                          seed=seed, extra=extra_args)
        res = runner([*brush, *args], "Brush training", brush_line, cwd=Path(cwd) if cwd else out_dir)
        if not ply.is_file():
            # some Brush builds name the export themselves: take the newest .ply of this run
            cands = sorted((p for p in out_dir.glob("*.ply") if p.stat().st_mtime >= t0 - 2.0),
                           key=lambda p: p.stat().st_mtime)
            ply = cands[-1] if cands else ply
        if not res.ok or not ply.is_file():
            tail = res.tail_text(5)
            # "the capture log" for the capture pipeline's capture.log, "the brush log" otherwise
            what = sink.path.stem.replace("_", " ") if sink.path is not None else "Brush"
            if re.search(r"memory|OutOfMemory|allocation", tail, re.I):
                hint = ("Your GPU may have run out of memory: lower the training steps / max splats or "
                        "close other apps.")
            else:
                hint = f"See the {what} log for details; updating the GPU driver often helps."
            raise AddonError(f"Brush training failed (exit code {res.returncode}).", hint=hint)
        return ply
    finally:
        if log_sink is None:
            sink.close()


def capture_to_splat(
    inputs: list[Path],
    out_dir: Path,
    *,
    video_fps: float = 2.0,
    max_images: int = 200,
    steps: int = 15000,
    max_splats: int = 750_000,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    crop_pct: float = 98.0,
    name: str | None = None,
    use_gpu: bool | None = None,
    keep_workdir: bool = False,
    timeout: float | None = None,
    subject_height_m: float = 0.0,
    camera_height_m: float = CAMERA_HEIGHT_M,
) -> Path:
    """Reconstruct a Gaussian splat from photos and/or videos.

    Returns the ``.splat`` path; a ``.ply`` with the same stem is written next to it, and
    ``<name>.capture.json`` with the metric scale applied (see the module docstring).
    ``subject_height_m`` (> 0) makes the subject near the centre that tall; otherwise the
    scale comes from the camera heights (``camera_height_m`` above the floor).
    Long-running (minutes to an hour); honours ``cancel`` (kills COLMAP/Brush) and
    reports ``progress``.  Raises :class:`InputError` for unusable inputs and
    :class:`AddonError` (with hints) when reconstruction fails.
    """
    progress = progress or ProgressReporter.null()
    subject_height_m = float(subject_height_m or 0.0)
    camera_height_m = float(camera_height_m)
    if not math.isfinite(subject_height_m) or subject_height_m < 0 or subject_height_m > 1000:
        raise InputError(f"The subject height must be 0 (automatic) or a positive number of metres (got {subject_height_m}).")
    if not math.isfinite(camera_height_m) or not 0.05 <= camera_height_m <= 100:
        raise InputError(f"The camera height must be a positive number of metres (got {camera_height_m}).")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos, videos = _expand_inputs(inputs)
    if not photos and not videos:
        raise InputError("No photos or videos were given for the capture.")
    stem = re.sub(r"[^\w.-]+", "_", name or (videos[0].stem if videos else photos[0].parent.name or "capture"))
    stem = stem.strip("._") or "capture"
    colmap = find_colmap()
    find_brush()  # fail before the long COLMAP stage when Brush is missing
    work = _work_dir(out_dir, stem)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    images_dir = work / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sink = LogSink(work / "capture.log")
    sink.header(f"capture {len(photos)} photos, {len(videos)} videos")
    runner = _Runner(sink, cancel, timeout)
    succeeded = False
    try:
        # ---------------------------------------------------------------- 1. images
        # One sub-folder per camera (each video; photos grouped by camera/lens) so COLMAP
        # can share intrinsics per folder (ImageReader.single_camera_per_folder).
        p_frames = progress.sub(0.0, 0.10)
        budget = int(max_images) if max_images else 0
        photo_budget = min(len(photos), budget // 2 if videos else budget) if budget else len(photos)
        video_budget = (budget - photo_budget) if budget else 0
        if videos:
            per_video = max(MIN_IMAGES, video_budget // len(videos)) if budget else None
            for vi, video in enumerate(videos):
                raw_dir = work / f"frames_{vi}"
                frames = extract_frames(video, raw_dir, fps=float(video_fps) * 2.0, prefix=f"v{vi}",
                                        cancel=cancel, log_sink=sink,
                                        progress=p_frames.sub(vi / len(videos), (vi + 0.8) / len(videos)))
                if cancel is not None:
                    cancel.check()
                keep = select_sharp(frames, group=2, max_images=per_video)
                sink.write(f"{video.name}: kept {len(keep)} of {len(frames)} frames (sharpness filter)")
                cam_dir = images_dir / f"video{vi}"
                cam_dir.mkdir(exist_ok=True)
                for f in keep:
                    os.replace(f, cam_dir / f.name)
                shutil.rmtree(raw_dir, ignore_errors=True)
        if photos:
            photos = sorted(photos, key=_photo_sort_key)
            if photo_budget and len(photos) > photo_budget:
                idx = np.linspace(0, len(photos) - 1, photo_budget).round().astype(int)
                photos = [photos[i] for i in sorted(set(idx.tolist()))]
            groups: dict[tuple, int] = {}
            for i, ph in enumerate(photos):
                if cancel is not None:
                    cancel.check()
                key = _camera_key(ph)
                cam_dir = images_dir / f"photos{groups.setdefault(key, len(groups))}"
                cam_dir.mkdir(exist_ok=True)
                safe = _ascii_name(ph.stem)
                _import_photo(ph, cam_dir / f"p{i:05d}_{safe}")
                p_frames(0.8 + 0.2 * (i + 1) / len(photos), "Preparing photos")
        n_images = sum(1 for f in images_dir.rglob("*") if f.is_file())
        if n_images < MIN_IMAGES:
            raise InputError(
                f"Only {n_images} usable images - at least {MIN_IMAGES} are needed (30+ recommended).",
                hint="Take more photos from many angles around the subject, or record a slower, longer video.",
            )
        progress(0.10, f"{n_images} images ready")

        # ---------------------------------------------------------------- 2. COLMAP
        info = colmap_info(colmap)
        gpu = _has_nvidia_gpu() if use_gpu is None else bool(use_gpu)
        db = work / "database.db"
        sequential = bool(videos) or n_images > EXHAUSTIVE_MAX

        p_feat = progress.sub(0.10, 0.25)

        def feat_line(line: str) -> None:
            m = _PROC_RE.search(line)
            if m:
                p_feat(int(m.group(1)) / max(int(m.group(2)), 1), "Finding features (COLMAP)")

        res = runner([*colmap, *colmap_feature_args(db, images_dir, use_gpu=gpu, info=info)], "COLMAP features", feat_line)
        if not res.ok and gpu:
            sink.write("GPU feature extraction failed; retrying on the CPU")
            with contextlib.suppress(OSError):
                db.unlink()
            gpu = False
            res = runner([*colmap, *colmap_feature_args(db, images_dir, use_gpu=False, info=info)],
                         "COLMAP features (CPU)", feat_line)
        if not res.ok:
            raise _fail_colmap("feature extraction", res)

        p_match = progress.sub(0.25, 0.40)

        def match_line(line: str) -> None:
            m = _PROC_RE.search(line)
            if m:
                p_match(int(m.group(1)) / max(int(m.group(2)), 1), "Matching images (COLMAP)")
                return
            m = _BLOCK_RE.search(line)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if m.group(3):
                    c, d = int(m.group(3)), int(m.group(4))
                    frac = ((a - 1) * d + c) / max(b * d, 1)
                else:
                    frac = a / max(b, 1)
                p_match(frac, "Matching images (COLMAP)")

        res = runner([*colmap, *colmap_matcher_args(db, sequential=sequential, use_gpu=gpu, info=info,
                                                     overlap=10 if videos else 20)], "COLMAP matching", match_line)
        if not res.ok and gpu:
            gpu = False
            res = runner([*colmap, *colmap_matcher_args(db, sequential=sequential, use_gpu=False, info=info,
                                                         overlap=10 if videos else 20)],
                         "COLMAP matching (CPU)", match_line)
        if not res.ok:
            raise _fail_colmap("matching", res)
        if cancel is not None:
            cancel.check()

        p_map = progress.sub(0.40, 0.55)
        sparse = work / "sparse"

        def map_line(line: str) -> None:
            m = _REG_RE.search(line)
            if m:
                p_map(min(0.95, int(m.group(1)) / n_images), "Solving camera positions (COLMAP)")

        model, n_reg = None, 0
        mappers = [True, False] if info.has_global_mapper else [False]
        for use_global in mappers:
            # each attempt writes its own folder so a worse fallback never replaces a
            # better first result
            attempt_dir = sparse / ("global" if use_global else "incremental")
            shutil.rmtree(attempt_dir, ignore_errors=True)
            attempt_dir.mkdir(parents=True)
            p_map(0.02, "Solving camera positions (COLMAP)")
            res = runner([*colmap, *colmap_mapper_args(db, images_dir, attempt_dir, global_mapper=use_global,
                                                        info=info)],
                         "COLMAP global mapper" if use_global else "COLMAP mapper", map_line)
            cand, cand_n = _best_model(attempt_dir)
            sink.write(f"{'global' if use_global else 'incremental'} mapper: exit {res.returncode}, "
                       f"{cand_n}/{n_images} images registered")
            if cand is not None and cand_n > n_reg:
                model, n_reg = cand, cand_n
            if model is not None and n_reg >= max(MIN_IMAGES, int(0.5 * n_images)):
                break
            if cancel is not None:
                cancel.check()
        min_reg = max(MIN_IMAGES, int(0.2 * n_images))
        if model is None or n_reg < min_reg:
            raise AddonError(
                f"COLMAP could not register enough images ({n_reg} of {n_images}) - take more overlapping photos.",
                hint="Move slowly around the subject, keep 60-80% overlap between shots, avoid motion blur, "
                "reflective/plain surfaces and moving people in the background.",
            )
        progress(0.55, f"{n_reg} of {n_images} cameras solved")

        undist = work / "undistorted"
        res = runner([*colmap, *colmap_undistort_args(images_dir, model, undist, info=info)], "COLMAP undistortion")
        if not res.ok or not (undist / "sparse").is_dir():
            raise _fail_colmap("image undistortion", res)
        images_bin = undist / "sparse" / "images.bin"
        if not images_bin.is_file():
            images_bin = model / "images.bin"
        up_raw = camera_up_vector(images_bin)
        try:
            cams_raw: np.ndarray | None = camera_centers(images_bin)
        except (OSError, struct.error, ValueError) as e:
            log.debug("cannot read the camera positions (%s)", e)
            cams_raw = None
        progress(0.60, "Training the splat (Brush)")
        if cancel is not None:
            cancel.check()

        # ---------------------------------------------------------------- 3. Brush
        train_cap = int(min(10_000_000, max(1_000_000, 2 * int(max_splats))))
        ply = train_splat(undist, work / "brush", steps=steps, max_splats=train_cap, seed=42,
                          progress=progress.sub(0.60, 0.95), cancel=cancel, timeout=runner.remaining(),
                          log_sink=sink, cwd=work)

        # ---------------------------------------------------------------- 4. post-process
        progress(0.95, "Cleaning up the splat")
        api = _splat_api()
        sc = api.read_splats(ply)
        if len(sc) == 0:
            raise AddonError("The trained splat is empty.", hint="Try more images or more training steps.")
        up = None
        raw_xyz: np.ndarray | None = None
        try:
            raw_xyz = ply_xyz(ply)
        except (AddonError, ValueError, KeyError, OSError) as e:
            log.debug("cannot read the raw PLY positions (%s)", e)
        if up_raw is not None:
            up = up_raw
            if raw_xyz is not None:
                try:
                    up = _map_vector_to_cloud(raw_xyz, np.asarray(sc.means), up_raw)
                except (ValueError, np.linalg.LinAlgError) as e:
                    log.debug("up-vector mapping failed (%s); using the raw vector", e)
        # camera positions: COLMAP world (= Brush's PLY frame) -> the reader's frame
        cams: np.ndarray | None = None
        if cams_raw is not None and len(cams_raw) and raw_xyz is not None:
            raw_ok = raw_xyz[np.isfinite(raw_xyz).all(axis=1)]  # the reader drops non-finite rows
            m_read = _fit_affine(raw_ok, np.asarray(sc.means)[: len(raw_ok)])
            cams = _apply_affine(m_read, cams_raw) if m_read is not None else None
        sc = api.crop_splats(sc, radius_pct=float(crop_pct))
        before = np.asarray(sc.means)
        sc = api.normalize_orientation(sc, up=up)
        if cams is not None:  # follow the rotation + floor shift of the orientation step
            m_orient = _fit_affine(before, np.asarray(sc.means)) if len(before) == len(sc) else None
            cams = _apply_affine(m_orient, cams) if m_orient is not None else None
        scale, method = metric_scale(np.asarray(sc.means), cams, subject_height_m=subject_height_m,
                                     camera_height_m=camera_height_m)
        sc = _scale_cloud(sc, scale)
        sc = api.decimate_splats(sc, max_n=int(max_splats))
        out_splat = Path(api.write_splats(sc, out_dir / f"{stem}.splat"))
        api.write_splats(sc, out_dir / f"{stem}.ply")
        height = _bbox_height(sc)
        info_scale: dict[str, Any] = {"scale": scale, "height_m": round(height, 4), "method": method}
        if method == "subject":  # the viewer labels the object with its subject's real height
            info_scale["subject_height_m"] = round(subject_height_m, 4)
        if cams is not None and len(cams):
            info_scale["camera_height_m"] = round(float(np.median(cams[:, 1])) * scale, 4)
        atomic_write_bytes(out_dir / f"{stem}.capture.json", json.dumps(info_scale, indent=2).encode("utf-8"))
        log.info("Capture scale x%.4g (%s): the splat is %.2f m tall", scale, method, height)
        sink.write(f"metric scale x{scale:.6g} ({method}); bounding box height {height:.3f} m")
        progress(1.0, f"Splat ready ({len(sc):,} splats)")
        sink.write(f"wrote {out_splat} ({len(sc)} splats)")
        succeeded = True
        return out_splat
    finally:
        sink.close()
        if not keep_workdir:
            # on failure keep the log next to the outputs for diagnostics; drop the
            # (large) intermediate data in any case
            logf = work / "capture.log"
            if not succeeded and logf.is_file():
                with contextlib.suppress(OSError):
                    shutil.copy2(logf, out_dir / f"{stem}_capture.log")
            shutil.rmtree(work, ignore_errors=True)


def run_task(
    task: str,
    inputs: dict[str, Any],
    params: dict[str, Any],
    out_dir: Path,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    timeout: float | None = None,
) -> dict[str, str]:
    """Entry point used by :func:`vrphoto.addons.manager.run` for the ``capture`` add-on.

    ``inputs``: ``{"paths": [...]}`` (photos, videos or folders).  ``params``: any keyword
    of :func:`capture_to_splat`.  Returns ``{"splat": ..., "ply": ...}``.
    """
    if task != "capture_to_splat":
        raise AddonError(f"Unknown capture task '{task}'.")
    paths = inputs.get("paths") or inputs.get("inputs") or []
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    allowed = {"video_fps", "max_images", "steps", "max_splats", "crop_pct", "name", "use_gpu", "keep_workdir",
               "subject_height_m", "camera_height_m"}
    unknown = set(params) - allowed
    if unknown:
        raise InputError(f"Unknown capture option(s): {', '.join(sorted(unknown))}")
    out = capture_to_splat([Path(p) for p in paths], Path(out_dir), progress=progress, cancel=cancel,
                           timeout=timeout, **params)
    return {"splat": str(out), "ply": str(out.with_suffix(".ply"))}


# --------------------------------------------------------------------------- Brush self-test
# A short Brush run on an analytic known-camera dataset, done once on the user's PC: Brush
# 0.3.0 on Pascal GPUs, its SH-degree-0 bug and mixing masked with transparent pictures are
# unverified, and SHARP 360's optional Brush polish relies on all three
# (docs/design/SHARP360.md §4.5).  Like the polish recipe (§3.11) the views are orthographic
# pictures expressed as distant telephoto pinholes: centre 40 m back, f = 40 m / (metres per pixel).
SELFTEST_STEPS = 300
_SELFTEST_PX = 128
_SELFTEST_VIEWS = 8
_SELFTEST_MASKED = (1, 5)  # RGB + mask (black = ignore); the other views are RGBA
_SELFTEST_RADIUS_M = 0.5
_SELFTEST_FILL = 0.7  # sphere diameter as a fraction of the picture
_SELFTEST_ELEV_DEG = 12.0  # cameras alternate this far above / below the equator
_TELEPHOTO_DIST_M = 40.0
_SELFTEST_INIT_POINTS = 2000
_SELFTEST_MAX_SPLATS = 100_000
_SELFTEST_TIMEOUT_S = 600.0  # per run; the first run also compiles Brush's GPU shaders
_SELFTEST_BG = 0.5  # grey behind the masked views (ignored when the masks work)
_SELFTEST_PALETTE = np.array(
    [[0.9, 0.2, 0.2], [0.95, 0.8, 0.2], [0.2, 0.75, 0.3], [0.2, 0.7, 0.9], [0.25, 0.3, 0.9], [0.8, 0.3, 0.8]]
)
_STEP_RE = re.compile(r"\(step (\d+)/")


def _selftest_colour(n: np.ndarray) -> np.ndarray:
    """Colour of the test sphere at unit normals ``n`` (..., 3): one colour per 60° of
    longitude, darker on every other 30° latitude band, so every view sees a different,
    unambiguous pattern."""
    lon = np.arctan2(n[..., 0], n[..., 2])
    lat = np.arcsin(np.clip(-n[..., 1], -1.0, 1.0))
    sector = np.floor((lon + np.pi) / (np.pi / 3)).astype(np.int64) % 6
    band = np.floor((lat + np.pi / 2) / (np.pi / 6)).astype(np.int64) % 2
    return _SELFTEST_PALETTE[sector] * np.where(band == 0, 1.0, 0.65)[..., None]


def _selftest_cameras() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """``(name, K, R, t)`` (world to camera, OpenCV axes, world y down) of the test views:
    a circle of telephoto cameras around the sphere at the origin."""
    m = 2 * _SELFTEST_RADIUS_M / (_SELFTEST_FILL * _SELFTEST_PX)  # metres per pixel
    f = _TELEPHOTO_DIST_M / m
    K = np.array([[f, 0.0, _SELFTEST_PX / 2], [0.0, f, _SELFTEST_PX / 2], [0.0, 0.0, 1.0]])
    cams = []
    for k in range(_SELFTEST_VIEWS):
        az = 2 * np.pi * k / _SELFTEST_VIEWS
        el = np.radians(_SELFTEST_ELEV_DEG * (1 if k % 2 else -1))
        fwd = -np.array([np.cos(el) * np.sin(az), -np.sin(el), -np.cos(el) * np.cos(az)])
        right = np.cross([0.0, 1.0, 0.0], fwd)
        right /= np.linalg.norm(right)
        R = np.stack([right, np.cross(fwd, right), fwd])  # rows: camera x, y (down), z
        C = -_TELEPHOTO_DIST_M * fwd
        cams.append((f"view_{k:02d}.png", K, R, -R @ C))
    return cams


def _render_selftest_view(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                          ss: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Exact ray-sphere render of one view: straight (not premultiplied) RGB in [0, 1] and
    coverage alpha, box-filtered from ``ss`` x ``ss`` samples per pixel."""
    n_px = _SELFTEST_PX
    s = (np.arange(n_px * ss) + 0.5) / ss  # pixel centres at i + 0.5 (COLMAP / OpenCV)
    u, v = np.meshgrid(s, s)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1) @ R
    C = -R.T @ t
    a = (d * d).sum(-1)
    b = 2.0 * (d @ C)
    c = float(C @ C) - _SELFTEST_RADIUS_M**2
    disc = b * b - 4 * a * c
    hit = disc > 0
    depth = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2 * a)
    nrm = (C + depth[..., None] * d) / _SELFTEST_RADIUS_M
    col = _selftest_colour(nrm) * hit[..., None]
    cov = hit.reshape(n_px, ss, n_px, ss).mean(axis=(1, 3))
    rgb = col.reshape(n_px, ss, n_px, ss, 3).mean(axis=(1, 3)) / np.maximum(cov, 1e-9)[..., None]
    return np.clip(rgb, 0.0, 1.0), cov


def _write_png(path: Path, img: np.ndarray) -> None:
    import cv2

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA if img.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise AddonError(f"Could not encode {path.name}.")
    atomic_write_bytes(path, buf.tobytes())  # unicode-safe (cv2.imwrite is not on Windows)


def write_selftest_dataset(folder: str | os.PathLike[str]) -> dict[str, Any]:
    """Write the Brush self-test dataset into ``folder``: ``images/`` (RGBA views, plus the
    :data:`_SELFTEST_MASKED` views as RGB on grey with ``masks/``), the COLMAP text model and
    ``init.ply`` (points on the sphere, read back to validate).  Returns a summary."""
    ds = Path(folder)
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "masks").mkdir(exist_ok=True)
    cams = _selftest_cameras()
    for k, (name, K, R, t) in enumerate(cams):
        rgb, cov = _render_selftest_view(K, R, t)
        if k in _SELFTEST_MASKED:
            comp = rgb * cov[..., None] + _SELFTEST_BG * (1.0 - cov[..., None])
            _write_png(ds / "images" / name, np.rint(comp * 255).astype(np.uint8))
            _write_png(ds / "masks" / name, np.where(cov > 0, 255, 0).astype(np.uint8))
        else:
            rgba = np.concatenate([rgb, cov[..., None]], axis=-1)
            _write_png(ds / "images" / name, np.rint(rgba * 255).astype(np.uint8))
    # Fibonacci sphere: evenly spread initial splats on the true surface
    n = _SELFTEST_INIT_POINTS
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    az = np.pi * (1.0 + 5.0**0.5) * i
    nrm = np.stack([np.cos(az) * np.sin(polar), np.cos(polar), np.sin(az) * np.sin(polar)], axis=1)
    pts = _SELFTEST_RADIUS_M * nrm
    colors = _selftest_colour(nrm)
    write_colmap_text(ds, [(*c, (_SELFTEST_PX, _SELFTEST_PX)) for c in cams], pts, colors)
    from ..core.types import SplatCloud

    spacing = _SELFTEST_RADIUS_M * math.sqrt(4 * math.pi / n)
    init = SplatCloud(
        means=pts.astype(np.float32),
        scales=np.full((n, 3), 0.6 * spacing, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        colors=colors.astype(np.float32),
        opacity=np.full(n, 0.5, np.float32),
    )
    api = _splat_api()
    api.write_splats(init, ds / "init.ply")
    if len(api.read_splats(ds / "init.ply")) != n:
        raise AddonError("The self-test's initial splats could not be written correctly.")
    return {"views": len(cams), "masked": len(_SELFTEST_MASKED), "points": n, "size": _SELFTEST_PX}


def _brush_version(brush: Sequence[str], cancel: CancelToken | None = None) -> str:
    """Brush's version from ``--version`` (empty when it does not say)."""
    lines: list[str] = []
    try:
        res = run_process([*brush, "--version"], env=_tool_env(brush), timeout=30, title="Brush version",
                          on_line=lines.append, cancel=cancel)
    except AddonError as e:  # a version probe must never fail the test
        log.debug("brush --version failed: %s", e)
        return ""
    m = re.search(r"(\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?)", "\n".join(lines)) if res.ok else None
    return m.group(1) if m else ""


def _check_export(ply: Path) -> int:
    """Number of splats of a trained export; raises AddonError if it is empty or broken."""
    try:
        sc = _splat_api().read_splats(ply)
    except Exception as e:  # noqa: BLE001 - any read failure means a broken export
        raise AddonError(f"Brush exported an unreadable splat ({e}).") from e
    if len(sc) == 0 or not np.isfinite(np.asarray(sc.means)).all():
        raise AddonError("Brush exported an empty splat or one with invalid positions.")
    return len(sc)


def brush_selftest(
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    *,
    work_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Check that Brush (Capture add-on) trains on this PC; takes a minute or two.

    Writes the analytic dataset (:func:`write_selftest_dataset`) to a temporary folder (or
    ``work_dir``, which is then kept), then trains :data:`SELFTEST_STEPS` steps until a run
    works: with masks at ``--sh-degree`` 0, then 1; without masks at 0, then 1; and last
    with Brush's own defaults (no ``--sh-degree``, no masks).  An SH degree is only judged
    against a run with the same mask setting, so the result tells which part does not work.

    Returns ``{"ok", "sh0", "masks", "sh_option", "steps_per_s", "seconds", "splats",
    "steps", "brush_version", "attempts", "date", "log"}`` plus ``"error"`` when no run
    worked.  ``ok``: a run exported a valid splat; ``sh0``: True when an SH-degree-0 run
    worked, False when it failed and SH degree 1 worked with the same masks setting, None
    when only the defaults run worked (unknown); ``masks``: a run with masked views worked;
    ``sh_option``: Brush accepted ``--sh-degree``; ``steps_per_s`` is timed from Brush's
    first progress line to the end (start-up and shader compilation excluded when possible);
    ``seconds`` is the wall time of the successful run.  Training failures are reported in the
    result, not raised; a missing Brush raises :class:`AddonNotInstalled` and ``cancel``
    raises :class:`Cancelled`.
    The caller stores the result (``Settings.brush_test``); see :func:`brush_test_verdict`.
    """
    import tempfile

    from ..core.config import data_root
    from ..core.errors import Cancelled

    progress = progress or ProgressReporter.null()
    t_start = time.monotonic()
    brush = find_brush()
    root = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="vrphoto_brush_test_"))
    log_path = data_root() / "logs" / "brush_test.log"
    sink = LogSink(log_path)
    sink.header("Brush self-test")
    result: dict[str, Any] = {
        "ok": False,
        "sh0": False,
        "masks": False,
        "sh_option": False,
        "steps_per_s": None,
        "seconds": 0.0,
        "splats": 0,
        "steps": SELFTEST_STEPS,
        "brush_version": "",
        "attempts": [],
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "log": str(log_path),
    }
    try:
        progress(0.0, "Writing the test pictures")
        ds = root / "dataset"
        if ds.exists():
            shutil.rmtree(ds)
        info = write_selftest_dataset(ds)
        sink.write(f"dataset: {info} in {ds}")
        result["brush_version"] = _brush_version(brush, cancel)
        if cancel is not None:
            cancel.check()
        # (SH degree or None = Brush's default, masks, progress range).  Each SH degree is
        # tried with and without masks before the next mask setting: a failure is only blamed
        # on SH degree 0 when degree 1 then works with the same masks setting
        plan: list[tuple[int | None, bool, float, float]] = [
            (0, True, 0.05, 0.6),
            (1, True, 0.6, 0.72),
            (0, False, 0.72, 0.82),
            (1, False, 0.82, 0.91),
            (None, False, 0.91, 0.98),
        ]
        error = ""
        for i, (sh, with_masks, lo, hi) in enumerate(plan):
            if not with_masks and (ds / "masks").is_dir():
                shutil.rmtree(root / "masks_off", ignore_errors=True)
                os.replace(ds / "masks", root / "masks_off")  # Brush picks masks up by folder name
            samples: list[tuple[float, int]] = []
            outer = progress.sub(lo, hi)

            def on_progress(frac: float, msg: str, samples=samples, outer=outer) -> None:
                m = _STEP_RE.search(msg)
                if m:
                    samples.append((time.monotonic(), int(m.group(1))))
                outer(frac, msg)

            what = ("Brush's default settings" if sh is None else f"SH degree {sh}") + (
                "" if with_masks else ", no masks"
            )
            if i:
                sink.write(f"retrying with {what}")
                progress(lo, f"Testing Brush again ({what})")
            t0 = time.monotonic()
            attempt: dict[str, Any] = {"sh_degree": sh, "masks": with_masks, "ok": False}
            try:
                ply = train_splat(
                    ds, root / f"out_{i}", steps=SELFTEST_STEPS, max_splats=_SELFTEST_MAX_SPLATS,
                    extra_args=[] if sh is None else ["--sh-degree", str(sh)],
                    export_name="selftest.ply", seed=0,
                    progress=ProgressReporter(on_progress), cancel=cancel, timeout=_SELFTEST_TIMEOUT_S,
                    log_sink=sink, cwd=root, label="Testing Brush",
                )
                result["splats"] = _check_export(ply)
            except Cancelled:
                raise
            except AddonError as e:
                error = str(e)
                attempt.update(seconds=round(time.monotonic() - t0, 2), error=error)
                result["attempts"].append(attempt)
                sink.write(f"{what}: failed - {error}")
                if time.monotonic() - t0 >= _SELFTEST_TIMEOUT_S - 1:
                    break  # a hang is no SH or mask problem: retrying would only wait again
                continue
            t_end = time.monotonic()
            secs = t_end - t0
            attempt.update(ok=True, seconds=round(secs, 2))
            result["attempts"].append(attempt)
            rate = SELFTEST_STEPS / secs if secs > 0 else None
            # Brush logs only every few hundred steps: time the steps after its first progress
            # line, which leaves out start-up and shader compilation
            if samples and samples[0][1] < SELFTEST_STEPS and t_end - samples[0][0] >= 0.5:
                rate = (SELFTEST_STEPS - samples[0][1]) / (t_end - samples[0][0])
            result.update(ok=True, sh0=None if sh is None else sh == 0, masks=with_masks,
                          sh_option=sh is not None, seconds=round(secs, 2),
                          steps_per_s=None if rate is None else round(rate, 2))
            break
        if not result["ok"]:
            result["error"] = error or "Brush did not run."
            result["seconds"] = round(time.monotonic() - t_start, 2)
        sink.write(f"result: {json.dumps({k: v for k, v in result.items() if k != 'attempts'})}")
        log.info("Brush self-test: ok=%s sh0=%s masks=%s %.1f steps/s", result["ok"], result["sh0"],
                 result["masks"], result["steps_per_s"] or 0.0)
        progress(1.0, "Brush works" if result["ok"] else "Brush test failed")
        return result
    finally:
        sink.close()
        if work_dir is None:
            shutil.rmtree(root, ignore_errors=True)


def brush_test_verdict(result: dict[str, Any]) -> str:
    """Plain-language summary of a :func:`brush_selftest` result."""
    if not result:
        return "Brush has not been tested on this PC yet."
    if not result.get("ok"):
        err = str(result.get("error") or "unknown error").strip()
        return (
            f"Brush could not train on this PC ({err}). SHARP 360's optional Brush polishing will "
            "not be offered; everything else works without it. The details are in "
            f"{result.get('log') or 'brush_test.log'}."
        )
    steps = int(result.get("steps") or SELFTEST_STEPS)
    rate = result.get("steps_per_s")
    speed = f" ({float(rate):.1f} steps per second)" if rate else ""
    secs = float(result.get("seconds") or 0)
    text = f"Brush works on this PC: {steps} training steps took {secs:.0f} s{speed}."
    sh0 = result.get("sh0")
    if result.get("sh_option") is False and sh0 is None:
        text += (" It only trained with its default settings: it did not accept the SH degree option"
                 " (--sh-degree), so polishing will leave that option out.")
    elif sh0:
        text += " Plain-colour training (SH degree 0) works."
    elif sh0 is None:
        text += " Whether plain-colour training (SH degree 0) works is not known."
    else:
        text += (" Plain-colour training (SH degree 0) failed, so Brush will train with SH degree 1"
                 " (a little slower).")
    if result.get("masks"):
        text += " Masked and transparent pictures can be mixed."
    else:
        text += " Masked pictures could not be mixed with transparent ones, so polishing will not use masks."
    return text


__all__ = [
    "ColmapInfo",
    "SELFTEST_STEPS",
    "brush_args",
    "brush_selftest",
    "brush_test_verdict",
    "camera_centers",
    "camera_up_vector",
    "capture_to_splat",
    "colmap_feature_args",
    "colmap_info",
    "colmap_mapper_args",
    "colmap_matcher_args",
    "colmap_undistort_args",
    "count_registered",
    "extract_frames",
    "find_brush",
    "find_colmap",
    "metric_scale",
    "ply_xyz",
    "read_images_bin",
    "run_task",
    "select_sharp",
    "sharpness",
    "train_splat",
    "write_colmap_text",
    "write_selftest_dataset",
]
