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
   average camera "up" becomes +y with the floor at y = 0, decimate, write ``.splat`` and
   ``.ply``.

The tools come from the ``capture`` add-on (``manifests/capture.toml``); ``VRPHOTO_COLMAP`` /
``VRPHOTO_BRUSH`` override their paths (a ``.py`` path runs with the current Python -
used by the tests with fake tools).
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import struct
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import AddonError, AddonNotInstalled, InputError
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

    def __call__(self, cmd: list[str], title: str, on_line: Callable[[str], None] | None = None,
                 cwd: Path | None = None) -> ProcResult:
        remaining = None
        if self.deadline is not None:
            remaining = max(1.0, self.deadline - time.monotonic())
        return run_process(cmd, env=_tool_env(cmd), cwd=cwd, log_sink=self.sink, on_line=on_line,
                           cancel=self.cancel, timeout=remaining, title=title)


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
               seed: int = 42) -> list[str]:
    """Arguments for a headless Brush training run with a single export at the end."""
    return [
        str(dataset),
        "--total-steps", str(int(steps)),
        "--export-every", str(int(steps)),
        "--eval-every", str(int(steps) + 1),
        "--export-path", str(export_dir),
        "--export-name", export_name,
        "--max-splats", str(int(max_splats)),
        "--seed", str(int(seed)),
    ]


def _fail_colmap(stage: str, res: ProcResult) -> AddonError:
    tail = res.tail_text(6)
    if re.search(r"CUDA|cuda|SiftGPU|OpenGL", tail):
        hint = "COLMAP's GPU code failed. Update your NVIDIA graphics driver; details are in the capture log."
    else:
        hint = "See the capture log for details. Make sure the photos overlap well and are sharp."
    return AddonError(f"COLMAP {stage} failed (exit code {res.returncode}).", hint=hint)


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
) -> Path:
    """Reconstruct a Gaussian splat from photos and/or videos.

    Returns the ``.splat`` path; a ``.ply`` with the same stem is written next to it.
    Long-running (minutes to an hour); honours ``cancel`` (kills COLMAP/Brush) and
    reports ``progress``.  Raises :class:`InputError` for unusable inputs and
    :class:`AddonError` (with hints) when reconstruction fails.
    """
    progress = progress or ProgressReporter.null()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos, videos = _expand_inputs(inputs)
    if not photos and not videos:
        raise InputError("No photos or videos were given for the capture.")
    stem = re.sub(r"[^\w.-]+", "_", name or (videos[0].stem if videos else photos[0].parent.name or "capture"))
    stem = stem.strip("._") or "capture"
    colmap = find_colmap()
    brush = find_brush()
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
        up_raw = camera_up_vector(undist / "sparse" / "images.bin") if (undist / "sparse" / "images.bin").is_file() \
            else camera_up_vector(model / "images.bin")
        progress(0.60, "Training the splat (Brush)")
        if cancel is not None:
            cancel.check()

        # ---------------------------------------------------------------- 3. Brush
        export_dir = work / "brush"
        p_brush = progress.sub(0.60, 0.95)
        train_cap = int(min(10_000_000, max(1_000_000, 2 * int(max_splats))))

        def brush_line(line: str) -> None:
            m = _BRUSH_ITER_RE.search(line)
            if m:
                p_brush(int(m.group(1)) / max(int(steps), 1), f"Training the splat (step {m.group(1)}/{steps})")

        res = runner([*brush, *brush_args(undist, export_dir, steps=steps, max_splats=train_cap)],
                     "Brush training", brush_line, cwd=work)
        ply = export_dir / "splat.ply"
        if not ply.is_file():
            cands = sorted(export_dir.glob("*.ply"), key=lambda p: p.stat().st_mtime) if export_dir.is_dir() else []
            ply = cands[-1] if cands else ply
        if not res.ok or not ply.is_file():
            tail = res.tail_text(5)
            hint = ("Your GPU may have run out of memory: lower the training steps / max splats or close other apps."
                    if re.search(r"memory|OutOfMemory|allocation", tail, re.I)
                    else "See the capture log for details; updating the GPU driver often helps.")
            raise AddonError(f"Brush training failed (exit code {res.returncode}).", hint=hint)

        # ---------------------------------------------------------------- 4. post-process
        progress(0.95, "Cleaning up the splat")
        api = _splat_api()
        sc = api.read_splats(ply)
        if len(sc) == 0:
            raise AddonError("The trained splat is empty.", hint="Try more images or more training steps.")
        up = None
        if up_raw is not None:
            try:
                up = _map_vector_to_cloud(ply_xyz(ply), np.asarray(sc.means), up_raw)
            except (AddonError, ValueError, KeyError, OSError) as e:
                log.debug("up-vector mapping failed (%s); using the raw vector", e)
                up = up_raw
        sc = api.crop_splats(sc, radius_pct=float(crop_pct))
        sc = api.normalize_orientation(sc, up=up)
        sc = api.decimate_splats(sc, max_n=int(max_splats))
        out_splat = Path(api.write_splats(sc, out_dir / f"{stem}.splat"))
        api.write_splats(sc, out_dir / f"{stem}.ply")
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
    allowed = {"video_fps", "max_images", "steps", "max_splats", "crop_pct", "name", "use_gpu", "keep_workdir"}
    unknown = set(params) - allowed
    if unknown:
        raise InputError(f"Unknown capture option(s): {', '.join(sorted(unknown))}")
    out = capture_to_splat([Path(p) for p in paths], Path(out_dir), progress=progress, cancel=cancel,
                           timeout=timeout, **params)
    return {"splat": str(out), "ply": str(out.with_suffix(".ply"))}


__all__ = [
    "ColmapInfo",
    "brush_args",
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
    "ply_xyz",
    "read_images_bin",
    "run_task",
    "select_sharp",
    "sharpness",
]
