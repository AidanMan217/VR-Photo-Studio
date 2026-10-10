"""Video encoding through an ffmpeg subprocess (raw RGB frames piped to stdin).

* :func:`ffmpeg_exe` - the ffmpeg binary shipped with ``imageio-ffmpeg`` (or on PATH).
* :func:`probe_encoders` - which H.264/HEVC encoders actually work on this machine
  (a one-frame test encode per candidate, cached for the process).
* :data:`PRESETS` - output profiles for Meta Quest 3 / Quest 2 / generic players.
* :class:`VideoWriter` - context manager that encodes frames as they are produced and
  optionally muxes the audio track of a source video.

All outputs are MP4/MOV with ``yuv420p`` 8-bit video, even dimensions, BT.709 colour tags
and ``+faststart`` (moov atom first, so the Quest can stream them over HTTP).  The file
is written to a temporary name next to the destination and renamed on success, so a
failed or cancelled encode never leaves a truncated video behind.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from ..core.errors import EncodeError, InputError
from ..core.fsutil import replace_with_retry
from ..core.log import get_logger

log = get_logger(__name__)

#: Encoders probed by :func:`probe_encoders` (contract order).
ENCODER_CANDIDATES: tuple[str, ...] = (
    "libx264",
    "libx265",
    "hevc_nvenc",
    "h264_nvenc",
    "hevc_amf",
    "hevc_qsv",
)

#: Short names accepted for ``VideoWriter(encoder=...)`` -> family, resolved per codec.
_ALIASES: dict[str, dict[str, str]] = {
    "nvenc": {"hevc": "hevc_nvenc", "h264": "h264_nvenc"},
    "nvidia": {"hevc": "hevc_nvenc", "h264": "h264_nvenc"},
    "amf": {"hevc": "hevc_amf", "h264": "h264_amf"},
    "amd": {"hevc": "hevc_amf", "h264": "h264_amf"},
    "qsv": {"hevc": "hevc_qsv", "h264": "h264_qsv"},
    "intel": {"hevc": "hevc_qsv", "h264": "h264_qsv"},
    # the preset decides the codec; "x264"/"x265" just mean "the software encoder"
    "x265": {"hevc": "libx265", "h264": "libx264"},
    "x264": {"hevc": "libx265", "h264": "libx264"},
    "software": {"hevc": "libx265", "h264": "libx264"},
    "cpu": {"hevc": "libx265", "h264": "libx264"},
}

#: Constant-quality rate control everywhere (a fixed bitrate target would make a short or
#: small clip as many Mbit/s as an 8K VR180 film).  NVENC: quality-targeted VBR with a
#: peak cap (the Quest's decoder limit); AMF / QSV: constant QP (supported by every
#: driver generation, unlike AMF QVBR / QSV ICQ).
_HEVC_HW_ARGS: dict[str, list[str]] = {
    "hevc_nvenc": "-preset p5 -rc vbr -cq 19 -b:v 0 -maxrate 100M -bufsize 200M -profile:v main".split(),
    "hevc_amf": ["-quality", "quality", "-rc", "cqp", "-qp_i", "20", "-qp_p", "22"],
    "hevc_qsv": ["-preset", "medium", "-q:v", "22"],
    "libx265": ["-preset", "fast", "-crf", "18", "-x265-params", "log-level=error"],
}

#: Output profiles.  ``encoders`` is the preference order for ``encoder="auto"``; ``args``
#: holds per-encoder rate-control options; ``max_width``/``max_height`` cap the output
#: size (larger inputs are scaled down, keeping the aspect ratio).
PRESETS: dict[str, dict[str, Any]] = {
    "quest3_hevc": {
        "title": "Meta Quest 3 (HEVC, high bitrate)",
        "codec": "hevc",
        "encoders": ["hevc_nvenc", "hevc_amf", "hevc_qsv", "libx265"],
        "args": dict(_HEVC_HW_ARGS),
        "max_width": 8192,
        "max_height": 8192,
        "tag": "hvc1",
    },
    "quest2_hevc": {
        "title": "Meta Quest 2 (HEVC, max 5760 px wide)",
        "codec": "hevc",
        "encoders": ["hevc_nvenc", "hevc_amf", "hevc_qsv", "libx265"],
        "args": dict(_HEVC_HW_ARGS),
        "max_width": 5760,
        "max_height": 5760,
        "tag": "hvc1",
    },
    "h264_compat": {
        "title": "H.264 (plays everywhere, max 4096 px)",
        "codec": "h264",
        "encoders": ["h264_nvenc", "libx264"],
        "args": {
            "h264_nvenc": "-preset p5 -rc vbr -cq 18 -b:v 0 -maxrate 60M -bufsize 120M -profile:v high".split(),
            "libx264": ["-preset", "medium", "-crf", "18", "-profile:v", "high"],
            "h264_amf": ["-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20", "-qp_b", "22"],
            "h264_qsv": ["-preset", "medium", "-q:v", "20"],
        },
        "max_width": 4096,
        "max_height": 4096,
        "tag": "avc1",
    },
    "preview": {
        "title": "Quick preview (small, fast)",
        "codec": "h264",
        "encoders": ["libx264", "h264_nvenc"],
        "args": {
            "libx264": ["-preset", "veryfast", "-crf", "26"],
            "h264_nvenc": ["-preset", "p2", "-rc", "vbr", "-cq", "28", "-b:v", "0"],
        },
        "max_width": 1280,
        "max_height": 1280,
        "tag": "avc1",
    },
}

#: Preset used for each ``Settings.quest_model`` when ``preset="auto"``.
QUEST_PRESETS: dict[str, str] = {"quest3": "quest3_hevc", "quest2": "quest2_hevc", "pcvr": "h264_compat"}
#: Preset for ``preset="auto"`` by output kind, overriding the headset preset: a flat
#: video is for phones / PCs, which may lack an HEVC decoder (stock Windows needs a paid
#: extension), so it is H.264.
OUTPUT_PRESETS: dict[str, str] = {"flat": "h264_compat"}

_CONTAINERS = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov"}
_LOG_TAIL_CHARS = 2000
_PROBE_SIZE = 256  # px, square test frame (hardware encoders reject tiny frames)
_PROBE_TIMEOUT_S = 30.0

_probe_lock = threading.Lock()
_probe_cache: dict[str, bool] | None = None
_size_cache: dict[tuple[str, int, int], bool] = {}
_SOFTWARE_ENCODERS = frozenset({"libx264", "libx265"})
_exe_cache: str | None = None


# --------------------------------------------------------------------------- process helpers
def popen_flags() -> dict[str, Any]:
    """Keyword arguments that keep ffmpeg from opening a console window on Windows."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def ffmpeg_exe() -> str:
    """Path of the ffmpeg executable (``imageio_ffmpeg.get_ffmpeg_exe()``, else PATH).

    Honours ``IMAGEIO_FFMPEG_EXE``.  Raises :class:`EncodeError` if none is found.
    """
    global _exe_cache
    if _exe_cache is not None and (os.path.isfile(_exe_cache) or shutil.which(_exe_cache)):
        return _exe_cache
    exe: str | None = None
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # RuntimeError when nothing found, ImportError when not installed
        log.debug("imageio-ffmpeg could not provide ffmpeg: %s", exc)
    if not exe:
        exe = shutil.which("ffmpeg")
    if not exe:
        raise EncodeError(
            "ffmpeg was not found.",
            hint="Reinstall VR Photo Studio, or install ffmpeg and add it to PATH.",
        )
    _exe_cache = exe
    return exe


def _tail(path: Path | None, n: int = _LOG_TAIL_CHARS) -> str:
    if path is None:
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - n))
            return fh.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # Windows: file still locked by a dying process
        log.debug("Could not delete %s: %s", path, exc)


# --------------------------------------------------------------------------- encoder probing
def _listed_encoders(exe: str) -> set[str] | None:
    """Names from ``ffmpeg -encoders`` (None if the call fails)."""
    try:
        res = subprocess.run(
            [exe, "-hide_banner", "-nostdin", "-encoders"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S,
            **popen_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("ffmpeg -encoders failed: %s", exc)
        return None
    names: set[str] = set()
    for line in res.stdout.decode("utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return names


def _pix_fmt_for(encoder: str) -> str:
    return "nv12" if encoder.endswith("_qsv") else "yuv420p"


def _test_encode(exe: str, encoder: str) -> bool:
    """Encode one black frame with ``encoder`` to the null muxer."""
    frame = bytes(_PROBE_SIZE * _PROBE_SIZE * 3)
    cmd = [
        exe, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{_PROBE_SIZE}x{_PROBE_SIZE}", "-r", "30", "-i", "-",
        "-frames:v", "1", "-c:v", encoder, "-pix_fmt", _pix_fmt_for(encoder), "-f", "null", "-",
    ]  # fmt: skip
    try:
        res = subprocess.run(cmd, input=frame, capture_output=True, timeout=_PROBE_TIMEOUT_S, **popen_flags())
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("Encoder probe %s failed to run: %s", encoder, exc)
        return False
    ok = res.returncode == 0
    if not ok:
        msg = res.stderr.decode("utf-8", errors="replace").strip().splitlines()
        log.debug("Encoder %s unavailable: %s", encoder, msg[-1] if msg else f"exit {res.returncode}")
    return ok


def encoder_works_at(encoder: str, width: int, height: int) -> bool:
    """Whether ``encoder`` accepts ``width x height`` frames (one-frame test, cached).

    Hardware encoders pass the small :func:`probe_encoders` test but may reject large
    frames (e.g. 7680-wide VR180 on AMF or older NVENC generations).
    """
    key = (encoder, int(width), int(height))
    with _probe_lock:
        if key in _size_cache:
            return _size_cache[key]
    try:
        exe = ffmpeg_exe()
    except EncodeError:
        return False
    cmd = [
        exe, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={key[1]}x{key[2]}:r=30:d=0.1",
        "-frames:v", "1", "-c:v", encoder, "-pix_fmt", _pix_fmt_for(encoder), "-f", "null", "-",
    ]  # fmt: skip
    try:
        res = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=_PROBE_TIMEOUT_S, **popen_flags()
        )
        ok = res.returncode == 0
        if not ok:
            lines = res.stderr.decode("utf-8", errors="replace").strip().splitlines()
            log.info(
                "%s cannot encode %dx%d: %s", encoder, key[1], key[2], lines[-1] if lines else res.returncode
            )
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("%s size test failed to run: %s", encoder, exc)
        ok = False
    with _probe_lock:
        _size_cache[key] = ok
    return ok


def probe_encoders(refresh: bool = False) -> dict[str, bool]:
    """Which of :data:`ENCODER_CANDIDATES` work here (test-encodes one tiny frame each).

    The result is cached for the lifetime of the process; ``refresh=True`` re-probes
    (e.g. after a driver update).  Never raises: a missing ffmpeg yields all ``False``.
    """
    global _probe_cache
    with _probe_lock:
        if _probe_cache is not None and not refresh:
            return dict(_probe_cache)
        result = dict.fromkeys(ENCODER_CANDIDATES, False)
        try:
            exe = ffmpeg_exe()
        except EncodeError as exc:
            log.warning("Cannot probe video encoders: %s", exc)
            _probe_cache = result
            return dict(result)
        listed = _listed_encoders(exe)
        todo = [e for e in ENCODER_CANDIDATES if listed is None or e in listed]
        if todo:
            with ThreadPoolExecutor(max_workers=min(4, len(todo))) as pool:
                for enc, ok in zip(todo, pool.map(lambda e: _test_encode(exe, e), todo)):
                    result[enc] = bool(ok)
        log.info("Video encoders: %s", ", ".join(k for k, v in result.items() if v) or "none")
        _probe_cache = result
        return dict(result)


# --------------------------------------------------------------------------- helpers
def resolve_preset(preset: str, quest_model: str | None = None, *, output: str | None = None) -> str:
    """Map ``"auto"`` to a preset name.

    ``output`` is the kind of video being made (``"flat"``, ``"sbs"``, ``"vr180"`` ...):
    a flat video plays on phones / PCs and becomes ``"h264_compat"``; anything else is
    for the headset and follows ``quest_model`` (default from settings).
    """
    name = str(preset or "auto").strip().lower()
    if name == "auto":
        kind = str(output or "").strip().lower()
        if kind in OUTPUT_PRESETS:
            return OUTPUT_PRESETS[kind]
        if quest_model is None:
            try:
                from ..core.config import get_settings

                quest_model = get_settings().quest_model
            except Exception:  # pragma: no cover - settings unreadable
                quest_model = "quest3"
        return QUEST_PRESETS.get(str(quest_model).lower(), "h264_compat")
    if name not in PRESETS:
        raise InputError(f"Unknown video preset {preset!r}. Choose one of: auto, {', '.join(PRESETS)}.")
    return name


def choose_encoder(preset: str, encoder: str = "auto", available: dict[str, bool] | None = None) -> str:
    """Pick the ffmpeg encoder name for ``preset``.

    ``encoder`` may be ``"auto"`` (first working encoder of the preset, by
    :func:`probe_encoders`), a family alias (``"nvenc"``, ``"x265"``, ``"amf"``, ... -
    falls back to ``"auto"`` with a warning when the probe says it does not work here) or
    an explicit ffmpeg encoder name, which is used as-is (ffmpeg reports it if unknown).
    """
    spec = PRESETS[resolve_preset(preset)]
    enc = str(encoder or "auto").strip()
    low = enc.lower()
    if low in _ALIASES:
        name = _ALIASES[low][spec["codec"]]
        if name not in ENCODER_CANDIDATES:
            return name  # not probed (e.g. h264_amf): let ffmpeg decide
        avail = probe_encoders() if available is None else available
        if avail.get(name):
            return name
        log.warning("Requested encoder %s (%s) is not available here; choosing automatically", name, enc)
        low = "auto"
    elif low != "auto":
        return enc
    avail = probe_encoders() if available is None else available
    for cand in spec["encoders"]:
        if avail.get(cand):
            return cand
    # Nothing probed OK (e.g. probing failed): fall back to the software encoder.
    fallback = spec["encoders"][-1]
    log.warning("No probed encoder for preset %s; trying %s", preset, fallback)
    return fallback


def output_size(width: int, height: int, preset: str) -> tuple[int, int]:
    """Encoded frame size for input ``width x height``: capped by the preset, even."""
    spec = PRESETS[resolve_preset(preset)]
    w, h = int(width), int(height)
    s = min(1.0, spec["max_width"] / w, spec["max_height"] / h)
    if s < 1.0:
        w2 = max(2, int(w * s) // 2 * 2)
        h2 = max(2, int(round(h * w2 / w)) // 2 * 2)
        return w2, h2
    return max(2, w // 2 * 2), max(2, h // 2 * 2)


_QUALITY_OPTS = ("-crf", "-cq", "-qp_i", "-qp_p", "-qp_b", "-q:v", "-b:v", "-maxrate", "-bufsize")


def _rate_args(spec: dict[str, Any], encoder: str, bitrate: str | None) -> list[str]:
    args = list(spec["args"].get(encoder, []))
    if bitrate:
        # replace quality-based rate control by the requested average bitrate
        out: list[str] = []
        skip = False
        for i, a in enumerate(args):
            if skip:
                skip = False
                continue
            if a in _QUALITY_OPTS and i + 1 < len(args):
                skip = True
                continue
            if a == "-rc" and i + 1 < len(args) and args[i + 1] == "cqp":
                out += ["-rc", "vbr_peak"]  # AMF: constant QP ignores bitrates
                skip = True
                continue
            out.append(a)
        out += ["-b:v", str(bitrate)]
        if encoder in ("libx264", "libx265"):
            out += ["-maxrate", str(bitrate), "-bufsize", _double_rate(str(bitrate))]
        return out
    return args


def _double_rate(rate: str) -> str:
    """'40M' -> '80M' (best effort; returns the input when it cannot be parsed)."""
    r = rate.strip()
    mult = r[-1:].upper() if r[-1:].isalpha() else ""
    try:
        val = float(r[:-1] if mult else r)
    except ValueError:
        return rate
    return f"{val * 2:g}{mult}"


_TAGS = {"hevc": "hvc1", "h264": "avc1"}


def _codec_of(encoder: str) -> str | None:
    """Codec family of an ffmpeg encoder name ("hevc", "h264" or None if unknown)."""
    e = encoder.lower()
    if "265" in e or "hevc" in e:
        return "hevc"
    if "264" in e:
        return "h264"
    return None


_NTSC_BASES = (24, 30, 48, 60, 120)


def ntsc_rate(fps: float) -> Fraction | None:
    """Exact ``N*1000/1001`` rate for an NTSC frame rate, else None.

    ffmpeg prints rates rounded to two decimals (24000/1001 shows as ``23.98``); using
    the rounded value for decoding while the encoder writes 24000/1001 would make the
    picture drift behind the sound (about 0.2 ms per second).
    """
    for base in _NTSC_BASES:
        if abs(float(fps) - base * 1000 / 1001) < 0.005:
            return Fraction(base * 1000, 1001)
    return None


def fps_arg(fps: float) -> str:
    """ffmpeg rate string; NTSC rates (29.97 ...) become exact ``N*1000/1001`` fractions.

    Use the same string for decoding (``fps`` filter) and encoding so both agree exactly.
    """
    fr = ntsc_rate(fps) or Fraction(float(fps)).limit_denominator(1001)
    if fr <= 0 < float(fps):  # very slow rates (< 1/2002 fps) would round to "0/1"
        fr = Fraction(float(fps)).limit_denominator(10**9)
    return f"{fr.numerator}/{fr.denominator}"


# --------------------------------------------------------------------------- writer
def ff_path(path: str | os.PathLike[str]) -> str:
    """Absolute path string for an ffmpeg argument.

    ffmpeg treats ``name:rest`` as a protocol URL, so a relative file name containing a
    colon (``"12:30 clip.mp4"``) would fail to open.  Absolute paths start with ``/``,
    a drive letter or ``\\\\`` and are always read as plain files.
    """
    return os.path.abspath(os.fspath(path))


def _source_has_audio(path: Path) -> bool:
    """True if ``path`` has at least one audio stream (by the ``ffmpeg -i`` banner)."""
    from .decode import parse_ffmpeg_info  # local: decode imports this module

    try:
        res = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", ff_path(path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S * 2,
            **popen_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Could not inspect the audio of %s: %s", path.name, exc)
        return False
    return bool(parse_ffmpeg_info(res.stderr.decode("utf-8", errors="replace"))["has_audio"])


def as_rgb24(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Coerce a frame to contiguous ``height x width x 3`` uint8 RGB.

    Accepts RGB, RGBA (alpha dropped) and grayscale (``HxW`` / ``HxWx1``) frames.  Float
    frames follow the project convention (:func:`vrphoto.core.imaging.to_uint8`): values
    in [0, 1], NaN -> 0; uint16 is scaled from [0, 65535]; other integer types are
    clipped to [0, 255].
    """
    a = np.asarray(frame)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4) or a.shape[:2] != (height, width):
        got = "x".join(str(s) for s in a.shape)
        raise InputError(
            f"Frame of shape {got} does not match the video size {width}x{height} "
            "(frames must be HxWx3 RGB, HxWx4 RGBA or HxW grayscale)."
        )
    if a.shape[2] == 4:
        a = a[..., :3]
    if a.dtype == np.uint8:
        pass
    elif a.dtype == np.bool_:
        a = a.astype(np.uint8) * np.uint8(255)
    elif np.issubdtype(a.dtype, np.floating):
        f = np.nan_to_num(a.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        a = (np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    elif a.dtype == np.uint16:  # 16-bit full range -> 8-bit
        a = ((a.astype(np.uint32) + 128) // 257).astype(np.uint8)
    elif np.issubdtype(a.dtype, np.integer):
        a = np.clip(a, 0, 255).astype(np.uint8)
    else:
        raise InputError(f"Unsupported frame data type {a.dtype}; use uint8 RGB.")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return np.ascontiguousarray(a)


class VideoWriter:
    """Encode RGB frames to an MP4/MOV with ffmpeg.

    Example::

        with VideoWriter(out, w, h, 30, preset="quest3_hevc") as vw:
            for frame in frames:
                vw.write(frame)
        # leaving the block calls close(); an exception aborts and deletes the output

    Args:
        path: output ``.mp4`` / ``.mov`` (created or replaced on success).
        width, height: size of the frames passed to :meth:`write`.  The encoded size is
            even and capped by the preset (see :func:`output_size`).
        fps: frame rate.
        preset: key of :data:`PRESETS` or ``"auto"`` (by ``Settings.quest_model``).
        audio_from: optional video whose first audio stream is muxed in (AAC).  The audio
            is cut - or padded with silence - to exactly the length of the encoded video,
            so the video never loses frames; a source without audio is fine.
        encoder: ``"auto"``, an alias such as ``"nvenc"``/``"x265"``, or an ffmpeg name.
        bitrate: e.g. ``"40M"`` - overrides the preset's quality settings.
        audio_start_s: seconds to skip in ``audio_from`` (when the video was cut).

    With audio the video is encoded first and the audio muxed in by a second, stream-copy
    ffmpeg pass in :meth:`close` (``-shortest`` on a live pipe would stop reading frames
    as soon as a shorter audio track ends).  If that pass fails the video is kept without
    sound and the reason is appended to :attr:`warnings`.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        width: int,
        height: int,
        fps: float,
        preset: str = "h264_compat",
        *,
        audio_from: str | os.PathLike[str] | None = None,
        encoder: str = "auto",
        bitrate: str | None = None,
        audio_start_s: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        if self.width < 2 or self.height < 2:
            raise InputError(f"Video frames must be at least 2x2 pixels (got {width}x{height}).")
        self.fps = float(fps)
        if not (self.fps > 0 and np.isfinite(self.fps)):
            raise InputError(f"Video frame rate must be positive (got {fps}).")
        self.preset = resolve_preset(preset)
        spec = PRESETS[self.preset]
        container = _CONTAINERS.get(self.path.suffix.lower())
        if container is None:
            raise InputError(f"Unsupported video file type {self.path.suffix!r}; use .mp4 or .mov.")
        audio_start = float(audio_start_s or 0.0)
        if not (np.isfinite(audio_start) and audio_start >= 0.0):
            raise InputError(f"Audio start time must be zero or positive (got {audio_start_s}).")
        self.audio_from = Path(audio_from) if audio_from else None
        if self.audio_from is not None and not self.audio_from.is_file():
            raise InputError(f"Audio source not found: {self.audio_from}")
        self.encoder = choose_encoder(self.preset, encoder)
        self.out_width, self.out_height = output_size(self.width, self.height, self.preset)
        req = str(encoder or "auto").strip().lower()
        if self.encoder not in _SOFTWARE_ENCODERS and (req == "auto" or req in _ALIASES):
            self.encoder = self._fit_encoder(self.encoder)
        if (self.out_width, self.out_height) != (self.width, self.height):
            log.info(
                "Encoding %dx%d frames as %dx%d (preset %s)",
                self.width, self.height, self.out_width, self.out_height, self.preset,
            )  # fmt: skip
        self.frames_written = 0
        #: Non-fatal problems (e.g. audio that could not be muxed); for the caller to show.
        self.warnings: list[str] = []
        self._spec = spec
        self._container = container
        self._audio_start = audio_start
        self._rate = fps_arg(self.fps)
        self._closed = False
        self._result: Path | None = None
        self._mux_audio = self.audio_from is not None and _source_has_audio(self.audio_from)
        if self.audio_from is not None and not self._mux_audio:
            log.info("%s has no audio stream; writing a silent video", self.audio_from.name)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:8]
        #: final file, renamed onto ``path`` by close()
        self._tmp = self.path.with_name(f".{self.path.stem}.{token}.encoding{self.path.suffix}")
        #: output of the frame pipe (== _tmp unless audio is muxed in a second pass)
        self._tmp_video = (
            self.path.with_name(f".{self.path.stem}.{token}.video{self.path.suffix}")
            if self._mux_audio
            else self._tmp
        )
        cmd = self._build_cmd(bitrate)
        self.command = cmd
        log.debug("ffmpeg: %s", " ".join(cmd))
        fd, log_name = tempfile.mkstemp(prefix="vrphoto-ffmpeg-", suffix=".log")
        os.close(fd)
        self._log_path: Path | None = Path(log_name)
        self._log_fh = open(self._log_path, "wb")  # noqa: SIM115 - closed in _finish
        try:
            self._proc: subprocess.Popen[bytes] | None = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._log_fh,
                **popen_flags(),
            )
        except OSError as exc:
            self._log_fh.close()
            _unlink(self._log_path)
            raise EncodeError(f"Could not start ffmpeg: {exc}") from exc

    def _fit_encoder(self, chosen: str) -> str:
        """``chosen`` if it handles the output size, else the next working encoder of the preset."""
        ow, oh = self.out_width, self.out_height
        if encoder_works_at(chosen, ow, oh):
            return chosen
        avail = probe_encoders()
        for cand in self._spec_encoders():
            if cand == chosen:
                continue
            if cand in _SOFTWARE_ENCODERS or (avail.get(cand) and encoder_works_at(cand, ow, oh)):
                log.warning("%s cannot encode %dx%d video here; using %s instead", chosen, ow, oh, cand)
                return cand
        return chosen  # nothing better: let ffmpeg report the problem

    def _spec_encoders(self) -> list[str]:
        return list(PRESETS[self.preset]["encoders"])

    # ---------------------------------------------------------------- command line
    def _filters(self) -> str:
        chain: list[str] = []
        ow, oh = self.out_width, self.out_height
        if (ow, oh) != (self.width, self.height):
            if (ow, oh) == (self.width // 2 * 2, self.height // 2 * 2):
                chain.append(f"crop={ow}:{oh}:0:0")  # odd size: drop the last column / row
            else:
                chain.append(f"scale={ow}:{oh}:flags=lanczos")
        chain.append("scale=out_color_matrix=bt709:out_range=tv")
        chain.append(f"format={_pix_fmt_for(self.encoder)}")
        return ",".join(chain)

    def _tag_args(self) -> list[str]:
        tag = _TAGS.get(_codec_of(self.encoder) or self._spec["codec"])
        return ["-tag:v", tag] if tag else []

    def _build_cmd(self, bitrate: str | None) -> list[str]:
        cmd = [
            ffmpeg_exe(), "-hide_banner", "-nostats", "-loglevel", "warning", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}",
            "-r", self._rate, "-i", "-",
            "-map", "0:v:0", "-vf", self._filters(), "-c:v", self.encoder,
        ]  # fmt: skip
        cmd += _rate_args(self._spec, self.encoder, bitrate)
        gop = max(1, int(round(self.fps * 2)))
        cmd += [
            "-g", str(gop),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        ]  # fmt: skip
        cmd += self._tag_args()
        if not self._mux_audio:  # the audio pass writes the final (faststart) file otherwise
            cmd += ["-movflags", "+faststart"]
        cmd += ["-f", self._container, ff_path(self._tmp_video)]
        return cmd

    def _mux_cmd(self, with_audio: bool) -> list[str]:
        """Second pass: stream-copy the video, add AAC audio of exactly the video length."""
        cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "warning", "-y",
               "-i", ff_path(self._tmp_video)]  # fmt: skip
        if with_audio:
            assert self.audio_from is not None
            if self._audio_start > 0:
                cmd += ["-ss", f"{self._audio_start:.6f}"]
            dur = Fraction(self.frames_written) / Fraction(self._rate)
            cmd += ["-i", ff_path(self.audio_from), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-af", "apad", "-t", f"{float(dur):.6f}"]  # fmt: skip
        else:
            cmd += ["-map", "0:v:0", "-c:v", "copy"]
        cmd += self._tag_args()
        cmd += ["-movflags", "+faststart", "-f", self._container, ff_path(self._tmp)]
        return cmd

    def _run_mux(self, with_audio: bool) -> str | None:
        """Run a mux pass; returns None on success, else the tail of ffmpeg's output."""
        cmd = self._mux_cmd(with_audio)
        log.debug("ffmpeg (mux): %s", " ".join(cmd))
        try:
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, **popen_flags())
        except OSError as exc:
            return str(exc)
        if res.returncode == 0 and self._tmp.is_file() and self._tmp.stat().st_size > 0:
            return None
        tail = res.stderr.decode("utf-8", errors="replace").strip()[-_LOG_TAIL_CHARS:]
        return tail or f"ffmpeg exited with code {res.returncode}"

    def _mux(self) -> None:
        """Add the audio track (or, if that fails, just finish the silent video)."""
        assert self.audio_from is not None
        err = self._run_mux(with_audio=True)
        if err is not None:
            _unlink(self._tmp)
            msg = (
                f"The sound of {self.audio_from.name} could not be added; the video was saved without sound."
            )
            log.warning("%s ffmpeg said: %s", msg, err)
            self.warnings.append(msg)
            err2 = self._run_mux(with_audio=False)
            if err2 is not None:
                _unlink(self._tmp)
                _unlink(self._tmp_video)
                raise EncodeError(
                    f"Video encoding failed (could not finish {self.path.name}).\nffmpeg said:\n{err2}"
                )
        _unlink(self._tmp_video)

    # ---------------------------------------------------------------- frames
    def write(self, rgb: np.ndarray) -> None:
        """Append one ``height x width`` frame (RGB uint8; RGBA/gray/float are converted)."""
        if self._closed or self._proc is None or self._proc.stdin is None:
            raise EncodeError("The video writer is already closed.")
        a = as_rgb24(rgb, self.width, self.height)
        try:
            self._proc.stdin.write(memoryview(a).cast("B"))
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._fail("ffmpeg stopped accepting frames", exc)
        self.frames_written += 1

    def close(self) -> Path:
        """Finish encoding and return the output path.  Raises :class:`EncodeError`."""
        if self._result is not None:
            return self._result
        if self._closed:
            raise EncodeError("The video writer was aborted; no video was written.")
        proc = self._proc
        assert proc is not None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            self.abort()
            raise
        if rc != 0:
            self._fail(f"ffmpeg exited with code {rc}")
        if self.frames_written == 0:
            self._fail("no frames were written")
        if not self._tmp_video.is_file() or self._tmp_video.stat().st_size == 0:
            self._fail("ffmpeg produced no output file")
        if self._mux_audio:
            try:
                self._mux()
            except BaseException:
                self._finish(delete_log=True)
                _unlink(self._tmp)
                _unlink(self._tmp_video)
                raise
        try:
            replace_with_retry(self._tmp, self.path)
        except OSError as exc:
            self._fail(
                f"could not move the video into place ({exc}); the file is busy (another program "
                "or antivirus) - close any player showing it and try again"
            )
        self._finish(delete_log=True)
        self._result = self.path
        log.info("Wrote %s (%d frames, %s)", self.path, self.frames_written, self.encoder)
        return self.path

    def abort(self) -> None:
        """Stop ffmpeg and delete the partial output (safe to call more than once)."""
        if self._closed:
            return
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                log.warning("ffmpeg did not exit after kill")
        self._finish(delete_log=True)
        _unlink(self._tmp)
        _unlink(self._tmp_video)

    def _fail(self, what: str, exc: BaseException | None = None) -> None:
        proc = self._proc
        if proc is not None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        tail = _tail(self._log_path)
        log_path = self._log_path
        self._finish(delete_log=False)
        _unlink(self._tmp)
        _unlink(self._tmp_video)
        log.error("Video encode failed (%s, %s): %s", what, self.encoder, tail)
        msg = f"Video encoding failed ({what}, encoder {self.encoder})."
        if tail:
            msg += "\nffmpeg said:\n" + tail
        raise EncodeError(msg, hint=f"Full ffmpeg log: {log_path}") from exc

    def _finish(self, *, delete_log: bool) -> None:
        self._closed = True
        try:
            self._log_fh.close()
        except OSError:
            pass
        if delete_log:
            _unlink(self._log_path)
            self._log_path = None
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass

    # ---------------------------------------------------------------- context manager
    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def __del__(self) -> None:  # pragma: no cover - safety net
        try:
            if not getattr(self, "_closed", True):
                self.abort()
        except Exception:
            pass


def encode_frames(
    frames: Sequence[np.ndarray] | Any,
    path: str | os.PathLike[str],
    fps: float,
    preset: str = "h264_compat",
    **kwargs: Any,
) -> Path:
    """Convenience: encode an iterable of equally sized RGB frames; returns the path."""
    it = iter(frames)
    try:
        first = np.asarray(next(it))
    except StopIteration:
        raise InputError("No frames to encode.") from None
    h, w = first.shape[:2]
    with VideoWriter(path, w, h, fps, preset, **kwargs) as vw:
        vw.write(first)
        for f in it:
            vw.write(f)
    return Path(path)


__all__ = [
    "PRESETS",
    "QUEST_PRESETS",
    "ENCODER_CANDIDATES",
    "ffmpeg_exe",
    "probe_encoders",
    "resolve_preset",
    "OUTPUT_PRESETS",
    "ntsc_rate",
    "fps_arg",
    "choose_encoder",
    "encoder_works_at",
    "output_size",
    "VideoWriter",
    "encode_frames",
    "popen_flags",
    "ff_path",
    "as_rgb24",
]
