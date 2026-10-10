"""Video probing and frame decoding with the bundled ffmpeg (no ffprobe needed).

* :func:`probe` parses the stream header printed by ``ffmpeg -i`` and counts video
  packets with a fast stream copy (no decoding).
* :func:`iter_frames` decodes to raw ``rgb24`` through a pipe, optionally downscaled
  (ffmpeg ``scale`` filter) and limited to a time range.

Display corrections applied while decoding (and reflected by :func:`probe`):

* rotation metadata (phone videos): ffmpeg auto-rotates, sizes are the displayed ones;
* non-square pixels (``SAR`` != 1): frames are resampled to square pixels;
* untagged HD video is decoded with the BT.709 matrix (what players assume), not BT.601;
* interlaced video (field order in the header) is deinterlaced (``yadif``, one frame per
  frame) so the depth model does not see combing;
* HDR video (PQ / HLG, e.g. iPhone recordings) is tone-mapped to SDR BT.709 when the
  ffmpeg build has the ``zscale`` and ``tonemap`` filters (else decoded as is, with a
  warning), so frames are not washed out.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.errors import UI_LOG, EncodeError, InputError
from ..core.log import get_logger
from ..core.types import CancelToken
from .encode import ff_path, ffmpeg_exe, fps_arg, ntsc_rate, popen_flags

log = get_logger(__name__)

_PROBE_TIMEOUT_S = 60.0
_COUNT_TIMEOUT_S = 600.0
_TAIL_CHARS = 2000

_RE_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RE_STREAM = re.compile(r"^\s*Stream #(\d+):(\d+)[^:]*:\s*(Video|Audio|Subtitle|Data|Attachment):\s*(.*)$")
_RE_SIZE = re.compile(r"(?<![\dx])(\d{1,5})x(\d{1,5})(?![\dx])")
_RE_FPS = re.compile(r"([\d.]+)(k?)\s+fps")
_RE_TBR = re.compile(r"([\d.]+)(k?)\s+tbr")
_RE_ROT = re.compile(r"rotation of\s+(-?[\d.]+)\s+degrees")
_RE_FRAME = re.compile(r"frame=\s*(\d+)")
_RE_SAR = re.compile(r"\[SAR\s+(\d+):(\d+)")
#: ffmpeg messages that mean "this file is broken / incomplete" (e.g. a phone video whose
#: copy was interrupted: the index at the end of the file is missing).
_DAMAGED = ("moov atom not found", "invalid data found", "partial file")
_RE_PIXFMT = re.compile(
    r"^(yuvj?\w*|nv\d\w*|p0\d\d\w*|p[24]\d\d\w*|gray\w*|ya\w*|rgb\w*|bgr\w*|gbr\w*|[0ax]?rgb\w*|"
    r"[0ax]?bgr\w*|pal8|mono[bw]|xyz\w*|y2\d\d\w*|vuy\w*|ayuv\w*)(?:\((.*)\))?$"
)
#: ffmpeg colourspace names (``av_color_space_name``) - a lone name in the banner means
#: matrix == primaries == transfer.
_COLOR_SPACES = {
    "gbr", "bt709", "unknown", "reserved", "fcc", "bt470bg", "smpte170m", "smpte240m", "ycgco",
    "bt2020nc", "bt2020c", "smpte2085", "chroma-derived-nc", "chroma-derived-c", "ictcp", "ipt-c2",
    "ycgco-re", "ycgco-ro",
}  # fmt: skip
HDR_TRANSFERS = ("smpte2084", "arib-std-b67")  # PQ, HLG
#: HDR -> SDR: linearise (100 nit reference), BT.2020 -> BT.709 primaries, Mobius
#: tone curve (linear mid-tones, soft highlight roll-off), BT.709 transfer, full-range RGB.
_TONEMAP = (
    "zscale=t=linear:npl=100{inp},format=gbrpf32le,zscale=p=bt709,"
    "tonemap=tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=pc,format=gbrp"
)
_filters_cache: dict[str, set[str]] = {}


@dataclass
class VideoInfo:
    """Basic properties of a video file (size as displayed, i.e. after rotation)."""

    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    has_audio: bool


def _check_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"Video file not found: {p}")
    return p


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, **popen_flags()
        )
    except subprocess.TimeoutExpired as exc:
        raise EncodeError(f"ffmpeg timed out after {timeout:.0f} s while reading the video.") from exc
    except OSError as exc:
        raise EncodeError(f"Could not run ffmpeg: {exc}") from exc


def _rate(m: re.Match[str] | None) -> float | None:
    if m is None:
        return None
    try:
        v = float(m.group(1)) * (1000.0 if m.group(2) else 1.0)
    except ValueError:
        return None
    return v if v > 0 and math.isfinite(v) else None


def parse_ffmpeg_info(text: str) -> dict:
    """Parse the ``ffmpeg -i`` banner into a dict (first video stream + audio flag).

    Keys: ``width``, ``height`` (displayed: square pixels, rotated), ``coded_width``,
    ``coded_height``, ``fps``, ``duration``, ``has_audio``, ``rotation`` (degrees),
    ``codec``, ``stream`` (index of the stream in the file), ``sar`` (``(num, den)``),
    ``pix_fmt``, ``color_range``, ``color_space``, ``color_primaries``, ``color_trc``
    (ffmpeg names, ``None`` when untagged), ``hdr`` (PQ / HLG transfer) and ``interlaced``
    (a field order is declared).  Missing values are ``None``.  NTSC rates, which the
    banner rounds (``23.98``), are returned exactly (``24000/1001``).
    """
    info: dict = {"width": None, "height": None, "fps": None, "duration": None,
                  "has_audio": False, "rotation": 0.0, "codec": None, "stream": None,
                  "coded_width": None, "coded_height": None, "sar": (1, 1), "pix_fmt": None,
                  "color_range": None, "color_space": None, "color_primaries": None,
                  "color_trc": None, "hdr": False, "interlaced": False}  # fmt: skip
    m = _RE_DURATION.search(text)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    in_video = False
    seen_video = False
    for line in text.splitlines():
        sm = _RE_STREAM.match(line)
        if sm:
            kind, rest = sm.group(3), sm.group(4)
            in_video = False
            if kind == "Audio":
                info["has_audio"] = True
            elif kind == "Video" and not seen_video and "attached pic" not in rest:
                seen_video = in_video = True
                info["stream"] = int(sm.group(2))
                info["codec"] = rest.split(",", 1)[0].split()[0] if rest.strip() else None
                _parse_pix_fmt(rest, info)
                # strip parenthesised groups like "yuv420p(tv, bt709, progressive)"
                flat = re.sub(r"\([^()]*\)", "", rest)
                sz = _RE_SIZE.search(flat)
                if sz:
                    info["width"], info["height"] = int(sz.group(1)), int(sz.group(2))
                sar = _RE_SAR.search(flat)
                if sar and int(sar.group(1)) > 0 and int(sar.group(2)) > 0:
                    info["sar"] = (int(sar.group(1)), int(sar.group(2)))
                fps = _rate(_RE_FPS.search(flat)) or _rate(_RE_TBR.search(flat))
                exact = ntsc_rate(fps) if fps else None
                info["fps"] = float(exact) if exact is not None else fps
            continue
        if in_video:
            rm = _RE_ROT.search(line)
            if rm:
                try:
                    info["rotation"] = float(rm.group(1))
                except ValueError:
                    pass
            elif line.strip().startswith(("Input #", "Output #", "Stream mapping")):
                in_video = False
    info["coded_width"], info["coded_height"] = info["width"], info["height"]
    num, den = info["sar"]
    if info["width"] and num != den:  # non-square pixels: display width = coded * SAR
        info["width"] = max(2, int(round(info["width"] * num / den)))
    rot = round(info["rotation"] / 90.0) % 2 if info["rotation"] else 0
    if rot and info["width"] and info["height"]:
        info["width"], info["height"] = info["height"], info["width"]
    return info


def _split_top(text: str) -> list[str]:
    """Split on commas that are not inside parentheses / brackets."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def _parse_pix_fmt(rest: str, info: dict) -> None:
    """Pixel format and colour tags from e.g. ``yuv420p10le(tv, bt2020nc/bt2020/arib-std-b67)``."""
    for part in _split_top(rest)[1:]:
        m = _RE_PIXFMT.match(part)
        if not m:
            continue
        info["pix_fmt"] = m.group(1)
        for tok in (t.strip() for t in (m.group(2) or "").split(",")):
            if tok in ("tv", "pc"):
                info["color_range"] = tok
            elif "/" in tok:
                sp, pr, trc = (tok.split("/") + ["", ""])[:3]
                info["color_space"], info["color_primaries"], info["color_trc"] = sp, pr, trc
            elif tok in _COLOR_SPACES:
                info["color_space"] = info["color_primaries"] = info["color_trc"] = tok
            elif tok.startswith(("top first", "bottom first", "top coded first", "bottom coded first")):
                info["interlaced"] = True
        for key in ("color_space", "color_primaries", "color_trc"):
            if info[key] in ("unknown", "reserved", ""):
                info[key] = None
        info["hdr"] = info["color_trc"] in HDR_TRANSFERS
        return


def available_filters(exe: str | None = None) -> set[str]:
    """Names of the filters of this ffmpeg build (cached per executable)."""
    exe = exe or ffmpeg_exe()
    if exe not in _filters_cache:
        names: set[str] = set()
        try:
            res = subprocess.run(
                [exe, "-hide_banner", "-nostdin", "-filters"],
                stdin=subprocess.DEVNULL, capture_output=True, timeout=_PROBE_TIMEOUT_S, **popen_flags(),
            )  # fmt: skip
            for line in res.stdout.decode("utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 3 and "->" in parts[2]:
                    names.add(parts[1])
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("ffmpeg -filters failed: %s", exc)
        _filters_cache[exe] = names
    return _filters_cache[exe]


def _count_packets(exe: str, path: Path, stream: str = "0:v:0") -> int | None:
    """Number of video packets (= frames) via stream copy to the null muxer."""
    cmd = [exe, "-hide_banner", "-nostdin", "-i", ff_path(path), "-map", stream, "-c", "copy", "-f", "null", "-"]  # fmt: skip
    res = _run(cmd, _COUNT_TIMEOUT_S)
    if res.returncode != 0:
        return None
    frames = _RE_FRAME.findall(res.stderr.decode("utf-8", errors="replace"))
    return int(frames[-1]) if frames else None


def _header(p: Path) -> tuple[str, dict]:
    """``(ffmpeg exe, parsed banner)`` for a video; raises if it has no video stream.

    The raw ffmpeg output goes to the log; the error names the likely cause (damaged /
    incompletely copied file, sound-only file) in plain words.
    """
    exe = ffmpeg_exe()
    res = _run([exe, "-hide_banner", "-nostdin", "-i", ff_path(p)], _PROBE_TIMEOUT_S)
    text = res.stderr.decode("utf-8", errors="replace")
    info = parse_ffmpeg_info(text)
    if not info["width"] or not info["height"]:
        raise _no_video_error(p, text, info)
    return exe, info


def _no_video_error(p: Path, text: str, info: dict) -> EncodeError | InputError:
    """User-facing error for a file in which ffmpeg found no picture."""
    log.info("ffmpeg found no video stream in %s:\n%s", p, text.strip()[-_TAIL_CHARS:])
    low = text.lower()
    if info.get("has_audio") and info.get("stream") is None:  # audio, and no video (cover art only)
        return InputError(
            f"{p.name} contains only sound, no picture.",
            hint="Choose a video file (with a picture) instead.",
        )
    if any(k in low for k in _DAMAGED):
        return EncodeError(
            f"{p.name} is damaged or was not copied completely.",
            hint="Copy the video from your phone or camera again and make sure the copy finished, "
            "then try again.",
        )
    return EncodeError(
        f"Could not read a video stream from {p.name}.",
        hint=f"This video format could not be read. Try converting it to MP4 (H.264); "
        f"details are in the log ({UI_LOG}).",
    )


def probe(path: str | os.PathLike[str]) -> VideoInfo:
    """Read size, frame rate, frame count, duration and audio presence of a video.

    The frame count comes from a stream copy of the video packets (fast, no decoding;
    reads the whole file once) and falls back to ``duration * fps``.

    Raises :class:`InputError` for a missing file and :class:`EncodeError` if ffmpeg
    cannot read it or it has no video stream.
    """
    p = _check_path(path)
    exe, info = _header(p)
    fps = float(info["fps"] or 30.0)
    duration = float(info["duration"] or 0.0)
    n = _count_packets(exe, p, _map_arg(info))
    if n is None or n <= 0:
        n = max(1, int(round(duration * fps))) if duration > 0 else 0
        log.debug("Frame count of %s estimated from duration: %d", p.name, n)
    if duration <= 0 and n > 0:
        duration = n / fps
    return VideoInfo(
        width=int(info["width"]),
        height=int(info["height"]),
        fps=fps,
        n_frames=int(n),
        duration=duration,
        has_audio=bool(info["has_audio"]),
    )


def _map_arg(info: dict) -> str:
    """``-map`` argument selecting the probed video stream (skips cover-art streams)."""
    return f"0:{info['stream']}" if info.get("stream") is not None else "0:v:0"


def _target_size(w: int, h: int, max_long: int | None) -> tuple[int, int]:
    if not max_long or max(w, h) <= max_long:
        return w, h
    s = max_long / float(max(w, h))
    return max(2, int(round(w * s / 2.0)) * 2), max(2, int(round(h * s / 2.0)) * 2)


def iter_frames(
    path: str | os.PathLike[str],
    *,
    max_long: int | None = None,
    start: float = 0.0,
    end: float | None = None,
    fps: float | None = None,
    cancel: CancelToken | None = None,
    hdr_to_sdr: bool = True,
) -> Iterator[np.ndarray]:
    """Yield the video's frames as ``HxWx3 uint8`` RGB arrays (each a new array).

    Args:
        path: video file.
        max_long: downscale (area filter) so the long side is at most this many pixels.
        start, end: time range in seconds (``end=None`` = to the end).
        fps: optional constant output frame rate (frames are dropped/duplicated by
            ffmpeg); default passes the decoded frames through unchanged
            (additive extension of the contract).
        cancel: checked before every frame (additive extension of the contract).
        hdr_to_sdr: tone-map PQ / HLG video to SDR BT.709 (when the ffmpeg build can).

    Arguments and the file header are validated immediately (``InputError`` /
    ``EncodeError``); ffmpeg starts on the first ``next()``.  Closing the iterator early
    (``break`` / ``close()``) stops ffmpeg.
    """
    p = _check_path(path)
    if max_long is not None and int(max_long) < 2:
        raise InputError(f"max_long must be at least 2 pixels (got {max_long}).")
    start = max(0.0, float(start or 0.0))
    if end is not None and float(end) <= start:
        raise InputError(f"Video end time ({end} s) must be after the start time ({start} s).")
    if fps is not None and not (float(fps) > 0 and math.isfinite(float(fps))):
        raise InputError(f"Frame rate must be positive (got {fps}).")
    exe, info = _header(p)
    w, h = int(info["width"]), int(info["height"])
    ow, oh = _target_size(w, h, max_long)
    cmd = [exe, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    if end is not None:
        cmd += ["-t", f"{float(end) - start:.6f}"]
    cmd += ["-i", ff_path(p), "-map", _map_arg(info), "-an", "-sn", "-dn"]
    cmd += ["-vf", ",".join(_decode_filters(exe, info, ow, oh, fps, hdr_to_sdr, p.name))]
    if not fps:
        cmd += ["-fps_mode", "passthrough"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return _frames(cmd, p.name, ow, oh, cancel)


def _decode_filters(
    exe: str, info: dict, ow: int, oh: int, fps: float | None, hdr_to_sdr: bool, name: str
) -> list[str]:
    """Filter chain: optional fps, scale to the displayed size, colour corrections."""
    filters: list[str] = []
    if info.get("interlaced"):
        filters.append("yadif=mode=send_frame:parity=auto:deint=interlaced")
    if fps:
        filters.append(f"fps={fps_arg(float(fps))}")  # same exact rate the encoder writes
    w, h = int(info["width"]), int(info["height"])
    # Always scale explicitly to the probed (displayed) size so the frame size we read
    # matches even for odd sizes / non-square pixels.
    scale = f"scale={ow}:{oh}:flags={'area' if (ow, oh) != (w, h) else 'bicubic'}"
    yuv = str(info.get("pix_fmt") or "yuv").startswith(("yuv", "nv", "p0", "p2", "p4", "y2", "vuy", "ayuv"))
    hdr = bool(info.get("hdr"))
    if yuv and not hdr and info.get("color_space") is None and max(w, h) >= 1280:
        scale += ":in_color_matrix=bt709"  # untagged HD: BT.709 like every player
    filters += [scale, "setsar=1"]
    if hdr and hdr_to_sdr:
        avail = available_filters(exe)
        if {"zscale", "tonemap"} <= avail:
            inp = ""
            if info.get("color_space") is None:
                inp += ":min=2020_ncl"
            if info.get("color_primaries") is None:
                inp += ":pin=2020"
            filters.append(_TONEMAP.format(inp=inp))
            log.info("%s is HDR (%s); tone-mapping to SDR", name, info.get("color_trc"))
        else:
            log.warning(
                "%s is HDR (%s) but this ffmpeg has no zscale/tonemap filter; colours will look flat",
                name, info.get("color_trc"),
            )  # fmt: skip
    return filters


def _frames(cmd: list[str], name: str, ow: int, oh: int, cancel: CancelToken | None) -> Iterator[np.ndarray]:
    """Run the decode command and yield ``oh x ow`` RGB frames from its stdout."""
    fd, log_name = tempfile.mkstemp(prefix="vrphoto-ffdec-", suffix=".log")
    os.close(fd)
    log_path = Path(log_name)
    log_fh = open(log_path, "wb")  # noqa: SIM115 - closed in finally
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log_fh,
            bufsize=0, **popen_flags(),
        )  # fmt: skip
    except OSError as exc:
        log_fh.close()
        log_path.unlink(missing_ok=True)
        raise EncodeError(f"Could not start ffmpeg: {exc}") from exc

    frame_bytes = ow * oh * 3
    finished = False
    n = 0
    try:
        assert proc.stdout is not None
        while True:
            if cancel is not None:
                cancel.check()
            buf = np.empty((oh, ow, 3), dtype=np.uint8)
            mv = memoryview(buf).cast("B")
            got = 0
            while got < frame_bytes:
                k = proc.stdout.readinto(mv[got:])
                if not k:
                    break
                got += k
            if got < frame_bytes:
                if got:
                    log.debug("Discarding a truncated final frame (%d of %d bytes)", got, frame_bytes)
                break
            n += 1
            yield buf
        rc = proc.wait()
        finished = True
        if rc != 0:
            log_fh.flush()
            tail = log_path.read_bytes()[-_TAIL_CHARS:].decode("utf-8", errors="replace").strip()
            last = tail.splitlines()[-1] if tail else ""
            raise EncodeError(
                f"Decoding {name} failed after {n} frames (ffmpeg exit code {rc}). {last}".strip(),
                hint=tail or None,
            )
        if n == 0:
            raise EncodeError(f"No frames could be decoded from {name} in the requested time range.")
    finally:
        if not finished and proc.poll() is None:
            proc.kill()
        try:
            if proc.stdout is not None:
                proc.stdout.close()
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
            pass
        log_fh.close()
        try:
            log_path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - Windows lock
            pass


__all__ = ["VideoInfo", "probe", "iter_frames", "parse_ffmpeg_info", "available_filters", "HDR_TRANSFERS"]
