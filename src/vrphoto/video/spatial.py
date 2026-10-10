"""Spherical (360 / VR180) and stereo metadata for MP4/MOV files (algorithm K).

:func:`inject` writes, for every video track:

* **V1** - the Google ``GSpherical`` RDF/XML in a ``uuid`` box at track level, with the
  cropped-area fields for 180 (``CroppedAreaImageWidthPixels`` = one eye,
  ``FullPanoWidthPixels`` = twice that, ``CroppedAreaLeftPixels`` = a quarter of the pano);
* **V2** - ``st3d`` (0 mono, 1 top-bottom, 2 left-right) and ``sv3d`` ->
  ``svhd`` + ``proj`` -> ``prhd`` + ``equi`` inside the video sample entry, with
  equirect bounds ``left = right = 0x40000000`` (25% cropped each side) for 180.

The box tree is parsed and re-serialised with the vendored Google ``spatialmedia``
library, which rewrites ``stco``/``co64`` chunk offsets when the ``moov`` box grows and
keeps the box order, so a faststart file (``moov`` before ``mdat``) stays faststart.
The file is replaced atomically (temp file + ``os.replace``, retried while Windows
reports the fresh file as busy).  :func:`read` reports what
a file contains.  Meta Quest players also rely on the filename tokens (``_180_LR``,
``_360_TB``...), which the pipeline adds.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, BinaryIO

from ..core.errors import EncodeError, InputError
from ..core.fsutil import replace_with_retry
from ..core.log import get_logger

log = get_logger(__name__)

#: 0.32 fixed-point quarter: the left/right equirect bounds of a 180 degree video.
BOUNDS_180 = 0x40000000
METADATA_SOURCE = "VR Photo Studio"

_STEREO_ALIASES = {
    "mono": "mono", "2d": "mono", "none": "mono",
    "lr": "lr", "sbs": "lr", "left-right": "lr", "side-by-side": "lr",
    "tb": "tb", "ou": "tb", "top-bottom": "tb", "over-under": "tb",
}  # fmt: skip
_ST3D_CODES = {"mono": 0, "tb": 1, "lr": 2}
_ST3D_NAMES = {v: k for k, v in _ST3D_CODES.items()}
_V1_STEREO = {"mono": "mono", "lr": "left-right", "tb": "top-bottom"}
_V1_STEREO_REV = {"mono": "mono", "left-right": "lr", "top-bottom": "tb"}
_EXTS = {".mp4", ".m4v", ".mov"}


def _sm() -> Any:
    """The vendored spatialmedia package (imported lazily)."""
    from ..third_party import spatialmedia

    return spatialmedia


def _norm_stereo(stereo: str) -> str:
    key = str(stereo).strip().lower()
    if key not in _STEREO_ALIASES:
        raise InputError(
            f"Unknown stereo layout {stereo!r}; use 'mono', 'lr' (side-by-side) or 'tb' (top-bottom)."
        )
    return _STEREO_ALIASES[key]


def _norm_coverage(coverage: int) -> int:
    try:
        cov = int(coverage)
    except (TypeError, ValueError):
        cov = -1
    if cov not in (180, 360):
        raise InputError(f"Spherical coverage must be 180 or 360 degrees (got {coverage!r}).")
    return cov


def _check_file(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"Video file not found: {p}")
    if p.suffix.lower() not in _EXTS:
        raise InputError(
            f"Spatial metadata is only supported for MP4/MOV files (got {p.suffix or 'no extension'})."
        )
    return p


# --------------------------------------------------------------------------- box tree helpers
def _load(fh: BinaryIO, name: str) -> Any:
    try:
        mp4 = _sm().mpeg.load(fh)
    except Exception as exc:  # struct.error / IndexError / ... on malformed boxes
        raise EncodeError(f"{name} is not a valid MP4/MOV file ({exc}).") from exc
    if mp4 is None:
        raise EncodeError(f"{name} is not a valid MP4/MOV file (could not parse its boxes).")
    return mp4


def _children(box: Any, tag: bytes) -> list[Any]:
    return [c for c in (getattr(box, "contents", None) or []) if getattr(c, "name", None) == tag]


def _handler(fh: BinaryIO, mdia: Any) -> bytes:
    c = _sm().mpeg.constants
    for hdlr in _children(mdia, c.TAG_HDLR):
        fh.seek(hdlr.content_start() + 8)  # version/flags + pre_defined
        return fh.read(4)
    return b""


def _video_tracks(mp4: Any, fh: BinaryIO) -> list[tuple[Any, list[Any]]]:
    """``[(trak, [video sample entries])]`` for every video track."""
    c = _sm().mpeg.constants
    out: list[tuple[Any, list[Any]]] = []
    for trak in _children(mp4.moov_box, c.TAG_TRAK):
        for mdia in _children(trak, c.TAG_MDIA):
            if _handler(fh, mdia) != c.TAG_VIDE:
                continue
            entries: list[Any] = []
            for minf in _children(mdia, c.TAG_MINF):
                for stbl in _children(minf, c.TAG_STBL):
                    for stsd in _children(stbl, c.TAG_STSD):
                        entries += [e for e in stsd.contents if e.name in c.VIDEO_SAMPLE_DESCRIPTIONS]
            out.append((trak, entries))
    return out


def _entry_size(fh: BinaryIO, entry: Any) -> tuple[int, int]:
    """Coded width/height from a VisualSampleEntry."""
    fh.seek(entry.content_start() + 24)
    w, h = struct.unpack(">HH", fh.read(4))
    return int(w), int(h)


_MAX_XML_BYTES = 1 << 20


def _uuid_payload(fh: BinaryIO, box: Any) -> bytes:
    if box.contents:
        return bytes(box.contents)
    fh.seek(box.content_start())
    return fh.read(min(box.content_size, _MAX_XML_BYTES))


def _is_spherical_uuid(fh: BinaryIO, box: Any) -> bool:
    if box.contents:
        return bytes(box.contents[:16]) == _sm().metadata_utils.SPHERICAL_UUID_ID
    fh.seek(box.content_start())
    return fh.read(16) == _sm().metadata_utils.SPHERICAL_UUID_ID


def _top_level_order(path: Path) -> list[bytes]:
    """Names of the top-level boxes (cheap scan, used to verify faststart)."""
    names: list[bytes] = []
    size = path.stat().st_size
    with open(path, "rb") as fh:
        pos = 0
        while pos + 8 <= size:
            fh.seek(pos)
            hdr = fh.read(8)
            n, name = struct.unpack(">I4s", hdr)
            if n == 1:
                n = struct.unpack(">Q", fh.read(8))[0]
            elif n == 0:
                n = size - pos
            if n < 8:
                break
            names.append(name)
            pos += n
    return names


# --------------------------------------------------------------------------- V1 XML
def spherical_xml(coverage: int, stereo: str, width: int, height: int) -> str:
    """GSpherical V1 RDF/XML for a ``width x height`` frame."""
    stereo = _norm_stereo(stereo)
    coverage = _norm_coverage(coverage)
    crop = None
    if coverage == 180:
        eye_w = width // 2 if stereo == "lr" else width
        eye_h = height // 2 if stereo == "tb" else height
        if eye_w < 1 or eye_h < 1:
            raise InputError(f"Video frame {width}x{height} is too small for {stereo} 180 metadata.")
        crop = f"{eye_w}:{eye_h}:{2 * eye_w}:{eye_h}:{eye_w // 2}:0"
    xml = _sm().metadata_utils.generate_spherical_xml(
        projection="equirectangular", stereo=_V1_STEREO[stereo], crop=crop
    )
    if not xml:
        raise EncodeError("Could not build the spherical metadata XML.")
    return xml.replace("Spherical Metadata Tool", METADATA_SOURCE)


def _parse_v1(xml_bytes: bytes) -> dict[str, str] | None:
    text = xml_bytes.decode("utf-8", errors="replace")
    if "<!DOCTYPE" in text or "<!ENTITY" in text:  # refuse entity tricks in untrusted files
        log.warning("Ignoring spherical XML with a DOCTYPE/ENTITY declaration")
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # some writers omit the rdf namespace declaration (see upstream parse_spherical_xml)
        fixed = text.replace(
            "<rdf:SphericalVideo",
            '<rdf:SphericalVideo xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            1,
        )
        try:
            root = ET.fromstring(fixed)
        except ET.ParseError:
            log.warning("Unparseable spherical XML metadata")
            return None
    out: dict[str, str] = {}
    for child in root:
        tag = re.sub(r"^\{[^}]*\}", "", child.tag)
        out[tag] = (child.text or "").strip()
    return out


# --------------------------------------------------------------------------- public API
def inject(path: str | os.PathLike[str], *, coverage: int, stereo: str) -> Path:
    """Add 180/360 spherical + stereo metadata (V1 and V2) to an MP4/MOV, in place.

    Args:
        path: the video (``.mp4``/``.mov``); replaced atomically.
        coverage: 180 or 360 (horizontal field of the equirect frame).
        stereo: ``"mono"``, ``"lr"`` (side-by-side, left eye left) or ``"tb"`` (left on top).

    Existing spherical metadata is replaced.  Returns ``path``.

    Raises:
        InputError: bad arguments or not an MP4/MOV.
        EncodeError: the file cannot be parsed or written.
    """
    p = _check_file(path)
    cov = _norm_coverage(coverage)
    st = _norm_stereo(stereo)
    sm = _sm()
    c, sv3d, container = sm.mpeg.constants, sm.mpeg.sv3d, sm.mpeg.container
    order_before = _top_level_order(p)
    tmp = p.with_name(f".{p.stem}.{uuid.uuid4().hex[:8]}.spatial{p.suffix}")
    try:
        with open(p, "rb") as fh:
            mp4 = _load(fh, p.name)
            tracks = _video_tracks(mp4, fh)
            if not tracks or not any(entries for _t, entries in tracks):
                raise EncodeError(f"{p.name} has no supported video track (H.264/HEVC/AV1/VP9/ProRes).")
            for trak, entries in tracks:
                if not entries:
                    continue
                width, height = _entry_size(fh, entries[0])
                # V1: replace the spherical uuid box only (keep other uuid boxes)
                trak.contents = [
                    b for b in trak.contents if not (b.name == c.TAG_UUID and _is_spherical_uuid(fh, b))
                ]
                xml = spherical_xml(cov, st, width, height)
                trak.contents.append(sm.metadata_utils.spherical_uuid(xml))
                # V2: st3d + sv3d in every video sample entry
                for entry in entries:
                    entry.contents = [b for b in entry.contents if b.name not in (c.TAG_ST3D, c.TAG_SV3D)]
                    st3d = sv3d.ST3DBox.create()
                    st3d.stereo_mode = _ST3D_CODES[st]
                    proj = container.Container(header_size=8)
                    proj.name = c.TAG_PROJ
                    proj.contents.append(sv3d.PRHDBox.create())
                    lr = BOUNDS_180 if cov == 180 else 0
                    proj.contents.append(sv3d.EQUIBox.create(bounds=[0, 0, lr, lr]))
                    box = container.Container(header_size=8)
                    box.name = c.TAG_SV3D
                    box.contents.append(sv3d.SVHDBox.create(METADATA_SOURCE))
                    box.contents.append(proj)
                    entry.contents.append(st3d)
                    entry.contents.append(box)
            mp4.resize()
            with open(tmp, "wb") as out:
                mp4.save(fh, out)
        got = read(tmp)
        if not (got["spherical"] and got["stereo"] == st and got["coverage"] == cov):
            raise EncodeError(f"Spatial metadata verification failed for {p.name}: {got}")
        if _top_level_order(tmp) != order_before:
            raise EncodeError(f"Rewriting {p.name} changed its box layout; metadata not applied.")
        try:
            shutil.copymode(p, tmp)  # keep the original file permissions
        except OSError as exc:  # pragma: no cover - exotic filesystems
            log.debug("Could not copy permissions of %s: %s", p.name, exc)
        replace_with_retry(tmp, p)  # Explorer thumbnailer / AV / sync may hold the fresh MP4
    except EncodeError:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        busy = isinstance(exc, PermissionError)
        raise EncodeError(
            f"Could not write spatial metadata to {p.name}: {exc}",
            hint="The file is busy (another program or antivirus). Close any player that shows the "
            "video and try again."
            if busy
            else "Check that the disk has free space and the folder is writable, then try again.",
        ) from exc
    except Exception as exc:  # parser failure on an unusual file
        tmp.unlink(missing_ok=True)
        raise EncodeError(f"Could not add spatial metadata to {p.name}: {exc}") from exc
    log.info("Injected %d %s spherical metadata into %s", cov, st, p.name)
    return p


def read(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Report the spherical / stereo metadata of an MP4/MOV.

    Returns ``{"spherical": bool, "stereo": "mono"|"lr"|"tb"|None, "coverage": 180|360|...|None,
    "projection": "equirectangular"|"cubemap"|"mesh"|None, "bounds": {"top", "bottom",
    "left", "right"} | None, "v1": {GSpherical tag: text} | None, "v2": bool}``.
    V2 values take priority over V1 when both exist.
    """
    p = _check_file(path)
    sm = _sm()
    c = sm.mpeg.constants
    res: dict[str, Any] = {"spherical": False, "stereo": None, "coverage": None, "projection": None,
                           "bounds": None, "v1": None, "v2": False}  # fmt: skip
    with open(p, "rb") as fh:
        mp4 = _load(fh, p.name)
        for trak, entries in _video_tracks(mp4, fh):
            for b in _children(trak, c.TAG_UUID):
                if res["v1"] is None and _is_spherical_uuid(fh, b):
                    res["v1"] = _parse_v1(_uuid_payload(fh, b)[16:])
            for entry in entries:
                for st3d in _children(entry, c.TAG_ST3D):
                    res["stereo"] = _ST3D_NAMES.get(int(st3d.stereo_mode), f"custom({st3d.stereo_mode})")
                for box in _children(entry, c.TAG_SV3D):
                    res["v2"] = True
                    for proj in _children(box, c.TAG_PROJ):
                        for sub in proj.contents:
                            if sub.name == c.TAG_EQUI:
                                res["projection"] = "equirectangular"
                                res["bounds"] = {"top": int(sub.bounds_top), "bottom": int(sub.bounds_bottom),
                                                 "left": int(sub.bounds_left), "right": int(sub.bounds_right)}  # fmt: skip
                            elif sub.name == b"cbmp":
                                res["projection"] = "cubemap"
                            elif sub.name == b"mshp":
                                res["projection"] = "mesh"
            if res["v2"] or res["v1"]:
                break
    if res["v2"]:
        res["spherical"] = res["projection"] is not None
        if res["bounds"] is not None:
            frac = 1.0 - (res["bounds"]["left"] + res["bounds"]["right"]) / float(1 << 32)
            res["coverage"] = int(round(360.0 * frac))
        elif res["spherical"]:
            res["coverage"] = 360
    v1 = res["v1"]
    if v1:
        if not res["v2"]:
            res["spherical"] = v1.get("Spherical", "").lower() == "true"
            proj = v1.get("ProjectionType", "").lower()
            res["projection"] = proj if res["spherical"] and proj else res["projection"]
            if res["spherical"]:
                try:
                    cw, fw = int(v1["CroppedAreaImageWidthPixels"]), int(v1["FullPanoWidthPixels"])
                    res["coverage"] = int(round(360.0 * cw / fw)) if fw > 0 else 360
                except (KeyError, ValueError):
                    res["coverage"] = 360
        if res["stereo"] is None and v1.get("StereoMode"):
            res["stereo"] = _V1_STEREO_REV.get(v1["StereoMode"].lower(), v1["StereoMode"])
    if res["spherical"] and res["stereo"] is None:
        res["stereo"] = "mono"
    return res


__all__ = ["inject", "read", "spherical_xml", "BOUNDS_180"]
