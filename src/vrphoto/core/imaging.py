"""Image I/O helpers: robust loading (EXIF orientation, HEIC, 16-bit, alpha), saving,
EXIF field-of-view, resizing."""

from __future__ import annotations

import io
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .errors import InputError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif", ".jps", ".mpo"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

_heif_registered = False


def _register_heif() -> None:
    global _heif_registered
    if _heif_registered:
        return
    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
    except Exception:  # pragma: no cover - optional dependency
        pass
    _heif_registered = True


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def open_pil(path: str | Path) -> Image.Image:
    _register_heif()
    try:
        im = Image.open(path)
        im.load()
    except FileNotFoundError as e:
        raise InputError(f"File not found: {path}") from e
    except Exception as e:
        raise InputError(f"Cannot read image {Path(path).name}: {e}") from e
    return im


def load_image(path: str | Path, with_alpha: bool = False) -> np.ndarray:
    """Load an image as HxWx3 (or HxWx4 if ``with_alpha`` and present) uint8 RGB,
    applying EXIF orientation and converting 16-bit / palette / CMYK images."""
    im = open_pil(path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float32)
        mx = max(1.0, float(arr.max()))
        arr = (arr / mx * 255.0).clip(0, 255).astype(np.uint8)
        return np.repeat(arr[..., None], 3, axis=2)
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    if with_alpha and has_alpha:
        return np.ascontiguousarray(np.asarray(im.convert("RGBA")))
    return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def image_from_bytes(data: bytes) -> np.ndarray:
    _register_heif()
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def exif_hfov(path: str | Path) -> float | None:
    """Horizontal FOV (degrees) of the *displayed* (EXIF-rotated) image from EXIF
    FocalLengthIn35mmFilm, or None.  35mm full frame is 36x24 mm; we use the long
    side for landscape and the short side for portrait images."""
    try:
        im = open_pil(path)
        exif = im.getexif()
        sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
        f35 = sub.get(0xA405) or exif.get(0xA405)
        if not f35:
            return None
        f35 = float(f35)
        if f35 <= 0:
            return None
        w, h = ImageOps.exif_transpose(im).size
        sensor = 36.0 if w >= h else 24.0
        # 35mm-equivalent focal length is defined against the 43.27mm diagonal; using the
        # side length directly is the common convention and accurate for 3:2 sensors.
        return math.degrees(2.0 * math.atan(sensor / (2.0 * f35)))
    except InputError:
        raise
    except Exception:
        return None


def save_image(path: str | Path, img: np.ndarray, quality: int = 92, xmp: bytes | None = None) -> Path:
    """Save RGB/RGBA uint8 (or float [0,1]) image.  Format from extension.  ``xmp`` is
    embedded for JPEG (e.g. GPano metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(img)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        if im.mode == "RGBA":
            im = im.convert("RGB")
        kw = {"quality": quality, "subsampling": 0 if quality >= 90 else 2, "optimize": True}
        if xmp:
            kw["xmp"] = xmp
        try:
            im.save(path, "JPEG", **kw)
        except (OSError, ValueError):
            # Pillow's optimize pass can fail on very detailed images ("Suspension not
            # allowed here"); retry without it.
            kw.pop("optimize", None)
            im.save(path, "JPEG", **kw)
    elif ext == ".png":
        im.save(path, "PNG", compress_level=6)
    elif ext == ".webp":
        im.save(path, "WEBP", quality=quality)
    else:
        im.save(path)
    return path


def save_depth16(path: str | Path, disparity: np.ndarray) -> Path:
    """Save a [0,1] float map as 16-bit grayscale PNG (1 = white = near)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(disparity, 0, 1) * 65535.0 + 0.5).astype(np.uint16)
    # cv2.imwrite cannot handle non-ASCII paths on Windows: encode in memory, write via Python
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise InputError(f"Cannot encode depth map for {path.name}")
    try:
        path.write_bytes(buf.tobytes())
    except OSError as e:
        raise InputError(f"Cannot write {path}: {e}") from e
    return path


def load_depth16(path: str | Path) -> np.ndarray:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)  # unicode-safe (cv2.imread is not on Windows)
    except OSError as e:
        raise InputError(f"Cannot read depth map {path}: {e}") from e
    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if arr is None:
        raise InputError(f"Cannot read depth map {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    mx = 65535.0 if arr.dtype == np.uint16 else 255.0
    return (arr.astype(np.float32) / mx).clip(0, 1)


def resize_long(img: np.ndarray, long_side: int, interp: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize so the longer side equals ``long_side`` exactly (up or down).
    Use :func:`limit_long` to only downscale."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if (nw, nh) == (w, h):
        return img
    if s > 1 and interp == cv2.INTER_AREA:
        interp = cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp)


def limit_long(img: np.ndarray, max_long: int) -> tuple[np.ndarray, float]:
    """Downscale so the long side is <= max_long.  Returns (image, scale)."""
    h, w = img.shape[:2]
    if max(h, w) <= max_long:
        return img, 1.0
    s = max_long / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA), s


def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def colorize_depth(disparity: np.ndarray) -> np.ndarray:
    """RGB visualisation of a [0,1] disparity map (turbo colormap, near = warm)."""
    u8 = (np.clip(disparity, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def gpano_xmp(full_w: int, full_h: int, crop_w: int, crop_h: int, left: int, top: int) -> bytes:
    """Minimal GPano XMP packet for equirectangular (VR180 / 360) JPEGs."""
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/" '
        'GPano:ProjectionType="equirectangular" GPano:UsePanoramaViewer="True" '
        f'GPano:FullPanoWidthPixels="{full_w}" GPano:FullPanoHeightPixels="{full_h}" '
        f'GPano:CroppedAreaImageWidthPixels="{crop_w}" GPano:CroppedAreaImageHeightPixels="{crop_h}" '
        f'GPano:CroppedAreaLeftPixels="{left}" GPano:CroppedAreaTopPixels="{top}"/>'
        '</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    ).encode()
