"""Image I/O helpers: robust loading (EXIF orientation, HEIC, 16-bit, alpha, ICC colour
profiles), saving, EXIF field-of-view, resizing."""

from __future__ import annotations

import io
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .errors import InputError
from .fsutil import atomic_write_bytes
from .log import get_logger

log = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif", ".jps", ".mpo"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

# Inputs are the user's own photos.  Pillow's default limit (89.5 MP warning, 179 MP error)
# rejects 200 MP phone photos (Samsung Galaxy Ultra: 16320x12240) as a "decompression bomb".
MAX_IMAGE_PIXELS = 1_000_000_000
if Image.MAX_IMAGE_PIXELS is not None and Image.MAX_IMAGE_PIXELS < MAX_IMAGE_PIXELS:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# The 35 mm-equivalent focal length is defined on the full-frame diagonal (CIPA DC-008).
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)  # 43.27 mm

_heif_registered = False
_srgb_profile = None


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


def _too_large(path: str | Path, e: Exception) -> InputError:
    import re

    m = re.search(r"\((\d+) pixels\)", str(e))
    size = f" ({int(m.group(1)) / 1e6:.0f} MP)" if m else ""
    return InputError(
        f"The photo {Path(path).name} is too large{size}.",
        hint="Save a smaller copy of the photo (for example 50 megapixels or less) and use that.",
    )


def _draft(im: Image.Image, max_px: int) -> None:
    """Let JPEG decode at 1/2, 1/4 or 1/8 scale when that still gives >= ``max_px`` long side."""
    if not max_px or max_px <= 0 or im.format not in ("JPEG", "MPO"):
        return
    w, h = im.size
    long_side = max(w, h)
    if long_side <= max_px:
        return
    try:
        im.draft(None, (max(1, max_px * w // long_side), max(1, max_px * h // long_side)))
    except Exception:  # pragma: no cover - draft is best effort
        pass


def open_pil(path: str | Path, *, max_px: int | None = None, load: bool = True) -> Image.Image:
    """Open an image file with Pillow (HEIC/AVIF via pillow-heif).

    ``max_px`` lets JPEGs decode at a reduced scale whose long side is still at least
    ``max_px`` (much faster and less memory for 50-200 MP photos); other formats always
    decode at full size.  ``load=False`` only reads the header (size, EXIF).
    Raises :class:`InputError` with a readable message for missing / unreadable files.
    """
    _register_heif()
    try:
        im = Image.open(path)
        if max_px:
            _draft(im, int(max_px))
        if load:
            im.load()
    except FileNotFoundError as e:
        raise InputError(f"File not found: {path}") from e
    except Image.DecompressionBombError as e:
        raise _too_large(path, e) from e
    except Exception as e:
        raise InputError(f"Cannot read image {Path(path).name}: {e}") from e
    return im


def _has_alpha(im: Image.Image) -> bool:
    return im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)


def _is_srgb(profile) -> bool:
    from PIL import ImageCms

    try:
        desc = ImageCms.getProfileDescription(profile) or ""
    except Exception:
        return False
    return desc.strip().lower().startswith("srgb")


def to_srgb(im: Image.Image) -> Image.Image:
    """Convert an image with an embedded ICC profile (e.g. iPhone Display P3) to sRGB.

    Returns an RGB (or RGBA, keeping transparency) image in sRGB, or ``im`` unchanged when
    it has no profile, is already sRGB, or the conversion is not possible.  Never raises.
    Everything downstream (outputs, WebXR textures, the Quest) assumes sRGB.
    """
    global _srgb_profile
    icc = im.info.get("icc_profile")
    if not icc or im.mode not in ("RGB", "RGBA", "CMYK", "L", "LA", "P", "PA"):
        return im
    try:
        from PIL import ImageCms

        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if _is_srgb(src):
            return im
        if _srgb_profile is None:
            _srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        alpha = None
        if _has_alpha(im):
            rgba = im.convert("LA" if im.mode == "LA" else "RGBA")
            alpha = rgba.getchannel("A")
            base = rgba.convert("L" if im.mode == "LA" else "RGB")
        elif im.mode == "P":
            base = im.convert("RGB")
        else:
            base = im
        out = ImageCms.profileToProfile(base, src, _srgb_profile, outputMode="RGB")
        if out is None:  # pragma: no cover - only with inPlace=True
            return im
        if alpha is not None:
            out.putalpha(alpha)
        out.info.pop("icc_profile", None)
        return out
    except Exception as e:
        log.debug("Colour profile conversion to sRGB failed (%s); using the pixel values as they are", e)
        return im


def load_image(path: str | Path, with_alpha: bool = False, *, max_px: int | None = None) -> np.ndarray:
    """Load an image as HxWx3 (or HxWx4 if ``with_alpha`` and present) uint8 RGB,
    applying EXIF orientation, converting embedded colour profiles (Display P3, Adobe RGB,
    CMYK) to sRGB and converting 16-bit / palette / CMYK images.

    ``max_px`` (optional) is a hint that the caller will downscale to that long side anyway:
    JPEGs are then decoded at a reduced scale (long side still >= ``max_px``)."""
    im = open_pil(path, max_px=max_px)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float32)
        mx = max(1.0, float(arr.max()))
        arr = (arr / mx * 255.0).clip(0, 255).astype(np.uint8)
        return np.repeat(arr[..., None], 3, axis=2)
    has_alpha = _has_alpha(im)
    im = to_srgb(im)
    if with_alpha and has_alpha:
        return np.ascontiguousarray(np.asarray(im.convert("RGBA")))
    return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def image_from_bytes(data: bytes) -> np.ndarray:
    _register_heif()
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Image.DecompressionBombError as e:
        raise _too_large("image", e) from e
    im = ImageOps.exif_transpose(im)
    im = to_srgb(im)
    return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def _displayed_size(im: Image.Image) -> tuple[int, int]:
    """(width, height) after EXIF orientation, without decoding the pixels."""
    w, h = im.size
    try:
        orientation = int(im.getexif().get(0x0112, 1) or 1)
    except Exception:
        orientation = 1
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def hfov_from_35mm(f35: float, width: int, height: int) -> float:
    """Horizontal FOV (degrees) of a ``width`` x ``height`` image from its 35 mm-equivalent focal length.

    The equivalence is defined on the 43.27 mm full-frame diagonal, so the image diagonal
    maps to 43.27 mm whatever the aspect ratio or orientation; this gives the same focal
    length in pixels for landscape and portrait shots of the same camera.
    """
    if f35 <= 0 or width <= 0 or height <= 0:
        raise ValueError("focal length and image size must be positive")
    half_w_mm = 0.5 * FULL_FRAME_DIAGONAL_MM * width / math.hypot(width, height)
    return math.degrees(2.0 * math.atan(half_w_mm / f35))


def exif_hfov(path: str | Path) -> float | None:
    """Horizontal FOV (degrees) of the *displayed* (EXIF-rotated) image from EXIF
    FocalLengthIn35mmFilm, or None.  Uses the 35 mm-equivalent definition on the
    43.27 mm diagonal (see :func:`hfov_from_35mm`).  Reads only the file header."""
    try:
        with open_pil(path, load=False) as im:  # closes the file (no lingering Windows handle)
            exif = im.getexif()
            sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
            f35 = sub.get(0xA405) or exif.get(0xA405)
            if not f35:
                return None
            f35 = float(f35)
            if f35 <= 0:
                return None
            w, h = _displayed_size(im)
        return hfov_from_35mm(f35, w, h)
    except InputError:
        raise
    except Exception:
        return None


def _encode(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def save_image(path: str | Path, img: np.ndarray, quality: int = 92, xmp: bytes | None = None) -> Path:
    """Save RGB/RGBA uint8 (or float [0,1]) image.  Format from extension.  ``xmp`` is
    embedded for JPEG (e.g. GPano metadata).

    The image is encoded in memory and written atomically (temp file + rename) through
    Python's file I/O, so a full disk raises ``OSError`` instead of leaving a truncated
    file, and an existing file is only replaced by a complete one."""
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
            data = _encode(im, "JPEG", **kw)
        except (OSError, ValueError):
            # Pillow's optimize pass can fail on very detailed images ("Suspension not
            # allowed here"); retry without it.
            kw.pop("optimize", None)
            data = _encode(im, "JPEG", **kw)
    elif ext == ".png":
        data = _encode(im, "PNG", compress_level=6)
    elif ext == ".webp":
        data = _encode(im, "WEBP", quality=quality)
    else:
        fmt = Image.registered_extensions().get(ext)
        if fmt is None:
            raise ValueError(f"unknown file extension: {ext}")
        data = _encode(im, fmt)
    atomic_write_bytes(path, data, fsync=False)
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
        atomic_write_bytes(path, buf.tobytes(), fsync=False)  # no truncated file on a full disk
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
