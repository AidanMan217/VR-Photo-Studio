"""Small Qt + non-Qt helpers shared by the GUI: image conversion, media discovery,
thumbnails, friendly error text, message boxes and opening files/folders.

Message-box helpers (:func:`ask_yes_no`, :func:`show_error`, :func:`show_info`) are the
single place where the GUI blocks on user input, so tests can monkeypatch them.
"""

from __future__ import annotations

import html
import os
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import QMessageBox, QWidget

from ..core.log import get_logger

log = get_logger(__name__)

#: Repository / documentation links shown in Help menus and dialogs.
PROJECT_URL = "https://github.com/AidanMan217/VR-Photo-Studio"
USER_GUIDE_URL = f"{PROJECT_URL}/blob/main/docs/USER_GUIDE.md"
ISSUES_URL = f"{PROJECT_URL}/issues"


# --------------------------------------------------------------------------- media files
def media_exts() -> tuple[set[str], set[str]]:
    """``(IMAGE_EXTS, VIDEO_EXTS)`` from :mod:`vrphoto.core.imaging` (lower-case, with dot)."""
    from ..core.imaging import IMAGE_EXTS, VIDEO_EXTS

    return set(IMAGE_EXTS), set(VIDEO_EXTS)


def is_media(path: str | Path) -> bool:
    """True if ``path`` has a supported photo or video extension."""
    images, videos = media_exts()
    return Path(path).suffix.lower() in images | videos


def is_video_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in media_exts()[1]


def collect_media_ex(
    paths: Iterable[str | Path], *, max_files: int = 5000, time_budget_s: float = 8.0
) -> tuple[list[Path], bool]:
    """Like :func:`collect_media`; also returns True if the search stopped early (more
    than ``max_files`` files, or a huge folder tree took longer than ``time_budget_s``)."""
    import time

    out: list[Path] = []
    seen: set[str] = set()
    deadline = time.monotonic() + max(0.1, float(time_budget_s))
    truncated = False

    def add(p: Path) -> bool:
        """Add ``p`` (duplicates are ignored); False once the limit is reached."""
        nonlocal truncated
        key = os.path.normcase(str(p.resolve()))
        if key in seen:
            return True
        if len(out) >= max_files:
            truncated = True
            return False
        seen.add(key)
        out.append(p)
        return True

    for raw in paths:
        p = Path(raw)
        try:
            if p.is_dir():
                for dirpath, dirnames, filenames in os.walk(p):
                    dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                    for name in sorted(filenames, key=str.lower):
                        if not name.startswith(".") and is_media(name) and not add(Path(dirpath) / name):
                            break
                    if truncated or time.monotonic() > deadline:
                        truncated = True
                        break
            elif p.is_file() and is_media(p):
                add(p)
        except OSError as e:  # unreadable folder, broken link...
            log.warning("Skipping %s: %s", p, e)
        if truncated:
            log.warning("Stopped looking for photos after %d files", len(out))
            break
    return out, truncated


def collect_media(paths: Iterable[str | Path], *, max_files: int = 5000) -> list[Path]:
    """Expand files and folders (recursively) into supported media files.

    Order is preserved for explicitly given files; folder contents are sorted by name.
    Hidden files/folders (starting with ``.``) are skipped.  Duplicates are removed.  The
    search stops after ``max_files`` files (or a few seconds in a gigantic folder tree).
    """
    return collect_media_ex(paths, max_files=max_files)[0]


def file_dialog_filter() -> str:
    """Name filter string for QFileDialog (photos, videos, all supported)."""
    images, videos = media_exts()
    img = " ".join(f"*{e}" for e in sorted(images))
    vid = " ".join(f"*{e}" for e in sorted(videos))
    return f"Photos and videos ({img} {vid});;Photos ({img});;Videos ({vid});;All files (*)"


# --------------------------------------------------------------------------- image loading (worker threads)
def _open_pil_fast(path: Path, size: int):
    """Open an image with Pillow, decoding JPEGs at reduced size (``draft``) for speed."""
    from PIL import Image, ImageOps

    from ..core import imaging

    reg = getattr(imaging, "_register_heif", None)
    if callable(reg):
        reg()
    im = Image.open(path)
    try:
        im.draft("RGB", (size, size))
    except Exception:  # draft is a no-op / unsupported for many formats
        pass
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im


def video_first_frame(path: str | Path, max_long: int = 1024) -> np.ndarray:
    """First frame of a video as RGB uint8 (via :mod:`vrphoto.video.decode`, else OpenCV)."""
    from ..core.errors import InputError

    try:
        from ..video.decode import iter_frames

        gen = iter_frames(path, max_long=max_long)
        try:
            frame = next(gen)
        finally:
            close = getattr(gen, "close", None)
            if callable(close):
                close()
        return np.ascontiguousarray(frame)
    except StopIteration as e:
        raise InputError(f"The video {Path(path).name} has no frames.") from e
    except Exception as e:  # decoder not available -> OpenCV fallback
        log.debug("video.decode failed for %s (%s); trying OpenCV", path, e)
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        ok, bgr = cap.read()
    finally:
        cap.release()
    if not ok or bgr is None:
        raise InputError(f"Cannot read the video {Path(path).name}.")
    from ..core.imaging import limit_long

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(limit_long(rgb, max_long)[0])


def load_thumbnail(path: str | Path, size: int = 96) -> np.ndarray:
    """Small RGB thumbnail (long side ``size``) of a photo or the first frame of a video."""
    p = Path(path)
    if is_video_path(p):
        from ..core.imaging import limit_long

        return limit_long(video_first_frame(p, max_long=size * 2), size)[0]
    im = _open_pil_fast(p, size * 2)
    im = im.convert("RGB")
    im.thumbnail((size, size))
    return np.ascontiguousarray(np.asarray(im))


def load_preview_source(path: str | Path, max_long: int = 1600) -> np.ndarray:
    """Load a photo (EXIF-rotated) or a video's first frame for previews, long side <= ``max_long``."""
    from ..core.imaging import limit_long, load_image

    p = Path(path)
    if is_video_path(p):
        return video_first_frame(p, max_long=max_long)
    img = load_image(p)
    return np.ascontiguousarray(limit_long(img, max_long)[0])


# --------------------------------------------------------------------------- numpy <-> Qt
def np_to_qimage(img: np.ndarray) -> QImage:
    """Convert ``HxW`` gray, ``HxWx3`` RGB or ``HxWx4`` RGBA uint8 (or float [0,1]) to a QImage.

    The returned image owns a copy of the data.
    """
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = np.ascontiguousarray(a)
        h, w = a.shape
        q = QImage(a.data, w, h, a.strides[0], QImage.Format.Format_Grayscale8)
        return q.copy()
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape {a.shape}")
    a = np.ascontiguousarray(a)
    h, w, c = a.shape
    fmt = QImage.Format.Format_RGB888 if c == 3 else QImage.Format.Format_RGBA8888
    q = QImage(a.data, w, h, a.strides[0], fmt)
    return q.copy()


def np_to_pixmap(img: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(np_to_qimage(img))


def png_to_pixmap(data: bytes) -> QPixmap:
    pm = QPixmap()
    pm.loadFromData(data, "PNG")
    return pm


# --------------------------------------------------------------------------- formatting
def human_size_mb(mb: float | None) -> str:
    """``1536`` -> ``"1.5 GB"``; ``12.3`` -> ``"12 MB"``; None/0 -> ``"?"``."""
    if not mb:
        return "?"
    if mb >= 1000:
        return f"{mb / 1000:.1f} GB"
    if mb >= 10:
        return f"{mb:.0f} MB"
    return f"{mb:.1f} MB"


def human_duration(seconds: float) -> str:
    """``0.4`` -> ``"0.4 s"``; ``5`` -> ``"5 s"``; ``75`` -> ``"1 min 15 s"``; hours too."""
    seconds = max(0.0, float(seconds))
    if seconds < 0.05:
        return "less than 0.1 s"
    if seconds < 10:
        return f"{seconds:.1f}".rstrip("0").rstrip(".") + " s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def error_text(exc: BaseException | str) -> tuple[str, str | None]:
    """``(message, hint)`` for showing an exception to a non-technical user."""
    if isinstance(exc, str):
        return exc, None
    from ..core.errors import VRPhotoError

    if isinstance(exc, VRPhotoError):
        return str(exc) or type(exc).__name__, getattr(exc, "hint", None)
    if isinstance(exc, MemoryError):
        return (
            "The computer ran out of memory.",
            "Lower 'Max processing size' in the options or close other programs, then try again.",
        )
    if isinstance(exc, PermissionError):
        return (f"Permission denied: {exc}", "Choose another output folder or close programs using the file.")
    if isinstance(exc, FileNotFoundError):
        return (f"File not found: {exc.filename or exc}", None)
    msg = str(exc).strip()
    return (f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__), None


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def rich_escape(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


# --------------------------------------------------------------------------- message boxes
def ask_yes_no(
    parent: QWidget | None,
    title: str,
    text: str,
    *,
    default_yes: bool = False,
    informative: str | None = None,
) -> bool:
    """Ask a Yes/No question; returns True for Yes.  (Tests monkeypatch this.)"""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    if informative:
        box.setInformativeText(informative)
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.setDefaultButton(QMessageBox.StandardButton.Yes if default_yes else QMessageBox.StandardButton.No)
    return box.exec() == QMessageBox.StandardButton.Yes


def show_error(
    parent: QWidget | None, title: str, exc: BaseException | str, *, details: str | None = None
) -> None:
    """Show an error with a friendly message, an optional hint and expandable details."""
    msg, hint = error_text(exc)
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(msg)
    if hint:
        box.setInformativeText(hint)
    if details is None and isinstance(exc, BaseException) and exc.__traceback__ is not None:
        details = format_exception(exc)
    if details:
        box.setDetailedText(details)
    box.exec()


def show_info(parent: QWidget | None, title: str, text: str) -> None:
    QMessageBox.information(parent, title, text)


# --------------------------------------------------------------------------- shell
def open_path(path: str | Path) -> bool:
    """Open a file or folder with the system's default application (Explorer, ...)."""
    p = Path(path)
    if not p.exists():
        log.warning("Cannot open %s: it does not exist", p)
        return False
    ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))
    if not ok:
        log.warning("The system could not open %s", p)
    return bool(ok)


def open_url(url: str) -> bool:
    ok = QDesktopServices.openUrl(QUrl(url))
    if not ok:
        log.warning("Could not open %s", url)
    return bool(ok)


def output_folder_of(outputs: dict[str, Any] | None, fallback: str | Path | None = None) -> Path | None:
    """Folder containing the first existing output file, else the first existing output
    folder (e.g. a viewer scene), else ``fallback``."""
    paths: list[Path] = []
    for v in (outputs or {}).values():
        try:
            paths.append(Path(str(v)))
        except (TypeError, ValueError):
            continue
    for p in paths:
        if p.is_file():
            return p.parent
    for p in paths:
        if p.is_dir():
            return p
    return Path(fallback) if fallback else None
