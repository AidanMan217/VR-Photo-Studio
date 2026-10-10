"""About / Licenses dialog: version, system info, links and third-party notices."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import Qt, qVersion
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ...core.log import get_logger
from .. import util
from ..workers import DIALOG_CLOSE_WAIT_MS, TaskRunner

log = get_logger(__name__)

NOTICES_FALLBACK = """# Third-party notices

The full list of third-party components and their licenses is in the file
THIRD_PARTY_NOTICES.txt next to the program (VRPhotoStudio.exe), and online at
{url}/blob/HEAD/THIRD_PARTY_NOTICES.md

VR Photo Studio is built with Python, Qt for Python (PySide6, LGPL-3.0), NumPy, OpenCV,
Pillow, ONNX Runtime, FFmpeg (via imageio-ffmpeg), three.js, Spark and other open-source
software. AI models are downloaded separately and keep their own licenses (shown in the
AI models window); models marked NON-COMMERCIAL may only be used privately.
"""


def notices_candidates() -> list[Path]:
    """Where THIRD_PARTY_NOTICES may live (bundled, next to the exe, source checkout)."""
    from ...core.config import is_frozen, resource_path

    cands = [resource_path("THIRD_PARTY_NOTICES.md"), resource_path("THIRD_PARTY_NOTICES.txt")]
    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        cands += [exe_dir / "THIRD_PARTY_NOTICES.txt", exe_dir / "THIRD_PARTY_NOTICES.md"]
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            cands += [
                Path(meipass) / "THIRD_PARTY_NOTICES.md",
                Path(meipass) / "licenses" / "THIRD_PARTY_NOTICES.txt",
            ]
    # source checkout: src/vrphoto -> repo root
    cands.append(resource_path().parent.parent / "THIRD_PARTY_NOTICES.md")
    return cands


def load_notices() -> tuple[str, bool]:
    """``(text, is_markdown)`` - the bundled notices, or a fallback text."""
    for p in notices_candidates():
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8", errors="replace"), True
        except OSError:
            continue
    return NOTICES_FALLBACK.format(url=util.PROJECT_URL), True


def system_summary() -> str:
    from ... import __version__

    lines = [
        f"VR Photo Studio {__version__}",
        f"Python {platform.python_version()} ({platform.architecture()[0]})",
        f"Qt {qVersion()} / PySide6 {pyside_version}",
        f"{platform.system()} {platform.release()} ({platform.machine()})",
    ]
    try:
        from ...models.ort import available_providers

        lines.append("ONNX Runtime providers: " + ", ".join(available_providers()))
    except Exception:
        pass
    try:
        from ...core.config import data_root

        lines.append(f"Data folder: {data_root()}")
    except Exception:
        pass
    return "\n".join(lines)


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from ... import APP_NAME, __version__

        self.setWindowTitle(f"About {APP_NAME}")
        self.resize(720, 600)
        icon = QLabel()
        try:
            from ...core.config import resource_path

            pm = QPixmap(str(resource_path("gui", "resources", "icon.png")))
            if not pm.isNull():
                icon.setPixmap(
                    pm.scaled(
                        96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                    )
                )
        except Exception:
            pass
        title = QLabel(f"{APP_NAME} {__version__}")
        title.setObjectName("Heading")
        blurb = QLabel(
            "Turn your photos into VR content for Meta Quest: 3D photos, living photos, VR180 and 360, "
            "walk-around 3D scenes and life-size people and objects in passthrough AR.<br><br>"
            f'<a href="{util.PROJECT_URL}">Project page</a> · '
            f'<a href="{util.USER_GUIDE_URL}">User guide</a> · '
            f'<a href="{util.ISSUES_URL}">Report a problem</a>'
        )
        blurb.setWordWrap(True)
        blurb.setOpenExternalLinks(True)
        blurb.setTextFormat(Qt.TextFormat.RichText)
        head = QHBoxLayout()
        head.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        hv = QVBoxLayout()
        hv.addWidget(title)
        hv.addWidget(blurb)
        head.addLayout(hv, 1)

        self.tabs = QTabWidget()
        self.notices = QTextBrowser()
        self.notices.setOpenExternalLinks(True)
        text, md = load_notices()
        if md:
            self.notices.setMarkdown(text)
        else:  # pragma: no cover - load_notices always returns markdown-compatible text
            self.notices.setPlainText(text)
        self.notices_text = text
        self.tabs.addTab(self.notices, "Licenses")
        self.system = QTextBrowser()
        self.system.setPlainText(f"{APP_NAME} {__version__}\nCollecting system information…")
        self.tabs.addTab(self.system, "System")
        # importing ONNX Runtime can take a moment: never on the UI thread
        self.tasks = TaskRunner(self, max_threads=1)
        self.tasks.start(
            system_summary,
            pass_progress=False,
            pass_cancel=False,
            on_result=self.system.setPlainText,
            name="system-info",
        )
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(head)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(box)

    def done(self, result: int) -> None:  # noqa: D401 - QDialog API
        self.tasks.shutdown(DIALOG_CLOSE_WAIT_MS)
        super().done(result)
