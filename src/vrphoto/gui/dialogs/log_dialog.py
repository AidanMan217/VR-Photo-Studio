"""Log viewer: the tail of the log file (auto-refreshing) and a button to open its folder."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.log import get_logger
from .. import util
from . import common as dc

log = get_logger(__name__)


def default_log_path() -> Path:
    """The current log file (as configured by ``setup_logging``) or the default location."""
    from ...core import log as core_log

    p = getattr(core_log.setup_logging, "_path", None)
    if p:
        return Path(p)
    try:
        from ...core.config import get_settings

        return get_settings().logs_dir / "vrphoto.log"
    except Exception:
        from ...core.config import data_root

        return data_root() / "logs" / "vrphoto.log"


class LogDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, *, path: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Log")
        self.resize(900, 600)
        self.path = Path(path) if path else default_log_path()
        self.info = QLabel(f"Log file: {self.path}")
        self.info.setObjectName("Muted")
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.auto = QCheckBox("Refresh automatically")
        self.auto.setChecked(True)
        self.auto.toggled.connect(self._auto_toggled)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        copy = QPushButton("Copy all")
        copy.setToolTip("Copy the log text, e.g. to paste it into a problem report.")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.view.toPlainText()))
        folder = QPushButton("Open log folder")
        folder.clicked.connect(self.open_folder)
        row = QHBoxLayout()
        row.addWidget(self.auto)
        row.addStretch(1)
        row.addWidget(refresh)
        row.addWidget(copy)
        row.addWidget(folder)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addWidget(self.view, 1)
        lay.addLayout(row)
        lay.addWidget(box)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh)
        self.refresh()
        self.timer.start()

    def refresh(self) -> None:
        text = dc.tail_text(self.path, max_bytes=512_000, max_lines=3000)
        if not text:
            text = "(The log is empty or could not be read.)"
        if text != self.view.toPlainText():
            sb = self.view.verticalScrollBar()
            at_end = sb.value() >= sb.maximum() - 4
            self.view.setPlainText(text)
            if at_end:
                sb.setValue(sb.maximum())

    def _auto_toggled(self, on: bool) -> None:
        if on:
            self.timer.start()
        else:
            self.timer.stop()

    def open_folder(self) -> None:
        folder = self.path.parent
        if not util.open_path(folder):
            util.show_error(self, "Log", f"Could not open {folder}")

    def done(self, result: int) -> None:  # noqa: D401 - QDialog API
        self.timer.stop()
        super().done(result)
