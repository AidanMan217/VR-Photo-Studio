"""Bottom panel: live log messages (thread-safe logging handler -> Qt signal -> text view)."""

from __future__ import annotations

import html
import logging

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from ..theme import current_colors

LOGGER_NAME = "vrphoto"


class _Emitter(QObject):
    record = Signal(int, str)


class QtLogHandler(logging.Handler):
    """Logging handler that forwards formatted records to a Qt signal (any thread)."""

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.emitter = _Emitter()
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.emitter.record.emit(record.levelno, msg)
        except RuntimeError:  # emitter already deleted during shutdown
            pass
        except Exception:
            self.handleError(record)


class LogPanel(QWidget):
    """Shows log messages from the ``vrphoto`` loggers (INFO and above by default)."""

    def __init__(self, parent: QWidget | None = None, *, level: int = logging.INFO, max_lines: int = 3000):
        super().__init__(parent)
        title = QLabel("Log")
        title.setObjectName("SectionTitle")
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(max_lines)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        clear = QPushButton("Clear")
        clear.setToolTip("Clear the messages shown here (the log file is kept).")
        clear.clicked.connect(self.view.clear)
        head = QHBoxLayout()
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(clear)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(head)
        lay.addWidget(self.view, 1)

        self.handler = QtLogHandler(level)
        self.handler.emitter.record.connect(self.append)
        self._logger = logging.getLogger(LOGGER_NAME)
        self._logger.addHandler(self.handler)
        if self._logger.getEffectiveLevel() > level:
            self._logger.setLevel(level)
        handler, logger = self.handler, self._logger
        # never reference ``self`` here: it is being destroyed when this runs
        self.destroyed.connect(lambda *_: logger.removeHandler(handler))

    def append(self, level: int, text: str) -> None:
        c = current_colors()
        color = c.danger if level >= logging.ERROR else c.warning if level >= logging.WARNING else None
        esc = html.escape(text)
        line = f'<span style="color:{color}">{esc}</span>' if color else esc
        self.view.appendHtml(line)
        self.view.moveCursor(QTextCursor.MoveOperation.End)

    def text(self) -> str:
        return self.view.toPlainText()

    def detach(self) -> None:
        """Remove the logging handler (call before the widget is destroyed)."""
        if self.handler in self._logger.handlers:
            self._logger.removeHandler(self.handler)
