"""Left panel: the list of input photos/videos (drag & drop, thumbnails, add/remove/clear)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, QSize, Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.log import get_logger
from ..util import collect_media_ex, file_dialog_filter, is_video_path, np_to_pixmap
from ..workers import TaskRunner

log = get_logger(__name__)

THUMB = 72
PATH_ROLE = Qt.ItemDataRole.UserRole
KEY_ROLE = Qt.ItemDataRole.UserRole + 1  # normalised path (duplicate detection)
MAX_FILES = 5000


def path_key(p: Path) -> str:
    """Normalised identity of a file (case-insensitive on Windows, symlinks resolved)."""
    try:
        return os.path.normcase(str(p.resolve()))
    except OSError:
        return os.path.normcase(str(p.absolute()))


def urls_to_paths(mime) -> list[Path]:  # noqa: ANN001 - QMimeData
    """Local file paths from a drag & drop mime payload."""
    out: list[Path] = []
    if mime is not None and mime.hasUrls():
        for url in mime.urls():
            if url.isLocalFile():
                out.append(Path(url.toLocalFile()))
    return out


class _DropList(QListWidget):
    """QListWidget that accepts dropped files and paints a hint when empty."""

    filesDropped = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setIconSize(QSize(THUMB, THUMB))
        self.setUniformItemSizes(True)
        self.setSpacing(2)
        self.setWordWrap(True)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:  # noqa: N802, ANN001
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e: QDropEvent) -> None:  # noqa: N802
        paths = urls_to_paths(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self.filesDropped.emit(paths)
        else:
            super().dropEvent(e)

    def paintEvent(self, e) -> None:  # noqa: N802, ANN001
        super().paintEvent(e)
        if self.count() == 0:
            p = QPainter(self.viewport())
            p.setPen(self.palette().placeholderText().color())
            r = self.viewport().rect().adjusted(12, 12, -12, -12)
            p.drawText(
                r,
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                "Drag photos or videos here\n(files or whole folders)\n\nor click 'Add photos'",
            )
            p.end()


def _placeholder_icon(video: bool) -> QIcon:
    pm = QPixmap(THUMB, THUMB)
    pm.fill(QColor("#2a2f3a"))
    p = QPainter(pm)
    p.setPen(QColor("#9aa3b5"))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "VIDEO" if video else "…")
    p.end()
    return QIcon(pm)


class InputListPanel(QWidget):
    """The input list.  API used by drag & drop: :meth:`add_paths`.

    Signals:
        currentChanged(object): the current (highlighted) path or None.
        pathsChanged(): items were added or removed.
        selectionChanged(): the selected set changed.
        notice(str): a message for the status bar (e.g. a folder was too big).
    """

    currentChanged = Signal(object)
    pathsChanged = Signal()
    selectionChanged = Signal()
    notice = Signal(str)

    def __init__(self, parent: QWidget | None = None, *, last_dir: str = "") -> None:
        super().__init__(parent)
        self.last_dir = last_dir
        self.tasks = TaskRunner(self, max_threads=2)
        self._items: dict[str, QListWidgetItem] = {}  # path_key -> item
        self.last_truncated = False  # the last add_paths() stopped early (see ``notice``)
        title = QLabel("Your photos and videos")
        title.setObjectName("SectionTitle")
        self.count_label = QLabel("")
        self.count_label.setObjectName("Muted")
        self.list = _DropList()
        self.list.filesDropped.connect(self.add_paths)
        self.list.currentItemChanged.connect(self._current_changed)
        self.list.itemSelectionChanged.connect(self.selectionChanged)

        self.add_button = QPushButton("Add photos…")
        self.add_button.setObjectName("PrimaryButton")
        self.add_button.setToolTip("Choose photos or videos to work with (you can also drag them here).")
        self.add_button.clicked.connect(self.browse_files)
        self.folder_button = QPushButton("Add folder…")
        self.folder_button.setToolTip("Add every photo and video in a folder (and its sub-folders).")
        self.folder_button.clicked.connect(self.browse_folder)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove the selected items from the list (the files are not deleted).")
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Remove everything from the list (the files are not deleted).")
        self.clear_button.clicked.connect(self.clear)

        top = QHBoxLayout()
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.count_label)
        row1 = QHBoxLayout()
        row1.addWidget(self.add_button, 1)
        row1.addWidget(self.folder_button)
        row2 = QHBoxLayout()
        row2.addWidget(self.remove_button)
        row2.addWidget(self.clear_button)
        row2.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(top)
        lay.addLayout(row1)
        lay.addWidget(self.list, 1)
        lay.addLayout(row2)
        self._update_buttons()

    # ---------------------------------------------------------------- queries
    def paths(self) -> list[Path]:
        return [Path(self.list.item(i).data(PATH_ROLE)) for i in range(self.list.count())]

    def selected_paths(self) -> list[Path]:
        """Selected paths in list order (all paths if nothing is selected)."""
        sel = [
            Path(self.list.item(i).data(PATH_ROLE))
            for i in range(self.list.count())
            if self.list.item(i).isSelected()
        ]
        return sel or self.paths()

    def current_path(self) -> Path | None:
        it = self.list.currentItem()
        return Path(it.data(PATH_ROLE)) if it is not None else None

    def count(self) -> int:
        return self.list.count()

    # ---------------------------------------------------------------- mutations
    def add_paths(self, paths: Iterable[str | Path]) -> int:
        """Add files and folders (expanded recursively); returns how many were added.

        Unsupported files and duplicates are skipped.  The added items become the selection
        (what "Create" works on) and the first of them is shown in the preview.
        """
        room = MAX_FILES - self.list.count()
        self.last_truncated = room <= 0
        if room <= 0:
            self.notice.emit(f"The list is full ({MAX_FILES} files). Remove some files first.")
            return 0
        found, truncated = collect_media_ex(paths, max_files=room)
        self.last_truncated = truncated
        added = 0
        first: QListWidgetItem | None = None
        new_items: list[QListWidgetItem] = []
        for p in found:
            key = path_key(p)
            if key in self._items:
                continue
            item = QListWidgetItem(_placeholder_icon(is_video_path(p)), self._label(p))
            item.setData(PATH_ROLE, str(p))
            item.setData(KEY_ROLE, key)
            item.setToolTip(str(p))
            self.list.addItem(item)
            self._items[key] = item
            self._load_thumb(key, p)
            first = first or item
            new_items.append(item)
            added += 1
        if truncated:
            self.notice.emit(
                f"Stopped after {len(found)} files: the list holds at most {MAX_FILES} photos/videos and "
                "very large folders are only partly searched. Add smaller folders instead."
            )
        if added and first is not None:
            # select what was just added, so "Create" processes it; preview the first one
            self.list.blockSignals(True)
            self.list.clearSelection()
            for it in new_items:
                it.setSelected(True)
            self.list.blockSignals(False)
            self.list.setCurrentItem(first, QItemSelectionModel.SelectionFlag.NoUpdate)
            self.pathsChanged.emit()
            self.selectionChanged.emit()
        self._update_buttons()
        return added

    def remove_selected(self) -> None:
        rows = sorted((self.list.row(i) for i in self.list.selectedItems()), reverse=True)
        if not rows and self.list.currentItem() is not None:
            rows = [self.list.currentRow()]
        for r in rows:
            item = self.list.takeItem(r)
            if item is not None:
                self._items.pop(str(item.data(KEY_ROLE)), None)
        if rows:
            self.pathsChanged.emit()
            if self.list.count() == 0:
                self.currentChanged.emit(None)
        self._update_buttons()

    def clear(self) -> None:
        if self.list.count():
            self._items.clear()
            self.list.clear()
            self.pathsChanged.emit()
            self.currentChanged.emit(None)
        self._update_buttons()

    def select_all(self) -> None:
        self.list.selectAll()

    # ---------------------------------------------------------------- dialogs
    def browse_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add photos or videos", self.last_dir, file_dialog_filter()
        )
        if files:
            self.last_dir = str(Path(files[0]).parent)
            n = self.add_paths(files)
            log.info("Added %d file(s)", n)

    def browse_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Add a folder of photos", self.last_dir)
        if d:
            self.last_dir = d
            n = self.add_paths([d])
            log.info("Added %d file(s) from %s", n, d)

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _label(p: Path) -> str:
        try:
            n = p.stat().st_size
            size = f"{n / 1e3:.0f} KB" if n < 1e6 else f"{n / 1e6:.1f} MB" if n < 1e8 else f"{n / 1e6:.0f} MB"
        except OSError:
            size = "?"
        kind = "Video" if is_video_path(p) else "Photo"
        return f"{p.name}\n{kind} · {size}"

    def _load_thumb(self, key: str, p: Path) -> None:
        from ..util import load_thumbnail

        def done(img) -> None:  # noqa: ANN001
            it = self._items.get(key)  # None if the item was removed meanwhile
            if it is not None:
                it.setIcon(QIcon(np_to_pixmap(img)))

        def failed(exc: BaseException) -> None:
            log.debug("No thumbnail for %s: %s", p, exc)

        self.tasks.start(
            load_thumbnail,
            p,
            THUMB,
            on_result=done,
            on_error=failed,
            pass_progress=False,
            pass_cancel=False,
            name="thumbnail",
        )

    def _current_changed(self, cur: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        self.currentChanged.emit(Path(cur.data(PATH_ROLE)) if cur is not None else None)
        self._update_buttons()

    def _update_buttons(self) -> None:
        n = self.list.count()
        self.remove_button.setEnabled(n > 0)
        self.clear_button.setEnabled(n > 0)
        self.count_label.setText(f"{n} item{'s' if n != 1 else ''}" if n else "")

    def shutdown(self) -> None:
        self.tasks.shutdown(2000)
