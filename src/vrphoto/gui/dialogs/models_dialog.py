"""AI models dialog: see which models are downloaded, download (with progress and
cancel) or delete them.  Non-commercial models ask for license acceptance first."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.log import get_logger
from .. import util
from ..workers import DIALOG_CLOSE_WAIT_MS, TaskRunner, Worker
from . import common as dc

log = get_logger(__name__)

TASK_TITLES = {
    "depth": "Depth - how far away things are",
    "inpaint": "Look behind - painting hidden areas",
    "segment": "Cut-outs - separating people and objects",
    "geometry": "Geometry",
}
COL_TITLE, COL_SIZE, COL_LICENSE, COL_STATUS, COL_ACTIONS = range(5)


class ModelsDialog(QDialog):
    """Lists :func:`vrphoto.models.registry.list_specs` grouped by task."""

    def __init__(self, parent: QWidget | None = None, *, settings: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI models")
        self.resize(980, 600)
        if settings is None:
            from ...core.config import get_settings

            settings = get_settings()
        self.settings = settings
        self.tasks = TaskRunner(self, max_threads=1)
        self.worker: Worker | None = None
        self.active_id: str | None = None
        self.buttons: dict[str, tuple[QPushButton, QPushButton]] = {}
        self.items: dict[str, QTreeWidgetItem] = {}

        intro = QLabel(
            "VR Photo Studio downloads AI models the first time they are needed. Here you can "
            "download them in advance (for example before going offline) or delete them to free "
            "disk space - they will simply be downloaded again when needed. Models marked "
            "NON-COMMERCIAL may only be used for private, non-commercial projects."
        )
        intro.setWordWrap(True)
        intro.setObjectName("Muted")
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Model", "Size", "License", "Status", ""])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(False)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(COL_TITLE, QHeaderView.ResizeMode.Stretch)
        for c in (COL_SIZE, COL_LICENSE, COL_STATUS, COL_ACTIONS):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("Muted")
        self.cancel_button = QPushButton("Cancel download")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel_download)
        prow = QHBoxLayout()
        prow.addWidget(self.progress, 1)
        prow.addWidget(self.cancel_button)

        self.folder_button = QPushButton("Open models folder")
        self.folder_button.clicked.connect(self._open_folder)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(self.folder_button)
        bottom.addStretch(1)
        bottom.addWidget(box)

        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        self.registry, err = dc.try_backend("models.registry")
        self.download_mod, err2 = dc.try_backend("models.download")
        if self.registry is None or self.download_mod is None:
            lay.addWidget(dc.unavailable_label("The model manager", err or err2 or "unknown error"))
            self.tree.setVisible(False)
        lay.addWidget(self.tree, 1)
        lay.addWidget(self.progress_label)
        lay.addLayout(prow)
        lay.addLayout(bottom)
        self.populate()

    # ---------------------------------------------------------------- listing
    def specs(self) -> list[Any]:
        if self.registry is None:
            return []
        try:
            return list(self.registry.list_specs())
        except Exception as e:
            log.warning("Cannot list models: %s", e)
            return []

    def is_downloaded(self, model_id: str) -> bool:
        try:
            return bool(self.download_mod.is_downloaded(model_id)) if self.download_mod else False
        except Exception:
            return False

    def populate(self) -> None:
        self.tree.clear()
        self.buttons.clear()
        self.items.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for spec in self.specs():
            if getattr(spec, "experimental", False) and not getattr(
                self.settings, "show_experimental", False
            ):
                continue
            task = getattr(spec, "task", "other")
            parent = groups.get(task)
            if parent is None:
                parent = QTreeWidgetItem([TASK_TITLES.get(task, task.title())])
                f = parent.font(0)
                f.setBold(True)
                parent.setFont(0, f)
                self.tree.addTopLevelItem(parent)
                parent.setFirstColumnSpanned(True)
                groups[task] = parent
            item = QTreeWidgetItem(parent)
            title = spec.title + ("  (experimental)" if getattr(spec, "experimental", False) else "")
            item.setText(COL_TITLE, title)
            item.setToolTip(COL_TITLE, (spec.notes or spec.title) + f"\nID: {spec.id}")
            item.setText(COL_SIZE, util.human_size_mb(getattr(spec, "size_mb", 0)))
            item.setData(COL_TITLE, Qt.ItemDataRole.UserRole, spec.id)
            self.items[spec.id] = item
            lic = QWidget()
            ll = QHBoxLayout(lic)
            ll.setContentsMargins(2, 2, 2, 2)
            ll.addWidget(QLabel(spec.license or "?"))
            if spec.noncommercial:
                b = dc.badge("NON-COMMERCIAL", "nc")
                b.setToolTip("Free for private use only - not for paid work or products.")
                ll.addWidget(b)
            ll.addStretch(1)
            self.tree.setItemWidget(item, COL_LICENSE, lic)
            actions = QWidget()
            al = QHBoxLayout(actions)
            al.setContentsMargins(2, 2, 2, 2)
            dl = QPushButton("Download")
            dl.setToolTip(f"Download {spec.title} now.")
            rm = QPushButton("Delete")
            rm.setToolTip(f"Delete {spec.title} from this PC (it is downloaded again when needed).")
            dl.clicked.connect(lambda _=False, mid=spec.id: self.download(mid))
            rm.clicked.connect(lambda _=False, mid=spec.id: self.delete(mid))
            al.addWidget(dl)
            al.addWidget(rm)
            self.tree.setItemWidget(item, COL_ACTIONS, actions)
            self.buttons[spec.id] = (dl, rm)
            self._update_row(spec.id)
        self.tree.expandAll()

    def _update_row(self, model_id: str) -> None:
        item = self.items.get(model_id)
        if item is None:
            return
        done = self.is_downloaded(model_id)
        busy = self.active_id is not None
        item.setText(
            COL_STATUS,
            "Downloading…" if model_id == self.active_id else ("Downloaded" if done else "Not downloaded"),
        )
        dl, rm = self.buttons[model_id]
        dl.setEnabled(not done and not busy)
        rm.setEnabled(done and not busy)

    def _update_all(self) -> None:
        for mid in self.items:
            self._update_row(mid)

    # ---------------------------------------------------------------- actions
    def _spec(self, model_id: str) -> Any:
        return self.registry.get_spec(model_id) if self.registry else None

    def download(self, model_id: str) -> bool:
        """Start downloading ``model_id`` (asks for license acceptance if needed)."""
        if self.active_id is not None or self.download_mod is None:
            return False
        spec = self._spec(model_id)
        if spec is None:
            return False
        if spec.noncommercial and not self.settings.license_accepted(model_id):
            url = getattr(spec, "page_url", "") or ""
            if not dc.confirm_license(self, spec.title, spec.license, url=url):
                return False
            self.settings.accept_license(model_id)
            dc.save_settings(self.settings)
        self.active_id = model_id
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_button.setVisible(True)
        self.progress_label.setText(f"Downloading {spec.title}…")
        token = getattr(self.settings, "hf_token", "") or None
        self.worker = self.tasks.start(
            self.download_mod.ensure_model,
            model_id,
            token=token,
            on_progress=self._on_progress,
            on_result=lambda _p: self._finish(model_id, None),
            on_error=lambda e: self._finish(model_id, e),
            on_cancelled=lambda: self._finish(model_id, "cancel"),
            name=f"download-{model_id}",
        )
        self._update_all()
        return True

    def _on_progress(self, frac: float, msg: str) -> None:
        self.progress.setValue(int(round(frac * 1000)))
        if msg:
            self.progress_label.setText(msg)

    def _finish(self, model_id: str, error: Any) -> None:
        self.active_id = None
        self.worker = None
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        if error is None:
            self.progress_label.setText("Download complete.")
            log.info("Model %s downloaded", model_id)
        elif error == "cancel":
            self.progress_label.setText("Download cancelled. It will resume where it stopped next time.")
        else:
            msg, hint = util.error_text(error)
            self.progress_label.setText(f"Download failed: {msg}")
            util.show_error(self, "Download failed", error)
        self._update_all()

    def cancel_download(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.progress_label.setText("Cancelling…")

    def delete(self, model_id: str) -> bool:
        if self.download_mod is None or self.active_id is not None:
            return False
        spec = self._spec(model_id)
        title = spec.title if spec else model_id
        if not util.ask_yes_no(
            self,
            "Delete model",
            f"Delete {title} from this PC?",
            informative="It will be downloaded again automatically when needed.",
        ):
            return False
        try:
            self.download_mod.delete_model(model_id)
        except Exception as e:
            util.show_error(self, "Delete model", e)
            return False
        self._update_all()
        return True

    def _open_folder(self) -> None:
        util.open_path(self.settings.models_dir)

    # ---------------------------------------------------------------- closing
    def busy(self) -> bool:
        return self.active_id is not None

    def done(self, result: int) -> None:  # noqa: D401 - QDialog API
        if self.busy():
            if not util.ask_yes_no(
                self,
                "Download in progress",
                "Stop the download and close?",
                informative="The download resumes where it stopped next time.",
            ):
                return
            self.cancel_download()
        self.tasks.shutdown(DIALOG_CLOSE_WAIT_MS)
        super().done(result)
