"""Bottom panel: the job queue (one row per job with progress, cancel, open folder and
"View in headset")."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core.log import get_logger
from .. import options_meta as om
from .. import util
from ..bridge import QueueBridge
from ..util import human_duration, output_folder_of

log = get_logger(__name__)

STATUS_TEXT = {
    "queued": "Waiting",
    "running": "Working",
    "finished": "Done",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


@dataclass
class JobInfo:
    job_id: str
    kind: str
    title: str
    inputs: list[str]
    output_dir: str | None = None
    status: str = "queued"
    frac: float = 0.0
    msg: str = ""
    result: Any = None
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    outputs: dict[str, str] = field(default_factory=dict)

    @property
    def scene_id(self) -> str | None:
        return getattr(self.result, "scene_id", None) if self.result is not None else None


class ElidedLabel(QLabel):
    """One-line label that shortens long text with "…" to fit (the full text is in the
    tooltip).  Keeps job rows a fixed height however long an error message is."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self._tip = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(40)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full = str(text or "")
        self._elide()

    def text(self) -> str:
        """The full (not shortened) text."""
        return self._full

    def setToolTip(self, tip: str) -> None:  # noqa: N802 - Qt API
        self._tip = str(tip or "")
        self._elide()

    def resizeEvent(self, e) -> None:  # noqa: N802, ANN001
        super().resizeEvent(e)
        self._elide()

    def _elide(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, max(20, self.width()))
        super().setText(shown)
        tip = self._tip or (self._full if shown != self._full else "")
        super().setToolTip(tip)


class JobRow(QFrame):
    """Widget for one job in the queue list."""

    cancelRequested = Signal(str)
    openRequested = Signal(str)
    viewRequested = Signal(str)
    errorRequested = Signal(str)

    def __init__(self, info: JobInfo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.info = info
        self.title = QLabel(info.title)
        self.title.setObjectName("ModeTitle")
        self.title.setToolTip("\n".join(info.inputs))
        self.status = ElidedLabel("Waiting")
        self.status.setObjectName("Muted")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(True)
        self.bar.setFormat("%p%")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Stop this job.")
        self.open_button = QPushButton("Open folder")
        self.open_button.setToolTip("Show the result files in Explorer.")
        self.view_button = QPushButton("View in headset")
        self.view_button.setToolTip("Open 'Send to Quest' to see this result in your Quest Browser.")
        self.error_button = QPushButton("Details")
        self.error_button.setToolTip("Show what went wrong.")
        self.cancel_button.clicked.connect(lambda: self.cancelRequested.emit(self.info.job_id))
        self.open_button.clicked.connect(lambda: self.openRequested.emit(self.info.job_id))
        self.view_button.clicked.connect(lambda: self.viewRequested.emit(self.info.job_id))
        self.error_button.clicked.connect(lambda: self.errorRequested.emit(self.info.job_id))

        top = QHBoxLayout()
        top.addWidget(self.title, 1)
        top.addWidget(self.status, 2)
        buttons = QHBoxLayout()
        buttons.addWidget(self.bar, 1)
        for b in (self.cancel_button, self.error_button, self.open_button, self.view_button):
            buttons.addWidget(b)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        lay.addLayout(top)
        lay.addLayout(buttons)
        self.refresh()

    def refresh(self) -> None:
        i = self.info
        st = i.status
        text = STATUS_TEXT.get(st, st)
        if st == "running":
            text = i.msg or "Working…"
        elif st == "finished":
            dur = (i.finished - i.started) if (i.finished and i.started) else None
            text = "Done" + (f" in {human_duration(dur)}" if dur else "")
            warnings = list(getattr(i.result, "warnings", []) or [])
            if warnings:
                text += f" ({len(warnings)} note{'s' if len(warnings) != 1 else ''})"
                self.status.setToolTip("\n".join(warnings))
        elif st == "failed":
            text = f"Failed: {i.error or 'unknown error'}"
            self.status.setToolTip(i.error or "")
        self.status.setText(text)
        self.status.setObjectName({"failed": "Error", "finished": "Success"}.get(st, "Muted"))
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        if st == "finished":
            self.bar.setValue(1000)
        elif st == "queued":
            self.bar.setValue(0)
        else:
            self.bar.setValue(int(round(max(0.0, min(1.0, i.frac)) * 1000)))
        self.bar.setFormat(
            {"finished": "Done", "failed": "Failed", "cancelled": "Cancelled", "queued": "Waiting"}.get(
                st, "%p%"
            )
        )
        self.bar.setProperty("state", {"failed": "failed", "finished": "done"}.get(st, ""))
        self.bar.style().unpolish(self.bar)
        self.bar.style().polish(self.bar)
        active = st in ("queued", "running")
        self.cancel_button.setVisible(active)
        self.error_button.setVisible(st == "failed")
        self.open_button.setVisible(st == "finished")
        self.view_button.setVisible(st == "finished" and bool(i.scene_id))


class QueuePanel(QWidget):
    """List of jobs bound to a :class:`QueueBridge`.

    Signals:
        viewInHeadset(str, object): job id, scene id - "View in headset" was clicked.
        jobDone(str): a job finished successfully.
    """

    viewInHeadset = Signal(str, object)
    jobDone = Signal(str)

    def __init__(self, bridge: QueueBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self.jobs: dict[str, JobInfo] = {}
        self.rows: dict[str, JobRow] = {}
        self.items: dict[str, QListWidgetItem] = {}
        title = QLabel("Jobs")
        title.setObjectName("SectionTitle")
        self.summary = QLabel("No jobs yet.")
        self.summary.setObjectName("Muted")
        self.clear_button = QPushButton("Clear finished")
        self.clear_button.setToolTip("Remove finished, failed and cancelled jobs from the list.")
        self.clear_button.clicked.connect(self.clear_finished)
        self.cancel_all_button = QPushButton("Cancel all")
        self.cancel_all_button.setObjectName("DangerButton")
        self.cancel_all_button.setToolTip("Stop the running job and remove all waiting jobs.")
        self.cancel_all_button.clicked.connect(self.cancel_all)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        head = QHBoxLayout()
        head.addWidget(title)
        head.addWidget(self.summary, 1)
        head.addWidget(self.clear_button)
        head.addWidget(self.cancel_all_button)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(head)
        lay.addWidget(self.list, 1)

        bridge.jobQueued.connect(self._on_queued)
        bridge.jobStarted.connect(self._on_started)
        bridge.jobProgress.connect(self._on_progress)
        bridge.jobFinished.connect(self._on_finished)
        bridge.jobFailed.connect(self._on_failed)
        bridge.jobCancelled.connect(self._on_cancelled)
        self._update_summary()

    # ---------------------------------------------------------------- adding jobs
    def add_job(self, job_id: str, spec: Any) -> JobInfo:
        """Register a submitted job (call right after ``bridge.submit``)."""
        if job_id in self.jobs:
            return self.jobs[job_id]
        mode = om.MODE_BY_KIND.get(getattr(spec, "kind", ""), None)
        inputs = [str(p) for p in getattr(spec, "inputs", [])]
        names = [Path(p).name for p in inputs]
        what = names[0] if len(names) == 1 else f"{len(names)} files"
        title = f"{mode.title if mode else spec.kind}: {what}"
        common = getattr(spec, "common", None)
        info = JobInfo(job_id, spec.kind, title, inputs, output_dir=getattr(common, "output_dir", None))
        self.jobs[job_id] = info
        row = JobRow(info)
        row.cancelRequested.connect(self.cancel)
        row.openRequested.connect(self.open_folder)
        row.viewRequested.connect(self._view)
        row.errorRequested.connect(self._show_error)
        item = QListWidgetItem()
        item.setSizeHint(QSize(100, row.sizeHint().height()))
        item.setData(Qt.ItemDataRole.UserRole, job_id)
        self.list.addItem(item)
        self.list.setItemWidget(item, row)
        self.rows[job_id] = row
        self.items[job_id] = item
        self.list.scrollToItem(item)
        self._update_summary()
        return info

    def _row(self, job_id: str) -> JobRow | None:
        if job_id not in self.jobs:
            # a job submitted elsewhere (e.g. CLI in-process) - show it anyway
            try:
                rec = self.bridge.get(job_id)
                self.add_job(job_id, rec.spec)
            except Exception:
                return None
        return self.rows.get(job_id)

    # ---------------------------------------------------------------- bridge events
    def _on_queued(self, job_id: str) -> None:
        row = self._row(job_id)
        if row and row.info.status == "queued":
            row.refresh()
        self._update_summary()

    def _on_started(self, job_id: str) -> None:
        row = self._row(job_id)
        if row:
            row.info.status = "running"
            row.info.started = time.time()
            row.refresh()
        self._update_summary()

    def _on_progress(self, job_id: str, frac: float, msg: str) -> None:
        row = self._row(job_id)
        if row:
            if row.info.status == "queued":
                row.info.started = row.info.started or time.time()
            row.info.status = "running"
            row.info.frac = frac
            if msg:
                row.info.msg = msg
            row.refresh()

    def _on_finished(self, job_id: str, result: Any) -> None:
        row = self._row(job_id)
        if row:
            i = row.info
            i.status = "finished"
            i.result = result
            i.frac = 1.0
            i.finished = time.time()
            i.started = i.started or i.finished
            i.outputs = dict(getattr(result, "outputs", {}) or {})
            row.refresh()
            log.info("Job finished: %s", i.title)
        self._update_summary()
        self.jobDone.emit(job_id)

    def _on_failed(self, job_id: str, error: str) -> None:
        row = self._row(job_id)
        if row:
            row.info.status = "failed"
            row.info.error = error
            row.info.finished = time.time()
            row.refresh()
            log.error("Job failed: %s - %s", row.info.title, error)
        self._update_summary()

    def _on_cancelled(self, job_id: str) -> None:
        row = self._row(job_id)
        if row:
            row.info.status = "cancelled"
            row.info.finished = time.time()
            row.refresh()
        self._update_summary()

    # ---------------------------------------------------------------- actions
    def cancel(self, job_id: str) -> None:
        self.bridge.cancel(job_id)
        row = self.rows.get(job_id)
        if row and row.info.status in ("queued", "running"):
            row.info.msg = "Cancelling…"
            row.status.setText("Cancelling…")
            row.cancel_button.setEnabled(False)

    def cancel_all(self) -> None:
        self.bridge.cancel_all()
        for row in self.rows.values():
            if row.info.status in ("queued", "running"):
                row.status.setText("Cancelling…")
                row.cancel_button.setEnabled(False)

    def clear_finished(self) -> None:
        self.bridge.clear_finished()
        for job_id in [j for j, i in self.jobs.items() if i.status in ("finished", "failed", "cancelled")]:
            item = self.items.pop(job_id)
            self.list.takeItem(self.list.row(item))
            self.rows.pop(job_id, None)
            self.jobs.pop(job_id, None)
        self._update_summary()

    def output_folder(self, job_id: str) -> Path | None:
        info = self.jobs.get(job_id)
        if info is None:
            return None
        fallback = info.output_dir
        if not fallback:
            try:
                from ...core.config import get_settings

                fallback = str(get_settings().outputs_dir)
            except Exception:
                fallback = None
        return output_folder_of(info.outputs, fallback)

    def output_files(self, job_id: str) -> list[Path]:
        info = self.jobs.get(job_id)
        if info is None:
            return []
        out = []
        for v in info.outputs.values():
            p = Path(str(v))
            if p.is_file():
                out.append(p)
        return out

    def finished_jobs(self) -> list[JobInfo]:
        return [i for i in self.jobs.values() if i.status == "finished"]

    def selected_job(self) -> str | None:
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    def open_folder(self, job_id: str) -> None:
        folder = self.output_folder(job_id)
        if folder is None or not util.open_path(folder):
            util.show_error(self, "Open folder", "The output folder could not be opened.")

    def _view(self, job_id: str) -> None:
        info = self.jobs.get(job_id)
        if info is not None:
            self.viewInHeadset.emit(job_id, info.scene_id)

    def _show_error(self, job_id: str) -> None:
        info = self.jobs.get(job_id)
        if info is not None:
            util.show_error(
                self,
                "Job failed",
                info.error or "Unknown error",
                details="\n".join(["Inputs:", *info.inputs]),
            )

    def _update_summary(self) -> None:
        counts = {"queued": 0, "running": 0, "finished": 0, "failed": 0, "cancelled": 0}
        for i in self.jobs.values():
            counts[i.status] = counts.get(i.status, 0) + 1
        if not self.jobs:
            text = "No jobs yet. Choose a mode and click Create."
        else:
            parts = []
            if counts["running"]:
                parts.append(f"{counts['running']} working")
            if counts["queued"]:
                parts.append(f"{counts['queued']} waiting")
            if counts["finished"]:
                parts.append(f"{counts['finished']} done")
            if counts["failed"]:
                parts.append(f"{counts['failed']} failed")
            if counts["cancelled"]:
                parts.append(f"{counts['cancelled']} cancelled")
            text = ", ".join(parts)
        self.summary.setText(text)
        active = counts["running"] + counts["queued"]
        self.cancel_all_button.setEnabled(active > 0)
        self.clear_button.setEnabled(len(self.jobs) - active > 0)

    def active_count(self) -> int:
        return sum(1 for i in self.jobs.values() if i.status in ("queued", "running"))
