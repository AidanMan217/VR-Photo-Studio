"""Add-ons dialog: optional heavy AI tools (TRELLIS, Capture, SDXL outpaint...).

One card per add-on with its description, GPU memory / disk requirements compared with
this PC, license notes (non-commercial, region restrictions), Install with live progress
+ log tail + cancel, Uninstall and "Open folder".  The installed Capture add-on also gets
"Test Brush on this PC" (:func:`vrphoto.addons.capture.brush_selftest`); the result is
kept in ``Settings.brush_test``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...core.log import get_logger
from .. import util
from ..workers import DIALOG_CLOSE_WAIT_MS, TaskRunner, Worker
from . import common as dc

log = get_logger(__name__)

#: The add-on whose card offers the Brush self-test (Brush ships with Capture).
BRUSH_ADDON = "capture"


def brush_test_summary(result: dict[str, Any] | None) -> str:
    """One line about the last Brush test (tooltip of the test button)."""
    if not result:
        return "Not tested yet."
    when = str(result.get("date", "")).replace("T", " ")[:16]
    if result.get("ok"):
        rate = result.get("steps_per_s")
        speed = f", {float(rate):.1f} steps/s" if rate else ""
        return f"Last test {when}: Brush works{speed}."
    return f"Last test {when}: Brush could not train."


def detect_gpu() -> dict[str, Any]:
    """``{"name": str | None, "vram_mb": int | None, "nvidia": bool}`` (worker thread)."""
    info: dict[str, Any] = {"name": None, "vram_mb": None, "nvidia": False}
    try:
        from ...models import gpu

        devs = list(gpu.list_devices())
        best = max(devs, key=lambda d: (d.vendor == "nvidia", d.vram_mb or 0), default=None)
        if best is not None:
            info.update(name=best.name, vram_mb=best.vram_mb, nvidia=best.vendor == "nvidia")
        if info["vram_mb"] is None or not devs:
            vram = gpu.nvidia_vram_mb()
            if vram:
                info.update(vram_mb=vram, nvidia=True, name=info["name"] or "NVIDIA GPU")
    except Exception as e:
        log.debug("GPU detection failed: %s", e)
    return info


_HW_NOTICE_PREFIXES = ("Needs an NVIDIA GPU", "Needs a GPU", "Download/disk")


def license_notices(m: Any) -> list[str]:
    """The manifest's notices without the hardware ones (shown separately, compared with this PC)."""
    try:
        notices = list(m.notices())
    except Exception:
        return []
    return [n for n in notices if not n.startswith(_HW_NOTICE_PREFIXES)]


def requirement_notes(m: Any, gpu: dict[str, Any] | None, free_gb: float | None) -> list[tuple[str, str]]:
    """``[(level, text)]`` comparing an add-on's needs with this PC; level ok|warn|info.

    ``venv`` add-ons (PyTorch/CUDA) need an NVIDIA GPU; ``binary`` add-ons (e.g. Capture)
    run on any modern GPU.
    """
    out: list[tuple[str, str]] = []
    need = float(getattr(m, "min_vram_gb", 0) or 0)
    nvidia_only = getattr(m, "kind", "venv") != "binary"
    gpu_word = "an NVIDIA GPU" if nvidia_only else "a GPU"
    if need:
        if gpu is None:
            out.append(
                ("info", f"Needs {gpu_word} with at least {need:g} GB of video memory (checking your GPU…).")
            )
        elif nvidia_only and not gpu.get("nvidia"):
            out.append(
                (
                    "warn",
                    f"Needs an NVIDIA GPU with at least {need:g} GB - no NVIDIA GPU was found on this PC.",
                )
            )
        elif not gpu.get("name") and not gpu.get("vram_mb"):
            out.append(
                (
                    "info",
                    f"Needs {gpu_word} with at least {need:g} GB of video memory "
                    "(no graphics card could be detected).",
                )
            )
        elif gpu.get("vram_mb"):
            have = gpu["vram_mb"] / 1024
            if have + 0.25 >= need:
                out.append(("ok", f"GPU: {gpu.get('name')} has {have:.0f} GB (needs {need:g} GB)."))
            else:
                out.append(
                    (
                        "warn",
                        f"GPU: {gpu.get('name')} has {have:.0f} GB but this add-on needs {need:g} GB. "
                        "It may fail or be very slow.",
                    )
                )
        else:
            out.append(("info", f"Needs {need:g} GB of video memory (yours could not be measured)."))
    disk = float(getattr(m, "disk_gb", 0) or 0)
    if disk:
        if free_gb is None:
            out.append(("info", f"Needs about {disk:g} GB of disk space."))
        elif free_gb >= disk * 1.1:
            out.append(("ok", f"Disk: needs about {disk:g} GB, {free_gb:.0f} GB free."))
        else:
            out.append(("warn", f"Disk: needs about {disk:g} GB but only {free_gb:.0f} GB is free."))
    return out


class AddonCard(QFrame):
    """UI for one add-on."""

    installRequested = Signal(str)
    uninstallRequested = Signal(str)
    cancelRequested = Signal(str)
    folderRequested = Signal(str)
    testRequested = Signal(str)

    def __init__(self, manifest: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.m = manifest
        self.status: dict[str, Any] = {}
        head = QHBoxLayout()
        title = QLabel(manifest.title)
        title.setObjectName("SectionTitle")
        head.addWidget(title)
        if getattr(manifest, "experimental", False):
            head.addWidget(dc.badge("EXPERIMENTAL", "exp"))
        if getattr(manifest, "noncommercial", False):
            head.addWidget(dc.badge("NON-COMMERCIAL", "nc"))
        if getattr(manifest, "region_restricted", None):
            head.addWidget(dc.badge("REGION LIMITS", "nc"))
        head.addStretch(1)
        self.state_label = QLabel("")
        head.addWidget(self.state_label)

        desc = QLabel(manifest.description)
        desc.setWordWrap(True)
        self.req_label = QLabel("")
        self.req_label.setWordWrap(True)
        self.req_label.setTextFormat(Qt.TextFormat.RichText)
        lic = f"License: {manifest.license or '?'}"
        if getattr(manifest, "homepage", ""):
            lic += f' - <a href="{manifest.homepage}">project page</a>'
        self.license_label = QLabel(lic)
        self.license_label.setOpenExternalLinks(True)
        self.license_label.setObjectName("Muted")
        self.license_label.setWordWrap(True)
        notes = license_notices(manifest)
        self.notes_label = QLabel("<br>".join(util.rich_escape(n) for n in notes))
        self.notes_label.setWordWrap(True)
        self.notes_label.setObjectName("Warning")
        self.notes_label.setVisible(bool(notes))
        details = str(getattr(manifest, "notes", "") or "").strip()
        self.details_label = QLabel(details)
        self.details_label.setWordWrap(True)
        self.details_label.setObjectName("Muted")
        self.details_label.setVisible(bool(details))

        self.install_button = QPushButton("Install")
        self.install_button.setObjectName("PrimaryButton")
        self.uninstall_button = QPushButton("Uninstall")
        self.folder_button = QPushButton("Open folder")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        # only shown on the installed Capture card (see set_status)
        self.test_button = QPushButton("Test Brush on this PC")
        self.test_button.setVisible(False)
        self._test_tip = (
            "Trains Brush for a minute or two on a small test scene to check that it works with "
            "your graphics card, and how fast. SHARP 360 uses the result to decide whether its "
            "optional Brush polishing can be offered."
        )
        self.set_test_result(None)
        self.install_button.clicked.connect(lambda: self.installRequested.emit(self.m.id))
        self.uninstall_button.clicked.connect(lambda: self.uninstallRequested.emit(self.m.id))
        self.cancel_button.clicked.connect(lambda: self.cancelRequested.emit(self.m.id))
        self.folder_button.clicked.connect(lambda: self.folderRequested.emit(self.m.id))
        self.test_button.clicked.connect(lambda: self.testRequested.emit(self.m.id))
        btns = QHBoxLayout()
        btns.addWidget(self.install_button)
        btns.addWidget(self.uninstall_button)
        btns.addWidget(self.folder_button)
        btns.addWidget(self.test_button)
        btns.addStretch(1)
        btns.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("Muted")
        self.progress_label.setWordWrap(True)
        self.progress_label.setVisible(False)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(400)
        self.log_view.setFixedHeight(110)
        self.log_view.setVisible(False)

        lay = QVBoxLayout(self)
        lay.addLayout(head)
        lay.addWidget(desc)
        lay.addWidget(self.req_label)
        lay.addWidget(self.notes_label)
        lay.addWidget(self.details_label)
        lay.addWidget(self.license_label)
        lay.addLayout(btns)
        lay.addWidget(self.progress)
        lay.addWidget(self.progress_label)
        lay.addWidget(self.log_view)
        self.set_requirements(None, None)

    def set_requirements(self, gpu: dict[str, Any] | None, free_gb: float | None) -> None:
        colors = {"ok": "#3ecf8e", "warn": "#ffb84d", "info": "#9aa3b5"}
        rows = [
            f'<span style="color:{colors[lvl]}">{"✓" if lvl == "ok" else "!" if lvl == "warn" else "•"} '
            f"{util.rich_escape(t)}</span>"
            for lvl, t in requirement_notes(self.m, gpu, free_gb)
        ]
        self.req_label.setText("<br>".join(rows))
        self.req_label.setVisible(bool(rows))

    def set_checking(self) -> None:
        """Placeholder while the install status is being read (buttons disabled)."""
        self.state_label.setText("Checking…")
        for b in (self.install_button, self.uninstall_button, self.folder_button):
            b.setEnabled(False)

    def set_status(self, status: dict[str, Any], *, busy: bool = False, any_busy: bool = False) -> None:
        self.status = status
        installed = bool(status.get("installed"))
        supported = bool(status.get("supported", True))
        if busy:
            text = "Working…"
        elif installed:
            size = status.get("size_mb") or 0
            text = "Installed" + (f" ({util.human_size_mb(size)})" if size else "")
            if status.get("update_available"):
                text += " - update available"
        elif status.get("partial"):
            text = "Partly installed - click Install to resume"
        elif status.get("error"):
            text = "Not installed (last attempt failed)"
        else:
            text = "Not installed"
        self.state_label.setText(text)
        self.state_label.setObjectName("Success" if installed and not busy else "Muted")
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)
        upd = installed and status.get("update_available")
        self.install_button.setText("Update" if upd else "Install")
        self.install_button.setVisible(not installed or bool(upd))
        self.install_button.setEnabled(supported and not any_busy)
        self.uninstall_button.setVisible(installed or bool(status.get("partial")))
        self.uninstall_button.setEnabled(not any_busy)
        self.folder_button.setEnabled(Path(str(status.get("path", ""))).is_dir())
        self.test_button.setVisible(self.m.id == BRUSH_ADDON and installed)
        self.test_button.setEnabled(not any_busy)
        self.cancel_button.setVisible(busy)
        if not supported:
            self.install_button.setToolTip("Not available on this operating system.")

    def set_test_result(self, result: dict[str, Any] | None) -> None:
        self.test_button.setToolTip(f"{self._test_tip}\n\n{brush_test_summary(result)}")

    def begin(self, title: str) -> None:
        self.progress.setValue(0)
        for w in (self.progress, self.progress_label, self.log_view):
            w.setVisible(True)
        self.progress_label.setText(title)
        self.log_view.clear()

    def set_progress(self, frac: float, msg: str) -> None:
        self.progress.setValue(int(round(max(0.0, min(1.0, frac)) * 1000)))
        if msg:
            self.progress_label.setText(msg)

    def set_log(self, text: str) -> None:
        if text != self.log_view.toPlainText():
            self.log_view.setPlainText(text)
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def end(self, message: str) -> None:
        self.progress.setVisible(False)
        self.progress_label.setText(message)
        self.cancel_button.setVisible(False)


class AddonsDialog(QDialog):
    """Install / uninstall add-ons via :mod:`vrphoto.addons.manager`.

    The install status is read in a background thread (for a partly installed add-on the
    manager measures its folder, which can take seconds for a large Python environment).

    Signals:
        experimentalChanged(bool): "Show experimental add-ons" was toggled (and saved).
        statusRefreshed(): the cards show freshly read install states.
    """

    experimentalChanged = Signal(bool)
    statusRefreshed = Signal()

    def __init__(self, parent: QWidget | None = None, *, settings: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add-ons")
        self.resize(820, 680)
        if settings is None:
            from ...core.config import get_settings

            settings = get_settings()
        self.settings = settings
        self.tasks = TaskRunner(self, max_threads=3)  # install + status + GPU check
        self.cards: dict[str, AddonCard] = {}
        self._empty_label: QLabel | None = None
        self.worker: Worker | None = None
        self.active_id: str | None = None
        self.testing = False  # the running task is the Brush test (not an install)
        self.gpu: dict[str, Any] | None = None
        self.statuses: dict[str, dict[str, Any]] = {}
        self._status_gen = 0
        self._status_pending = False
        self.log_timer = QTimer(self)
        self.log_timer.setInterval(1000)
        self.log_timer.timeout.connect(self._refresh_log)

        intro = QLabel(
            "Add-ons are optional, large AI tools (several GB each). The app installs them into a "
            "private folder - nothing else on your PC changes - and you can uninstall them at any "
            "time. Most need an NVIDIA graphics card with an up-to-date driver."
        )
        intro.setWordWrap(True)
        intro.setObjectName("Muted")
        self.gpu_label = QLabel("Checking your graphics card…")
        self.exp_check = QCheckBox("Show experimental add-ons")
        self.exp_check.setToolTip("Experimental add-ons may fail to install or give poor results.")
        self.exp_check.setChecked(bool(getattr(settings, "show_experimental", False)))
        self.exp_check.toggled.connect(self._toggle_experimental)
        top = QHBoxLayout()
        top.addWidget(self.gpu_label, 1)
        top.addWidget(self.exp_check)

        self.cards_box = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_box)
        self.cards_layout.setContentsMargins(0, 0, 6, 0)
        self.cards_layout.addStretch(1)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.cards_box)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addLayout(top)
        self.manager, err = dc.try_backend("addons.manager")
        if self.manager is None:
            lay.addWidget(dc.unavailable_label("The add-on manager", err or "unknown error"))
        lay.addWidget(self.scroll, 1)
        lay.addWidget(box)
        self.populate()
        self.tasks.start(
            detect_gpu, pass_progress=False, pass_cancel=False, on_result=self._gpu_done, name="detect-gpu"
        )

    # ---------------------------------------------------------------- listing
    def manifests(self) -> list[Any]:
        if self.manager is None:
            return []
        try:
            return list(self.manager.list_addons(include_experimental=self.exp_check.isChecked()))
        except Exception as e:
            log.warning("Cannot list add-ons: %s", e)
            util.show_error(self, "Add-ons", e)
            return []

    def _status(self, addon_id: str) -> dict[str, Any]:
        """Install status of one add-on (may walk its folder: call from a worker thread)."""
        try:
            return dict(self.manager.status(addon_id))
        except Exception as e:
            log.debug("status(%s) failed: %s", addon_id, e)
            return {"installed": False, "error": str(e)}

    def _free_gb(self) -> float | None:
        try:
            d = Path(self.settings.addons_dir)
            return shutil.disk_usage(d).free / 1e9
        except Exception:
            return None

    def populate(self) -> None:
        for card in self.cards.values():
            card.setParent(None)
            card.deleteLater()
        self.cards.clear()
        free = self._free_gb()
        for m in self.manifests():
            card = AddonCard(m)
            card.installRequested.connect(self.install)
            card.uninstallRequested.connect(self.uninstall)
            card.cancelRequested.connect(self.cancel)
            card.folderRequested.connect(self.open_folder)
            card.testRequested.connect(self.test_brush)
            card.set_test_result(getattr(self.settings, "brush_test", None))
            card.set_requirements(self.gpu, free)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
            self.cards[m.id] = card
        if self._empty_label is not None:
            self._empty_label.setParent(None)
            self._empty_label.deleteLater()
            self._empty_label = None
        if self.manager is not None and not self.cards:
            self._empty_label = QLabel("No add-ons are available.")
            self._empty_label.setObjectName("Muted")
            self.cards_layout.insertWidget(0, self._empty_label)
        self.refresh_status()

    @property
    def status_pending(self) -> bool:
        """True while install states are being read in the background."""
        return self._status_pending

    def refresh_status(self) -> None:
        """Re-read every card's install status in the background, then update the cards."""
        if self.manager is None or not self.cards:
            self._status_pending = False
            return
        self._status_gen += 1
        gen = self._status_gen
        ids = list(self.cards)
        self._status_pending = True
        for aid, card in self.cards.items():
            if aid not in self.statuses:
                card.set_checking()
        self._apply_status()

        def work() -> dict[str, dict[str, Any]]:
            return {aid: self._status(aid) for aid in ids}

        def done(result: dict[str, dict[str, Any]]) -> None:
            if gen != self._status_gen:
                return  # a newer refresh is running
            self.statuses.update(result)
            self._status_pending = False
            self._apply_status()
            self.statusRefreshed.emit()

        def failed(exc: BaseException) -> None:
            if gen == self._status_gen:
                self._status_pending = False
                log.warning("Could not read the add-on states: %s", exc)

        self.tasks.start(
            work, pass_progress=False, pass_cancel=False, on_result=done, on_error=failed, name="addon-status"
        )

    def _apply_status(self) -> None:
        """Show the last known states with the current busy flags (UI thread, no I/O)."""
        for aid, card in self.cards.items():
            st = self.statuses.get(aid)
            if st is None:
                continue
            card.set_status(st, busy=aid == self.active_id, any_busy=self.active_id is not None)

    def _gpu_done(self, info: dict[str, Any]) -> None:
        self.gpu = info
        if info.get("name"):
            vram = f", {info['vram_mb'] / 1024:.0f} GB video memory" if info.get("vram_mb") else ""
            self.gpu_label.setText(f"Your graphics card: <b>{util.rich_escape(info['name'])}</b>{vram}")
        else:
            self.gpu_label.setText("No graphics card could be detected (add-ons need an NVIDIA GPU).")
        free = self._free_gb()
        for card in self.cards.values():
            card.set_requirements(info, free)

    def _toggle_experimental(self, on: bool) -> None:
        self.settings.show_experimental = bool(on)
        dc.save_settings(self.settings)
        self.experimentalChanged.emit(bool(on))
        if self.active_id is None:
            self.populate()

    # ---------------------------------------------------------------- actions
    def install(self, addon_id: str) -> bool:
        if self.manager is None or self.active_id is not None:
            return False
        card = self.cards.get(addon_id)
        try:
            m = self.manager.get_manifest(addon_id)
        except Exception as e:
            util.show_error(self, "Add-ons", e)
            return False
        needs_ack = bool(getattr(m, "requires_license_ack", lambda: False)())
        if needs_ack and not self.settings.license_accepted(addon_id):
            notices = license_notices(m)
            if not dc.confirm_license(
                self, m.title, m.license, url=getattr(m, "homepage", ""), notices=notices
            ):
                return False
            self.settings.accept_license(addon_id)
            dc.save_settings(self.settings)
        if getattr(m, "hf_gated", False) and not getattr(self.settings, "hf_token", ""):
            util.show_error(
                self,
                "Hugging Face token needed",
                f"{m.title} downloads a model that requires a free Hugging Face account.",
                details="1. Create an account at https://huggingface.co\n"
                "2. Open the model page linked on the add-on card and accept its terms\n"
                "3. Create a 'Read' token under Settings > Access Tokens\n"
                "4. Paste it into VR Photo Studio: Settings > AI models > Hugging Face token",
            )
            return False
        warn = [t for lvl, t in requirement_notes(m, self.gpu, self._free_gb()) if lvl == "warn"]
        if warn and not util.ask_yes_no(
            self, "Install anyway?", "\n".join(warn), informative="Do you want to try installing anyway?"
        ):
            return False
        self.active_id = addon_id
        if card:
            card.begin(f"Installing {m.title}… this can take 5-30 minutes.")
        token = getattr(self.settings, "hf_token", "") or None
        self.worker = self.tasks.start(
            self.manager.install,
            addon_id,
            hf_token=token,
            on_progress=lambda f, msg: card.set_progress(f, msg) if card else None,
            on_result=lambda _r: self._finished(addon_id, None),
            on_error=lambda e: self._finished(addon_id, e),
            on_cancelled=lambda: self._finished(addon_id, "cancel"),
            name=f"install-{addon_id}",
        )
        self.log_timer.start()
        self._apply_status()
        return True

    def _finished(self, addon_id: str, error: Any) -> None:
        self.log_timer.stop()
        self._refresh_log()
        self.active_id = None
        self.worker = None
        card = self.cards.get(addon_id)
        if error is None:
            msg = "Installed successfully."
            log.info("Add-on %s installed", addon_id)
        elif error == "cancel":
            msg = "Installation cancelled. Click Install to resume later."
        else:
            text, _hint = util.error_text(error)
            msg = f"Installation failed: {text}"
            util.show_error(self, "Installation failed", error)
        if card:
            card.end(msg)
        self.statuses.pop(addon_id, None)  # show "Checking…" until the new state is read
        self.refresh_status()

    def _refresh_log(self) -> None:
        aid = self.active_id or next(iter(self.cards), None)
        if aid is None or self.manager is None:
            return
        card = self.cards.get(aid)
        path_fn = getattr(self.manager, "install_log_path", None)
        if card is None or not callable(path_fn):
            return
        try:
            card.set_log(dc.tail_text(path_fn(aid), max_lines=200))
        except Exception:
            log.debug("Cannot read install log of %s", aid, exc_info=True)

    def cancel(self, addon_id: str) -> None:
        if self.worker is not None and addon_id == self.active_id:
            self.worker.cancel()
            card = self.cards.get(addon_id)
            if card:
                card.progress_label.setText("Cancelling… (waiting for the current step to stop)")

    def uninstall(self, addon_id: str) -> bool:
        if self.manager is None or self.active_id is not None:
            return False
        card = self.cards.get(addon_id)
        title = card.m.title if card else addon_id
        if not util.ask_yes_no(
            self,
            "Uninstall add-on",
            f"Uninstall {title}?",
            informative="This deletes its files and frees the disk space.",
        ):
            return False
        self.active_id = addon_id
        self._apply_status()

        def done(_r: Any = None, err: Any = None) -> None:
            self.active_id = None
            if err is not None:
                util.show_error(self, "Uninstall failed", err)
            self.statuses.pop(addon_id, None)
            self.refresh_status()

        self.tasks.start(
            self.manager.uninstall,
            addon_id,
            pass_progress=False,
            pass_cancel=False,
            on_result=lambda r: done(r),
            on_error=lambda e: done(err=e),
            name=f"uninstall-{addon_id}",
        )
        return True

    def open_folder(self, addon_id: str) -> None:
        st = self.statuses.get(addon_id) or {}
        p = st.get("path")
        if p and Path(p).is_dir():
            util.open_path(p)

    def test_brush(self, addon_id: str = BRUSH_ADDON) -> bool:
        """Run the Brush self-test in the background; the verdict is shown in a message box
        and stored in ``Settings.brush_test``."""
        if self.active_id is not None:
            return False
        try:
            from ...addons import capture
        except Exception as e:
            util.show_error(self, "Brush test", e)
            return False
        card = self.cards.get(addon_id)
        self.active_id = addon_id
        self.testing = True
        if card:
            card.begin("Testing Brush on this PC… this takes a minute or two.")
            card.log_view.setVisible(False)  # the install log is not this task's log
        self.worker = self.tasks.start(
            capture.brush_selftest,
            on_progress=lambda f, msg: card.set_progress(f, msg) if card else None,
            on_result=lambda r: self._test_finished(addon_id, r, None),
            on_error=lambda e: self._test_finished(addon_id, None, e),
            on_cancelled=lambda: self._test_finished(addon_id, None, "cancel"),
            name="brush-test",
        )
        self._apply_status()
        return True

    def _test_finished(self, addon_id: str, result: dict[str, Any] | None, error: Any) -> None:
        self.active_id = None
        self.worker = None
        self.testing = False
        card = self.cards.get(addon_id)
        if error == "cancel":
            msg = "Brush test cancelled."
        elif error is not None:
            text, _hint = util.error_text(error)
            msg = f"Brush test failed: {text}"
            util.show_error(self, "Brush test failed", error)
        else:
            from ...addons import capture

            result = dict(result or {})
            self.settings.brush_test = result
            dc.save_settings(self.settings)
            verdict = capture.brush_test_verdict(result)
            if card:
                card.set_test_result(result)
            if result.get("ok"):
                msg = "Brush works on this PC."
                util.show_info(self, "Brush test", verdict)
            else:
                msg = "Brush could not train on this PC."
                util.show_error(self, "Brush test", verdict)
        if card:
            card.end(msg)
        self._apply_status()

    # ---------------------------------------------------------------- closing
    def busy(self) -> bool:
        return self.active_id is not None

    def done(self, result: int) -> None:  # noqa: D401 - QDialog API
        if self.worker is not None and self.active_id is not None:
            if self.testing:
                title, question, info = "Brush test running", "Stop the Brush test and close?", None
            else:
                title, question = "Installation in progress", "Cancel the installation and close?"
                info = "You can resume it later; finished steps are kept."
            if not util.ask_yes_no(self, title, question, informative=info):
                return
            self.worker.cancel()
        self.log_timer.stop()
        self.tasks.shutdown(DIALOG_CLOSE_WAIT_MS)
        super().done(result)
