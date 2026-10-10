"""The main window: inputs (left), previews (centre), modes + options (right), job queue
and log (bottom), plus menus, toolbar and the dialogs."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QWidget,
)

from .. import APP_NAME
from ..core.errors import InputError, VRPhotoError
from ..core.log import get_logger
from . import options_meta as om
from . import util
from .bridge import QueueBridge
from .dialogs import common as dc
from .panels.input_list import InputListPanel, urls_to_paths
from .panels.log_panel import LogPanel
from .panels.modes import ModePanel
from .panels.preview import PreviewPanel
from .panels.queue_panel import QueuePanel
from .preview import PreviewEngine
from .server_ctl import ServerController
from .state import GuiState
from .workers import TaskRunner

log = get_logger(__name__)

#: depth/inpaint/segment option values whose model id differs from the option value
MODEL_ALIASES = {
    "moge2": "moge2-vitl",
    "moge": "moge2-vitl",
    "da3": "da3-mono-large",
    "isnet": "isnet-general",
}
#: On exit: how long to wait for cancelled jobs to stop (so add-on worker processes and
#: ffmpeg are terminated by the job instead of being left running without the app).
JOB_STOP_WAIT_S = 8.0


def app_icon() -> QIcon:
    try:
        from ..core.config import resource_path

        p = resource_path("gui", "resources", "icon.png")
        if p.is_file():
            return QIcon(str(p))
    except Exception:
        pass
    return QIcon()


def _b64(data: QByteArray) -> str:
    return base64.b64encode(bytes(data.data())).decode("ascii")


def _unb64(text: str) -> QByteArray | None:
    if not text:
        return None
    try:
        return QByteArray(base64.b64decode(text.encode("ascii")))
    except (ValueError, TypeError):
        return None


class MainWindow(QMainWindow):
    """The application window.

    Args:
        settings: :class:`~vrphoto.core.config.Settings` (default: the global settings).
        queue: an existing JobQueue (tests); default: created lazily by :class:`QueueBridge`.
        runner: job runner for a lazily created queue (default ``pipeline.run.run_job``).
        preview_engine: preview computations (tests inject a fake depth estimator).
        server_factory: ``f(settings) -> ViewerServer``-like object (tests).
        state: GUI state (default: loaded from ``gui_state.json``).
    """

    def __init__(
        self,
        settings: Any = None,
        *,
        queue: Any = None,
        runner: Callable[..., Any] | None = None,
        preview_engine: PreviewEngine | None = None,
        server_factory: Callable[[Any], Any] | None = None,
        state: GuiState | None = None,
    ) -> None:
        super().__init__()
        if settings is None:
            from ..core.config import get_settings

            settings = get_settings()
        self.settings = settings
        self.state = state or GuiState.load()
        self.dialogs: dict[str, QWidget] = {}
        self._shut_down = False
        self._declined_licenses: set[str] = set()  # preview prompts declined this session
        self.instance_guard: Any = None  # single-instance guard (set by run_gui)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.resize(1440, 900)
        self.setMinimumSize(980, 640)

        self.bridge = QueueBridge(queue, runner=runner, parent=self)
        self.server = ServerController(settings, self, server_factory=server_factory)
        self.tasks = TaskRunner(self, max_threads=1)

        # ---- panels
        self.inputs = InputListPanel(
            last_dir=self.state.last_open_dir or getattr(settings, "last_open_dir", "")
        )
        self.modes = ModePanel(
            common_defaults=self._common_defaults(),
            show_experimental=bool(getattr(settings, "show_experimental", False)),
            remembered=self.state.options,
        )
        self.preview = PreviewPanel(
            engine=preview_engine,
            options_provider=self.modes.current_options,
            license_gate=self._preview_license_ok,
        )
        self.queue = QueuePanel(self.bridge)
        self.log_panel = LogPanel()

        self.top_split = QSplitter(Qt.Orientation.Horizontal)
        self.top_split.addWidget(self.inputs)
        self.top_split.addWidget(self.preview)
        self.top_split.addWidget(self.modes)
        self.top_split.setStretchFactor(0, 0)
        self.top_split.setStretchFactor(1, 1)
        self.top_split.setStretchFactor(2, 0)
        self.top_split.setSizes([280, 720, 460])
        self.top_split.setChildrenCollapsible(False)
        self.bottom_split = QSplitter(Qt.Orientation.Horizontal)
        self.bottom_split.addWidget(self.queue)
        self.bottom_split.addWidget(self.log_panel)
        self.bottom_split.setSizes([900, 500])
        self.main_split = QSplitter(Qt.Orientation.Vertical)
        self.main_split.addWidget(self.top_split)
        self.main_split.addWidget(self.bottom_split)
        self.main_split.setStretchFactor(0, 1)
        self.main_split.setSizes([700, 190])
        self.setCentralWidget(self.main_split)

        self._build_actions()
        self._build_status()

        # ---- wiring
        self.inputs.currentChanged.connect(self.preview.set_path)
        self.inputs.pathsChanged.connect(self._update_create_hint)
        self.inputs.selectionChanged.connect(self._update_create_hint)
        self.modes.modeChanged.connect(self._mode_changed)
        self.modes.optionsChanged.connect(self.preview.options_changed)
        self.modes.createRequested.connect(self.create_jobs)
        self.queue.viewInHeadset.connect(self.view_in_headset)
        self.preview.statusMessage.connect(lambda m: self.statusBar().showMessage(m, 6000))
        self.inputs.notice.connect(lambda m: self.statusBar().showMessage(m, 12000))
        self.bridge.countsChanged.connect(self._counts_changed)
        self.server.started.connect(self._server_status)
        self.server.stopped.connect(self._server_status)
        self.server.failed.connect(lambda _m: self._server_status())

        if self.state.last_mode in om.MODE_BY_KIND:
            self.modes.set_mode(self.state.last_mode)
        self._restore_layout()
        self._update_create_hint()
        self.tasks.start(
            self._detect_gpu,
            pass_progress=False,
            pass_cancel=False,
            on_result=self._gpu_detected,
            name="gpu-status",
        )
        log.info("%s ready", APP_NAME)

    # ================================================================= construction
    def _common_defaults(self) -> dict[str, Any]:
        s = self.settings
        return {
            "depth_model": getattr(s, "depth_model", "auto") or "auto",
            "inpaint_model": getattr(s, "inpaint_model", "auto") or "auto",
            "segment_model": getattr(s, "segment_model", "auto") or "auto",
            "device": getattr(s, "device", "auto") or "auto",
            "max_process_px": int(getattr(s, "max_process_px", 4096) or 4096),
            "quest_model": getattr(s, "quest_model", "quest3") or "quest3",
        }

    def _action(
        self, text: str, slot: Callable[[], Any], *, shortcut: str | QKeySequence | None = None, tip: str = ""
    ) -> QAction:
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.triggered.connect(lambda _checked=False: slot())
        return a

    def _build_actions(self) -> None:
        self.act_add = self._action(
            "Add photos…",
            self.inputs.browse_files,
            shortcut=QKeySequence.StandardKey.Open,
            tip="Add photos or videos to work with.",
        )
        self.act_add_folder = self._action(
            "Add folder…", self.inputs.browse_folder, tip="Add all photos and videos in a folder."
        )
        self.act_create = self._action(
            "Create",
            self.create_jobs,
            shortcut="Ctrl+Return",
            tip="Create the selected kind of result for the selected photos.",
        )
        self.act_quest = self._action(
            "Send to Quest",
            self.open_quest,
            tip="See your results in the Quest Browser (Wi-Fi or USB cable).",
        )
        self.act_models = self._action("AI models", self.open_models, tip="Download or delete AI models.")
        self.act_addons = self._action(
            "Add-ons",
            self.open_addons,
            tip="Install optional AI tools (TRELLIS, Capture, AI outpainting...).",
        )
        self.act_settings = self._action(
            "Settings",
            self.open_settings,
            shortcut=QKeySequence.StandardKey.Preferences,
            tip="Graphics card, folders, headset, viewer ports and more.",
        )
        self.act_log = self._action("Log", self.open_log, tip="Show the detailed log file.")
        self.act_guide = self._action(
            "User guide",
            lambda: util.open_url(util.USER_GUIDE_URL),
            shortcut=QKeySequence.StandardKey.HelpContents,
            tip="Open the user guide in your web browser.",
        )
        self.act_about = self._action(f"About {APP_NAME}", self.open_about)
        self.act_outputs = self._action(
            "Open output folder", self._open_outputs, tip="Open the folder where results are saved."
        )
        self.act_quit = self._action("Quit", self.close, shortcut=QKeySequence.StandardKey.Quit)
        self.addAction(self.act_create)  # keyboard shortcut works anywhere in the window

        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        for a in (self.act_add, self.act_add_folder, self.act_outputs):
            m_file.addAction(a)
        m_file.addSeparator()
        m_file.addAction(self.act_quit)
        m_tools = mb.addMenu("&Tools")
        for a in (self.act_quest, self.act_models, self.act_addons, self.act_settings):
            m_tools.addAction(a)
        m_help = mb.addMenu("&Help")
        for a in (self.act_guide, self.act_log, self.act_about):
            m_help.addAction(a)

        tb = QToolBar("Main")
        tb.setObjectName("MainToolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        for a in (self.act_add, self.act_quest, self.act_models, self.act_addons, self.act_settings):
            tb.addAction(a)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        tb.addAction(self.act_guide)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self.toolbar = tb

    def _build_status(self) -> None:
        sb = self.statusBar()
        self.gpu_label = QLabel("GPU: checking…")
        self.gpu_label.setObjectName("Muted")
        self.server_label = QLabel("Viewer: off")
        self.server_label.setObjectName("Muted")
        self.jobs_label = QLabel("")
        self.jobs_label.setObjectName("Muted")
        sb.addPermanentWidget(self.jobs_label)
        sb.addPermanentWidget(self.server_label)
        sb.addPermanentWidget(self.gpu_label)

    def _restore_layout(self) -> None:
        g = _unb64(self.state.geometry)
        if g is not None:
            self.restoreGeometry(g)
        for name, split in (
            ("top", self.top_split),
            ("bottom", self.bottom_split),
            ("main", self.main_split),
        ):
            data = _unb64(self.state.splitters.get(name, ""))
            if data is not None:
                split.restoreState(data)

    # ================================================================= status
    @staticmethod
    def _detect_gpu() -> dict[str, Any]:
        """Worker thread: GPU name/VRAM for the status bar + the device picker choices."""
        from .dialogs.addons_dialog import detect_gpu
        from .panels.options import detect_device_choices

        info = detect_gpu()
        info["device_choices"] = detect_device_choices()
        return info

    def _gpu_detected(self, info: dict[str, Any]) -> None:
        if info.get("device_choices") and len(info["device_choices"]) > 2:
            self.modes.refresh_choices(bool(getattr(self.settings, "show_experimental", False)))
        dev = str(getattr(self.settings, "device", "auto"))
        if dev == "cpu":
            self.gpu_label.setText("AI runs on: CPU (set in Settings)")
        elif info.get("name"):
            vram = f" ({info['vram_mb'] / 1024:.0f} GB)" if info.get("vram_mb") else ""
            self.gpu_label.setText(f"GPU: {info['name']}{vram}")
        else:
            self.gpu_label.setText("GPU: automatic")
        self.gpu_info = info

    def _server_status(self, *_: Any) -> None:
        if self.server.running:
            self.server_label.setText(f"Viewer: {self.server.https_url() or self.server.local_url()}")
        else:
            self.server_label.setText("Viewer: off")

    def _counts_changed(self, running: int, queued: int) -> None:
        if running or queued:
            self.jobs_label.setText(f"Jobs: {running} working, {queued} waiting")
        else:
            self.jobs_label.setText("")

    def _mode_changed(self, kind: str) -> None:
        self._update_create_hint()
        self.preview.options_changed()

    def _update_create_hint(self) -> None:
        self.modes.update_inputs_hint(self.inputs.selected_paths(), self.inputs.count())

    # ================================================================= inputs
    def add_inputs(self, paths: Iterable[str | Path]) -> int:
        """Add files/folders to the input list (same API as drag & drop)."""
        n = self.inputs.add_paths(paths)
        if n and not self.inputs.last_truncated:  # keep the "folder too big" notice visible
            self.statusBar().showMessage(f"Added {n} file{'s' if n != 1 else ''}", 4000)
        return n

    def bring_to_front(self, files: list[str] | None = None) -> None:
        """Show this window (another start of the app was handed over) and add its files."""
        if self._shut_down:
            return
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        if files:
            self.add_inputs(files)

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e) -> None:  # noqa: N802, ANN001
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:  # noqa: N802
        paths = urls_to_paths(e.mimeData())
        if paths:
            e.acceptProposedAction()
            n = self.add_inputs(paths)
            if not n and not self.inputs.last_truncated:
                self.statusBar().showMessage(
                    "Nothing new to add: no supported photos or videos, or they are already in the list.",
                    6000,
                )

    # ================================================================= jobs
    def _model_spec(self, name: str) -> Any:
        """Registry spec of a depth/inpaint/segment option value, or None ("auto", unknown)."""
        reg, _err = dc.try_backend("models.registry")
        if reg is None or not name:
            return None
        try:
            return reg.get_spec(MODEL_ALIASES.get(name, name))
        except Exception:
            return None

    def ensure_license(self, name: str) -> bool:
        """True if model option ``name`` may be used; asks to accept a non-commercial license
        (remembered via ``Settings.accept_license``)."""
        m = self._model_spec(str(name or ""))
        if m is None or not getattr(m, "noncommercial", False) or self.settings.license_accepted(m.id):
            return True
        if not dc.confirm_license(self, m.title, m.license, url=getattr(m, "page_url", "")):
            return False
        self.settings.accept_license(m.id)
        self._declined_licenses.discard(m.id)
        dc.save_settings(self.settings)
        return True

    def _preview_license_ok(self, depth_model: str) -> bool:
        """License gate of the preview: asks once per session, then shows a hint instead."""
        m = self._model_spec(str(depth_model or ""))
        if m is None or not getattr(m, "noncommercial", False) or self.settings.license_accepted(m.id):
            return True
        if m.id in self._declined_licenses:
            return False
        if self.ensure_license(depth_model):
            return True
        self._declined_licenses.add(m.id)
        return False

    def _check_model_licenses(self, spec: Any) -> bool:
        """Ask for non-commercial license acceptance of models the job will download."""
        common = getattr(spec, "common", None)
        for attr in ("depth_model", "inpaint_model", "segment_model"):
            if not self.ensure_license(str(getattr(common, attr, "") or "")):
                return False
        return True

    def _check_addons(self, kind: str, values: dict[str, Any]) -> bool:
        """True if the add-ons this job needs are installed (or cannot be checked)."""
        cands = om.required_addon_candidates(kind, values)
        if not cands:
            return True
        mgr, _err = dc.try_backend("addons.manager")
        if mgr is None:
            return True
        for aid in cands:
            try:
                m = mgr.get_manifest(aid)
            except Exception:
                continue  # unknown id -> let the pipeline report it
            try:
                is_installed = getattr(mgr, "is_installed", None)
                installed = (
                    bool(is_installed(aid))
                    if callable(is_installed)
                    else bool(mgr.status(aid).get("installed"))
                )
            except Exception:
                installed = True  # cannot check: let the pipeline report a missing add-on
            if installed:
                return True
            if util.ask_yes_no(
                self,
                "Add-on needed",
                f"This needs the '{m.title}' add-on, which is not installed yet.",
                informative="Open the Add-ons window to install it? (It is a large download.)",
                default_yes=True,
            ):
                self.open_addons()
            return False
        return True

    def create_jobs(self) -> list[str]:
        """Build and submit jobs for the selected inputs in the current mode; returns job ids."""
        fw = QApplication.focusWidget()
        if isinstance(fw, QAbstractSpinBox):  # Ctrl+Enter while typing a number: use it
            fw.interpretText()
        if not self.modes.create_button.isEnabled() and self.inputs.count() == 0:
            util.show_info(
                self, "Nothing to do", "Add photos on the left first (or drag them into the window)."
            )
            return []
        kind = self.modes.current_kind()
        mode = om.MODE_BY_KIND[kind]
        paths, skipped = self.modes.accepted(self.inputs.selected_paths())
        if not paths:
            need = "videos" if mode.accepts == frozenset({"video"}) else "photos"
            util.show_info(self, mode.title, f"{mode.title} needs {need}. Add or select some on the left.")
            return []
        try:
            specs = self.modes.build_specs(paths)
        except InputError as e:
            util.show_error(self, "Please check the options", e)
            return []
        except Exception as e:  # unexpected (e.g. bad remembered JSON)
            util.show_error(self, "Please check the options", e)
            return []
        if not specs:
            return []
        if not self._check_addons(kind, self.modes.current_panel().job_values()):
            return []
        if not self._check_model_licenses(specs[0]):
            return []
        ids: list[str] = []
        for spec in specs:
            try:
                job_id = self.bridge.submit(spec)
            except VRPhotoError as e:
                util.show_error(self, "Could not start the job", e)
                break
            except Exception as e:
                util.show_error(self, "Could not start the job", e)
                break
            self.queue.add_job(job_id, spec)
            ids.append(job_id)
        if ids:
            msg = f"Added {len(ids)} {mode.title} job{'s' if len(ids) != 1 else ''} to the queue"
            if skipped:
                msg += f" ({len(skipped)} unsuitable file{'s' if len(skipped) != 1 else ''} skipped)"
            self.statusBar().showMessage(msg, 6000)
            log.info(msg)
            try:  # remember these options now (not only on a clean exit)
                self.save_state()
            except Exception:
                log.debug("Could not save the GUI state", exc_info=True)
        return ids

    # ================================================================= dialogs
    def _show_dialog(self, name: str, factory: Callable[[], QWidget]) -> QWidget | None:
        """Show the single instance of dialog ``name`` (created by ``factory`` if needed).

        Returns None (after a friendly message) if the dialog cannot be created, e.g.
        because one of its modules is broken in this installation.
        """
        dlg = self.dialogs.get(name)
        if dlg is not None:
            try:
                if dlg.isVisible():
                    dlg.raise_()
                    dlg.activateWindow()
                    return dlg
            except RuntimeError:  # already deleted
                pass
            self.dialogs.pop(name, None)
        try:
            dlg = factory()
        except Exception as e:
            log.exception("Cannot open the %s window", name)
            util.show_error(
                self,
                "Not available",
                f"The {name} window could not be opened: {e}",
                details=util.format_exception(e),
            )
            return None
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.finished.connect(lambda _r, n=name, d=dlg: self._dialog_closed(n, d))
        self.dialogs[name] = dlg
        dlg.show()
        dlg.raise_()
        return dlg

    def _dialog_closed(self, name: str, dlg: QWidget) -> None:
        if self.dialogs.get(name) is dlg:
            self.dialogs.pop(name, None)
        if name == "models":  # a license may have been accepted / a model downloaded
            self.preview.options_changed()

    def open_models(self) -> QWidget | None:
        def make() -> QWidget:
            from .dialogs.models_dialog import ModelsDialog

            return ModelsDialog(self, settings=self.settings)

        return self._show_dialog("models", make)

    def open_addons(self) -> QWidget | None:
        def make() -> QWidget:
            from .dialogs.addons_dialog import AddonsDialog

            d = AddonsDialog(self, settings=self.settings)
            d.experimentalChanged.connect(lambda _on: self.apply_settings({"show_experimental"}))
            return d

        return self._show_dialog("addons", make)

    def open_settings(self) -> QWidget | None:
        def make() -> QWidget:
            from .dialogs.settings_dialog import SettingsDialog

            d = SettingsDialog(self, settings=self.settings)
            d.saved.connect(self.apply_settings)
            return d

        return self._show_dialog("settings", make)

    def open_about(self) -> QWidget | None:
        def make() -> QWidget:
            from .dialogs.about_dialog import AboutDialog

            return AboutDialog(self)

        return self._show_dialog("about", make)

    def open_log(self) -> QWidget | None:
        def make() -> QWidget:
            from .dialogs.log_dialog import LogDialog

            return LogDialog(self)

        return self._show_dialog("log", make)

    def _job_files(self, job_id: str | None) -> list[Path]:
        if job_id:
            return self.queue.output_files(job_id)
        return []

    def open_quest(self, scene_id: str | None = None, job_id: str | None = None) -> QWidget | None:
        """Open "Send to Quest" (focused on ``scene_id``, offering the files of ``job_id``)."""
        job_id = job_id or self.queue.selected_job()
        if scene_id is None and job_id and job_id in self.queue.jobs:
            scene_id = self.queue.jobs[job_id].scene_id
        files = self._job_files(job_id)
        existing = self.dialogs.get("quest")

        def make() -> QWidget:
            from .dialogs.quest_dialog import QuestDialog

            return QuestDialog(
                self, controller=self.server, settings=self.settings, focus_scene=scene_id, files=files
            )

        dlg = self._show_dialog("quest", make)
        if dlg is not None and existing is dlg:  # already open: refocus
            dlg.set_files(files)  # type: ignore[attr-defined]
            dlg.refresh_library()  # type: ignore[attr-defined]
            dlg.set_focus_scene(scene_id)  # type: ignore[attr-defined]
        return dlg

    def view_in_headset(self, job_id: str, scene_id: Any) -> QWidget | None:
        return self.open_quest(scene_id=scene_id or None, job_id=job_id)

    def _open_outputs(self) -> None:
        try:
            util.open_path(self.settings.outputs_dir)
        except Exception as e:
            util.show_error(self, "Output folder", e)

    # ================================================================= settings
    def apply_settings(self, changed: set) -> None:
        """React to a saved Settings dialog."""
        app = QApplication.instance()
        if "theme" in changed and app is not None:
            from .theme import apply_theme

            apply_theme(app, self.settings.theme)
        if "show_experimental" in changed:
            self.modes.refresh_choices(bool(self.settings.show_experimental))
        common = {"depth_model", "inpaint_model", "segment_model", "device", "max_process_px", "quest_model"}
        if changed & common:
            defaults = self._common_defaults()
            self.modes.apply_common_defaults(defaults, changed & common)
            self.preview.options_changed()
        if changed & {"https_port", "http_port", "output_dir"} and self.server.running:
            self.server.restart()
        if "device" in changed:
            self._gpu_detected(getattr(self, "gpu_info", {}) or {})

    # ================================================================= closing
    def save_state(self) -> None:
        st = self.state
        st.last_mode = self.modes.current_kind()
        st.options = self.modes.remember_all()
        st.geometry = _b64(self.saveGeometry())
        st.splitters = {
            "top": _b64(self.top_split.saveState()),
            "bottom": _b64(self.bottom_split.saveState()),
            "main": _b64(self.main_split.saveState()),
        }
        st.last_open_dir = self.inputs.last_dir
        st.save()

    def closeEvent(self, e: QCloseEvent) -> None:  # noqa: N802 - Qt API
        active = self.queue.active_count()
        if active and not self._shut_down:
            if not util.ask_yes_no(
                self,
                "Jobs are still running",
                f"{active} job{'s are' if active != 1 else ' is'} still running or waiting.",
                informative="Cancel them and quit?",
            ):
                e.ignore()
                return
        for name, dlg in list(self.dialogs.items()):
            try:
                busy = getattr(dlg, "busy", None)
                if callable(busy) and busy():
                    dlg.raise_()
                dlg.close()
                if dlg.isVisible():  # the dialog refused (user wants to keep a download running)
                    e.ignore()
                    return
            except RuntimeError:
                pass
            self.dialogs.pop(name, None)
        self.shutdown()
        e.accept()

    def shutdown(self) -> None:
        """Stop jobs, previews, the viewer server and save state.  Idempotent.

        The window is hidden first; cancelled jobs then get up to ``JOB_STOP_WAIT_S`` to
        stop, so they can end their add-on / ffmpeg processes instead of leaving them
        running (and using the GPU) after the app is gone.
        """
        if self._shut_down:
            return
        self._shut_down = True
        if self.instance_guard is not None:  # a new start must not hand over to a closing app
            try:
                self.instance_guard.close()
            except Exception:
                log.debug("Could not close the single-instance guard", exc_info=True)
        try:
            self.save_state()
        except Exception:
            log.exception("Could not save the window state")
        self.hide()
        self.preview.shutdown()
        self.inputs.shutdown()
        self.server.shutdown()  # free the viewer ports first (in case the app is restarted)
        self.bridge.shutdown(wait_s=JOB_STOP_WAIT_S)
        self.tasks.shutdown(2000)
        self.log_panel.detach()
        log.info("Main window closed")
