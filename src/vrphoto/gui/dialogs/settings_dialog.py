"""Settings dialog: GPU, performance, default models, output folder, headset, video
encoder, viewer ports, Hugging Face token, experimental features and theme."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.log import get_logger
from .. import options_meta as om
from .. import util
from ..panels.options import dynamic_choices
from ..workers import DIALOG_CLOSE_WAIT_MS, TaskRunner
from . import common as dc

log = get_logger(__name__)

QUEST_MODELS = [
    ("quest3", "Meta Quest 3 / 3S"),
    ("quest2", "Meta Quest 2"),
    ("pcvr", "PC VR / other headset"),
]
ENCODERS = [
    ("auto", "Automatic (fastest that works)"),
    ("nvenc", "NVIDIA (NVENC)"),
    ("amf", "AMD (AMF)"),
    ("qsv", "Intel (Quick Sync)"),
    ("x265", "Software (slow, always works)"),
]
THEMES = [("system", "Same as Windows"), ("dark", "Dark"), ("light", "Light")]


def _combo(choices: list[tuple[str, str]], current: str, tooltip: str = "") -> QComboBox:
    c = QComboBox()
    for v, label in choices:
        c.addItem(label, v)
    i = c.findData(current)
    if i < 0 and current:
        c.addItem(str(current), current)
        i = c.count() - 1
    c.setCurrentIndex(max(0, i))
    if tooltip:
        c.setToolTip(tooltip)
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(22)
    return c


class SettingsDialog(QDialog):
    """Edit :class:`~vrphoto.core.config.Settings`; "Save" persists them.

    Signals:
        saved(set): names of the settings that changed.
    """

    saved = Signal(set)

    def __init__(self, parent: QWidget | None = None, *, settings: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(700, 720)
        if settings is None:
            from ...core.config import get_settings

            settings = get_settings()
        self.settings = settings
        self.tasks = TaskRunner(self, max_threads=1)
        s = settings

        # ---- graphics & performance
        self.device = _combo(
            [("auto", "Automatic (fastest GPU)"), ("cpu", "CPU only (slow)")],
            s.device,
            "Which graphics card runs the AI models. 'Automatic' picks your best GPU.",
        )
        self.gpu_info = QLabel("Detecting graphics cards…")
        self.gpu_info.setObjectName("Muted")
        self.gpu_info.setWordWrap(True)
        self.fp16 = QCheckBox("Use half precision (faster, uses less GPU memory)")
        self.fp16.setChecked(bool(s.use_fp16))
        self.fp16.setToolTip(
            "Runs AI models in 16-bit floating point where supported. Turn off if results look wrong."
        )
        self.max_px = QSpinBox()
        self.max_px.setRange(512, 16384)
        self.max_px.setSingleStep(256)
        self.max_px.setSuffix(" px")
        self.max_px.setValue(int(s.max_process_px))
        self.max_px.setToolTip(
            "Photos larger than this (long side) are reduced while processing. "
            "Lower it if you run out of memory."
        )
        g1 = QGroupBox("Graphics card and speed")
        f1 = QFormLayout(g1)
        f1.addRow("Device", self.device)
        f1.addRow("", self.gpu_info)
        f1.addRow("", self.fp16)
        f1.addRow("Max processing size", self.max_px)

        # ---- models
        exp = om.COMMON_META["depth_model"].experimental_choices
        depth_choices = [
            (v, label)
            for v, label in self._choices("depth_models")
            if v not in exp or s.show_experimental or v == s.depth_model
        ]
        self.depth_model = _combo(depth_choices, s.depth_model, "Default depth model for new jobs.")
        self.inpaint_model = _combo(
            self._choices("inpaint_models"),
            s.inpaint_model,
            "Default look-behind (inpainting) model for new jobs.",
        )
        self.segment_model = _combo(
            self._choices("segment_models"), s.segment_model, "Default cut-out model for new jobs."
        )
        self.hf_token = QLineEdit(s.hf_token or "")
        self.hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token.setPlaceholderText("Only needed for some add-ons (hf_…)")
        self.hf_token.setToolTip(
            "A free Hugging Face 'Read' access token. Only some add-ons need it. It is stored on this PC "
            "and only ever sent to huggingface.co."
        )
        self.show_token = QPushButton("Show")
        self.show_token.setCheckable(True)
        self.show_token.toggled.connect(self._toggle_token)
        trow = QHBoxLayout()
        trow.addWidget(self.hf_token, 1)
        trow.addWidget(self.show_token)
        twrap = QWidget()
        twrap.setLayout(trow)
        trow.setContentsMargins(0, 0, 0, 0)
        g2 = QGroupBox("AI models")
        f2 = QFormLayout(g2)
        f2.addRow("Depth model", self.depth_model)
        f2.addRow("Look-behind model", self.inpaint_model)
        f2.addRow("Cut-out model", self.segment_model)
        f2.addRow("Hugging Face token", twrap)

        # ---- output
        self.output_dir = QLineEdit(s.output_dir or "")
        self.output_dir.setPlaceholderText(str(self._default_outputs()))
        self.output_dir.setToolTip("Where results are saved. Empty = Pictures\\VR Photo Studio.")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_output)
        orow = QHBoxLayout()
        orow.setContentsMargins(0, 0, 0, 0)
        orow.addWidget(self.output_dir, 1)
        orow.addWidget(browse)
        owrap = QWidget()
        owrap.setLayout(orow)
        self.quest_model = _combo(
            QUEST_MODELS, s.quest_model, "Picks resolutions and video formats your headset plays smoothly."
        )
        self.video_encoder = _combo(
            ENCODERS,
            s.video_encoder,
            "Which video encoder to use. 'Automatic' uses your graphics card if possible.",
        )
        g3 = QGroupBox("Results")
        f3 = QFormLayout(g3)
        f3.addRow("Output folder", owrap)
        f3.addRow("Headset", self.quest_model)
        f3.addRow("Video encoder", self.video_encoder)

        # ---- viewer
        self.https_port = QSpinBox()
        self.https_port.setRange(1024, 65535)
        self.https_port.setValue(int(s.https_port))
        self.https_port.setToolTip(
            "Port of the secure (https) viewer address used over Wi-Fi. Default 8443. While the viewer "
            "runs, devices on your network can open the viewer and the scenes in its library - "
            "nothing else on this PC (not your output folder)."
        )
        self.http_port = QSpinBox()
        self.http_port.setRange(1024, 65535)
        self.http_port.setValue(int(s.http_port))
        self.http_port.setToolTip(
            "Port of the local viewer address used over a USB cable and by 'Open on this PC' "
            "(only reachable from this PC). Default 8080; change it if another program uses it."
        )
        g4 = QGroupBox("Live viewer (Send to Quest)")
        f4 = QFormLayout(g4)
        f4.addRow("Wi-Fi (https) port", self.https_port)
        f4.addRow("USB (http) port", self.http_port)

        # ---- app
        self.show_experimental = QCheckBox("Show experimental features (models, modes and add-ons)")
        self.show_experimental.setChecked(bool(s.show_experimental))
        self.theme = _combo(THEMES, s.theme, "Colour scheme of the app.")
        g5 = QGroupBox("App")
        f5 = QFormLayout(g5)
        f5.addRow("", self.show_experimental)
        f5.addRow("Theme", self.theme)

        self.error_label = QLabel("")
        self.error_label.setObjectName("Error")
        self.error_label.setWordWrap(True)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.save)
        box.rejected.connect(self.reject)

        content = QWidget()
        cl = QVBoxLayout(content)
        for g in (g1, g2, g3, g4, g5):
            cl.addWidget(g)
        cl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        lay = QVBoxLayout(self)
        lay.addWidget(scroll, 1)
        lay.addWidget(self.error_label)
        lay.addWidget(box)

        self.tasks.start(
            self._detect_devices,
            pass_progress=False,
            pass_cancel=False,
            on_result=self._devices_done,
            on_error=lambda e: self.gpu_info.setText(""),
            name="list-devices",
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _choices(name: str) -> list[tuple[str, str]]:
        return [(str(v), str(label)) for v, label in dynamic_choices(name)]

    @staticmethod
    def _default_outputs() -> Path:
        try:
            from ...core.config import default_library_dir

            return default_library_dir()
        except Exception:
            return Path.home() / "Pictures" / "VR Photo Studio"

    @staticmethod
    def _detect_devices() -> list[tuple[str, str]]:
        from ...models.gpu import list_devices

        return [(str(d.index), d.label) for d in list_devices()]

    def _devices_done(self, devs: list[tuple[str, str]]) -> None:
        current = self.device.currentData()
        for value, label in devs:
            if self.device.findData(value) < 0:
                self.device.addItem(label, value)
            else:
                self.device.setItemText(self.device.findData(value), label)
        i = self.device.findData(current)
        if i >= 0:
            self.device.setCurrentIndex(i)
        self.gpu_info.setText(
            "Found: " + ", ".join(label for _, label in devs)
            if devs
            else "No GPU list available (normal on Linux); 'Automatic' still uses your GPU when possible."
        )

    def _toggle_token(self, on: bool) -> None:
        self.hf_token.setEchoMode(QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password)
        self.show_token.setText("Hide" if on else "Show")

    def _browse_output(self) -> None:
        start = self.output_dir.text().strip() or str(self._default_outputs())
        d = QFileDialog.getExistingDirectory(self, "Output folder", start)
        if d:
            self.output_dir.setText(str(Path(d)))

    # ---------------------------------------------------------------- saving
    def values(self) -> dict[str, Any]:
        return {
            "device": str(self.device.currentData() or "auto"),
            "use_fp16": self.fp16.isChecked(),
            "max_process_px": int(self.max_px.value()),
            "depth_model": str(self.depth_model.currentData() or "auto"),
            "inpaint_model": str(self.inpaint_model.currentData() or "auto"),
            "segment_model": str(self.segment_model.currentData() or "auto"),
            "hf_token": self.hf_token.text().strip(),
            "output_dir": self.output_dir.text().strip(),
            "quest_model": str(self.quest_model.currentData() or "quest3"),
            "video_encoder": str(self.video_encoder.currentData() or "auto"),
            "https_port": int(self.https_port.value()),
            "http_port": int(self.http_port.value()),
            "show_experimental": self.show_experimental.isChecked(),
            "theme": str(self.theme.currentData() or "system"),
        }

    def validate(self, v: dict[str, Any]) -> str | None:
        if v["https_port"] == v["http_port"]:
            return "The Wi-Fi and USB ports must be different."
        if v["output_dir"]:
            p = Path(v["output_dir"]).expanduser()
            if p.exists() and not p.is_dir():
                return f"The output folder {p} is a file, not a folder."
            if not p.is_absolute():
                return "Please choose the output folder with 'Browse…' (a full path such as C:\\Users\\…)."
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                return f"The output folder {p} cannot be created ({e.strerror or e}). Choose another folder."
        return None

    def save(self) -> bool:
        v = self.values()
        err = self.validate(v)
        if err:
            self.error_label.setText(err)
            return False
        changed = {k for k, val in v.items() if getattr(self.settings, k, None) != val}
        for k, val in v.items():
            setattr(self.settings, k, val)
        if not dc.save_settings(self.settings):
            util.show_error(self, "Settings", "The settings could not be saved (see the log for details).")
        try:
            from ...core.config import get_settings, set_settings

            if get_settings() is not self.settings:
                set_settings(self.settings)
        except Exception:  # pragma: no cover - defensive
            log.debug("set_settings failed", exc_info=True)
        if "device" in changed:
            self._reset_models_for_new_device()
        log.info("Settings saved%s", f" ({', '.join(sorted(changed))} changed)" if changed else "")
        self.saved.emit(changed)
        self.accept()
        return True

    @staticmethod
    def _reset_models_for_new_device() -> None:
        """Another GPU (or the CPU) was chosen: forget the sticky CPU fallbacks and drop
        every cached ONNX session, so models are loaded again on the new device.  Never
        blocks on a running inference (those models are closed when it finishes)."""
        try:
            from ...models import ort

            ort.reset_cpu_fallbacks()
            ort.unload_model(wait=False)
        except Exception:
            log.warning("Could not reset the loaded AI models after the device change", exc_info=True)

    def done(self, result: int) -> None:  # noqa: D401 - QDialog API
        self.tasks.shutdown(DIALOG_CLOSE_WAIT_MS)
        super().done(result)
