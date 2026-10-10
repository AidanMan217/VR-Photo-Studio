"""Send to Quest dialog: the live viewer (Wi-Fi), USB helpers and the scene library.

* **Wi-Fi** - start/stop the viewer server, show the HTTPS address in big letters plus a
  QR code, step-by-step instructions for the Quest Browser certificate warning, firewall
  status and an "Allow through firewall" button.
* **USB cable** - detect adb, install Google's platform tools, set up ``adb reverse`` and
  open the page in the headset, copy result files to the Quest (MTP on Windows, else adb).
* **Your scenes** - the viewer library with open / show / delete.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.log import get_logger
from .. import util
from ..server_ctl import ServerController
from ..workers import DIALOG_CLOSE_WAIT_MS, TaskRunner
from . import common as dc

log = get_logger(__name__)

WIFI_STEPS = """
<ol style="margin-left:-20px">
<li>Make sure this PC and your Quest are on the <b>same network</b> (same router, not a guest network).</li>
<li>Put on the Quest, open the <b>Browser</b> and type the address above exactly as shown -
including <b>https://</b> and the <b>:{port}</b> at the end. (If your Quest can scan QR codes, you can scan
the code instead.)</li>
<li>The browser warns <b>"Your connection is not private"</b>. This is expected: tap <b>Advanced</b>, then
<b>Proceed to {host} (unsafe)</b>.<br><span style="color:gray">Why is this safe? VR in a web page needs an
encrypted (https) connection. Your PC made its own certificate for this, which the browser cannot check -
on your home network that is fine.</span></li>
<li>Tap a scene in the gallery and press <b>Enter VR</b> (or <b>Enter AR</b> to see your room). Allow the
permission prompts.</li>
<li>Tip: bookmark the page in the Quest Browser.</li>
</ol>
"""

USB_STEPS = """
<p>Use a USB cable if your network blocks devices from seeing each other, or to avoid the certificate
warning. One-time setup:</p>
<ol style="margin-left:-20px">
<li>Turn on <b>Developer Mode</b> for your headset in the Meta Horizon phone app
(<i>Devices → Headset settings → Developer mode</i>; needs a free Meta developer account).</li>
<li>Click <b>Install USB tools</b> below (or install Meta Quest Developer Hub).</li>
<li>Connect the Quest with a USB-C data cable and choose <b>Always allow</b> on <b>Allow USB debugging?</b>
inside the headset.</li>
<li>Click <b>Set up USB connection</b>. The page opens in the Quest Browser at
<b>http://localhost:{port}/</b> - no certificate warning.</li>
</ol>
<p><b>Copying files</b> works without Developer Mode on Windows: connect the cable and choose <b>Allow</b>
when the headset asks to allow access to data.</p>
"""


class QuestDialog(QDialog):
    """See results in the Quest: live viewer over Wi-Fi or USB, file copy, library."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        controller: ServerController,
        settings: Any = None,
        focus_scene: str | None = None,
        files: list[Path] | None = None,
        autostart: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Send to Quest")
        self.resize(860, 720)
        if settings is None:
            from ...core.config import get_settings

            settings = get_settings()
        self.settings = settings
        self.ctl = controller
        self.focus_scene = focus_scene
        self.files: list[Path] = [Path(f) for f in (files or [])]
        self.tasks = TaskRunner(self, max_threads=2)
        self.net, self._net_err = dc.try_backend("webxr.net")
        self.quest, self._quest_err = dc.try_backend("webxr.quest")
        self.scene_mod, self._scene_err = dc.try_backend("webxr.scene")
        self.adb_path: str | None = None
        self.devices: list[str] = []
        self._usb_after_start = False  # "Set up USB" was clicked while the viewer was stopped
        self._http_problem_text = ""  # last "USB (http) port busy" message shown in the USB tab
        self._long_tasks: set[str] = set()  # "copy" / "install" while running (asked on close)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_wifi(), "Wi-Fi (recommended)")
        self.usb_tab_index = self.tabs.addTab(self._build_usb(), "USB cable")
        self.tabs.addTab(self._build_library(), "Your scenes")
        self._usb_checked_once = False
        # adb is only looked for when the USB tab is opened: asking adb for devices starts
        # its background server, which Wi-Fi users do not need
        self.tabs.currentChanged.connect(self._tab_changed)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(box)

        self.ctl.started.connect(self._server_started)
        self.ctl.stopped.connect(self._server_stopped)
        self.ctl.failed.connect(self._server_failed)
        self.ctl.busyChanged.connect(self._update_server_ui)
        self._update_server_ui()
        if self.ctl.running:
            self._server_started(self.ctl.urls())
        elif autostart:
            self.start_server()
        self.refresh_library()

    def _tab_changed(self, index: int) -> None:
        if index == self.usb_tab_index and not self._usb_checked_once:
            self.check_usb()

    # ================================================================= Wi-Fi tab
    def _build_wifi(self) -> QWidget:
        w = QWidget()
        self.status_label = QLabel("The live viewer is stopped.")
        self.status_label.setWordWrap(True)
        self.start_button = QPushButton("Start viewer")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self.toggle_server)
        top = QHBoxLayout()
        top.addWidget(self.status_label, 1)
        top.addWidget(self.start_button)

        self.url_label = QLabel("")
        self.url_label.setObjectName("BigUrl")
        self.url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.url_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.url_label.setToolTip("Type this address into the Quest Browser.")
        self.scene_label = QLabel("")
        self.scene_label.setObjectName("Muted")
        self.scene_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.scene_label.setWordWrap(True)
        self.other_urls = QLabel("")
        self.other_urls.setObjectName("Muted")
        self.other_urls.setWordWrap(True)
        self.other_urls.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.copy_button = QPushButton("Copy address")
        self.copy_button.clicked.connect(self.copy_url)
        self.open_pc_button = QPushButton("Open on this PC")
        self.open_pc_button.setToolTip("Open the viewer in this PC's browser to check that it works.")
        self.open_pc_button.clicked.connect(self.open_on_pc)
        urow = QHBoxLayout()
        urow.addWidget(self.copy_button)
        urow.addWidget(self.open_pc_button)
        urow.addStretch(1)

        self.qr_label = QLabel()
        self.qr_label.setFixedSize(220, 220)
        self.qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr_label.setToolTip("QR code of the address (for phones / Quest passthrough scanning).")
        self.steps = QLabel("")
        self.steps.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.steps.setWordWrap(True)
        self.steps.setTextFormat(Qt.TextFormat.RichText)
        mid = QHBoxLayout()
        mid.addWidget(self.steps, 1)
        mid.addWidget(self.qr_label, 0, Qt.AlignmentFlag.AlignTop)

        self.fw_label = QLabel("Firewall: not checked yet.")
        self.fw_label.setWordWrap(True)
        self.fw_button = QPushButton("Allow through firewall")
        self.fw_button.setToolTip(
            "Adds a Windows Firewall rule so the Quest can reach this PC (Windows asks for permission)."
        )
        self.fw_button.clicked.connect(self.allow_firewall)
        self.fw_check_button = QPushButton("Check again")
        self.fw_check_button.clicked.connect(self.check_firewall)
        fw = QHBoxLayout()
        fw.addWidget(self.fw_label, 1)
        fw.addWidget(self.fw_check_button)
        fw.addWidget(self.fw_button)

        lay = QVBoxLayout(w)
        lay.addLayout(top)
        lay.addWidget(self.url_label)
        lay.addWidget(self.scene_label)
        lay.addLayout(urow)
        lay.addWidget(self.other_urls)
        lay.addLayout(mid, 1)
        lay.addWidget(dc.hline())
        lay.addLayout(fw)
        if self.net is None:
            lay.addWidget(dc.unavailable_label("Network helpers", self._net_err or ""))
        return w

    def _update_server_ui(self, *_: Any) -> None:
        busy = self.ctl.busy
        running = self.ctl.running
        self.start_button.setEnabled(not busy)
        self.start_button.setText("Stop viewer" if running else "Start viewer")
        self.start_button.setObjectName("" if running else "PrimaryButton")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        for b in (self.copy_button, self.open_pc_button):
            b.setEnabled(running)
        if busy:
            self.status_label.setText("Starting the live viewer…" if not running else "Stopping…")

    def toggle_server(self) -> None:
        if self.ctl.running:
            self.ctl.stop()
        else:
            self.start_server()

    def start_server(self) -> None:
        self.status_label.setText("Starting the live viewer…")
        self.ctl.start()

    def current_url(self) -> str:
        return self.ctl.scene_url(self.focus_scene)

    def _server_started(self, urls: dict) -> None:
        self._update_server_ui()
        https = urls.get("https", "")
        self.status_label.setText(
            "The live viewer is running. Leave this app open while you use the headset."
        )
        self.status_label.setObjectName("Success")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.url_label.setText(https or urls.get("local", ""))
        port, host = "8443", "this PC"
        try:
            from urllib.parse import urlparse

            u = urlparse(https)
            port, host = str(u.port or port), u.hostname or host
        except Exception:
            pass
        self.steps.setText(WIFI_STEPS.format(port=port, host=host))
        others = [v for k, v in urls.items() if k.startswith("https_")]
        self.other_urls.setText(
            ("If that address does not work, try: " + "   ".join(others)) if others else ""
        )
        self._update_focus()
        self.check_firewall()
        self.usb_steps.setText(USB_STEPS.format(port=self._http_port()))
        self._show_http_problem()
        if self._usb_after_start:
            self._usb_after_start = False
            self.setup_usb()

    def _show_http_problem(self) -> str:
        """Tell the USB tab when the configured http port was busy (Wi-Fi still works).

        Returns the problem text only when USB mode is unavailable (no http listener at all);
        a note about a substitute port is shown but does not block USB mode.
        """
        problem = self.ctl.http_problem()
        text = problem or self.ctl.http_note()
        if text:
            self.usb_result.setText(text)
        elif self.usb_result.text() == self._http_problem_text:
            self.usb_result.setText("")
        self._http_problem_text = text
        return problem

    def _update_focus(self) -> None:
        url = self.current_url()
        if self.focus_scene and url:
            self.scene_label.setText(f"This scene: {url}")
        else:
            self.scene_label.setText("The gallery shows all your scenes, newest first.")
        self._set_qr(url or self.url_label.text())

    def _set_qr(self, text: str) -> None:
        if not text or self.net is None:
            self.qr_label.clear()
            return
        try:
            pm = util.png_to_pixmap(self.net.qr_png(text, box_size=6))
            self.qr_label.setPixmap(
                pm.scaled(
                    QSize(210, 210),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )
        except Exception as e:
            log.debug("QR code failed: %s", e)
            self.qr_label.setText("(QR code unavailable)")

    def _server_stopped(self) -> None:
        self._update_server_ui()
        self.status_label.setText("The live viewer is stopped.")
        self.status_label.setObjectName("")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.url_label.setText("")
        self.scene_label.setText("")
        self.other_urls.setText("")
        self.qr_label.clear()

    def _server_failed(self, msg: str) -> None:
        self._update_server_ui()
        if self._usb_after_start:
            self._usb_after_start = False
            self.usb_result.setText("The live viewer could not start, so the USB connection was not set up.")
        self.status_label.setText(f"The live viewer could not start:\n{msg}")
        self.status_label.setObjectName("Error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def copy_url(self) -> None:
        url = self.current_url() or self.url_label.text()
        if url:
            QGuiApplication.clipboard().setText(url)

    def open_on_pc(self) -> None:
        url = self.ctl.scene_url(self.focus_scene, base=self.ctl.local_url() or None) or self.current_url()
        if url:
            util.open_url(url)

    def set_focus_scene(self, scene_id: str | None) -> None:
        """Show the address / QR code of one scene (None = gallery)."""
        self.focus_scene = scene_id
        if self.ctl.running:
            self._update_focus()
        self._select_scene(scene_id)
        self.tabs.setCurrentIndex(0)

    def set_files(self, files: list[Path]) -> None:
        self.files = [Path(f) for f in files]
        self._update_files_label()

    # firewall
    def _https_port(self) -> int:
        try:
            return int(self.ctl.server.https_port or self.settings.https_port)
        except Exception:
            return int(self.settings.https_port)

    def _http_port(self) -> int:
        try:
            return int(self.ctl.server.http_port or self.settings.http_port)
        except Exception:
            return int(self.settings.http_port)

    def check_firewall(self) -> None:
        if self.net is None:
            return
        self.fw_label.setText("Firewall: checking…")
        fn = getattr(self.net, "firewall_report", None)

        def work() -> dict[str, str]:
            if callable(fn):
                return dict(fn(self._https_port()))
            return {"status": self.net.firewall_status(self._https_port()), "reason": ""}

        self.tasks.start(
            work,
            pass_progress=False,
            pass_cancel=False,
            on_result=self._firewall_done,
            on_error=lambda e: self.fw_label.setText(f"Firewall: could not check ({e})."),
            name="firewall-status",
        )

    def _firewall_done(self, rep: dict[str, str]) -> None:
        st = rep.get("status", "unknown")
        reason = rep.get("reason", "")
        if st == "ok":
            text, name = "Firewall: OK - the Quest can connect.", "Success"
        elif st == "blocked":
            text, name = (
                ("Firewall: the Quest is probably BLOCKED. Click 'Allow through firewall'. " + reason),
                "Warning",
            )
        else:
            text = (
                "Firewall: unknown."
                + (f" {reason}" if reason else "")
                + (
                    " If the Quest cannot connect, allow VR Photo Studio through the firewall."
                    if sys.platform == "win32"
                    else ""
                )
            )
            name = "Muted"
        self.fw_label.setText(text.strip())
        self.fw_label.setObjectName(name)
        self.fw_label.style().unpolish(self.fw_label)
        self.fw_label.style().polish(self.fw_label)
        self.fw_button.setVisible(sys.platform == "win32" or st == "blocked")

    def allow_firewall(self) -> None:
        if self.net is None:
            return
        self.fw_label.setText("Waiting for Windows to ask for permission…")
        # Only the Wi-Fi (https) port: the USB (http) listener is bound to 127.0.0.1 (desktop
        # browser and adb reverse), which the firewall never filters.  Opening it would expose
        # whatever other program listens on that port to the whole network.
        ports = [self._https_port()]
        self.tasks.start(
            self.net.add_firewall_rule,
            ports,
            pass_progress=False,
            pass_cancel=False,
            on_result=self._firewall_added,
            on_error=lambda e: util.show_error(self, "Firewall", e),
            name="firewall-rule",
        )

    def _firewall_added(self, ok: bool) -> None:
        if ok:
            self.fw_label.setText("Firewall rule added.")
            self.check_firewall()
        else:
            self.fw_label.setText(
                "The firewall rule was not added (cancelled, or not on Windows). You can allow the app in "
                "Windows Security → Firewall & network protection → Allow an app through firewall."
            )

    # ================================================================= USB tab
    def _build_usb(self) -> QWidget:
        w = QWidget()
        self.usb_steps = QLabel(USB_STEPS.format(port=self._http_port()))
        self.usb_steps.setWordWrap(True)
        self.usb_steps.setTextFormat(Qt.TextFormat.RichText)
        self.adb_label = QLabel("USB tools (adb): not checked yet.")
        self.adb_label.setWordWrap(True)
        self.usb_check_button = QPushButton("Check again")
        self.usb_check_button.clicked.connect(self.check_usb)
        self.install_tools_button = QPushButton("Install USB tools")
        self.install_tools_button.setToolTip("Download Google's Android SDK Platform-Tools (about 7 MB).")
        self.install_tools_button.clicked.connect(self.install_tools)
        self.usb_setup_button = QPushButton("Set up USB connection")
        self.usb_setup_button.setObjectName("PrimaryButton")
        self.usb_setup_button.clicked.connect(self.setup_usb)
        row = QHBoxLayout()
        row.addWidget(self.adb_label, 1)
        row.addWidget(self.usb_check_button)
        row.addWidget(self.install_tools_button)
        row.addWidget(self.usb_setup_button)
        self.usb_progress = QProgressBar()
        self.usb_progress.setRange(0, 1000)
        self.usb_progress.setVisible(False)
        self.usb_result = QLabel("")
        self.usb_result.setWordWrap(True)

        files_title = QLabel("Copy files to the Quest")
        files_title.setObjectName("SectionTitle")
        self.files_label = QLabel("")
        self.files_label.setWordWrap(True)
        self.copy_files_button = QPushButton("Copy files to Quest")
        self.copy_files_button.setToolTip(
            "Copies the results into the Quest's Movies folder, where headset players find them."
        )
        self.copy_files_button.clicked.connect(self.copy_files)
        frow = QHBoxLayout()
        frow.addWidget(self.files_label, 1)
        frow.addWidget(self.copy_files_button)

        lay = QVBoxLayout(w)
        lay.addWidget(self.usb_steps)
        lay.addLayout(row)
        lay.addWidget(self.usb_progress)
        lay.addWidget(self.usb_result)
        lay.addWidget(dc.hline())
        lay.addWidget(files_title)
        lay.addLayout(frow)
        lay.addStretch(1)
        if self.quest is None:
            lay.addWidget(dc.unavailable_label("USB helpers", self._quest_err or ""))
        self._update_files_label()
        return w

    def _update_files_label(self) -> None:
        if not hasattr(self, "files_label"):
            return
        existing = [f for f in self.files if f.is_file()]
        if existing:
            names = ", ".join(f.name for f in existing[:4]) + (" …" if len(existing) > 4 else "")
            self.files_label.setText(f"{len(existing)} file(s) from the selected job: {names}")
        else:
            self.files_label.setText("Select a finished job in the queue first, then open 'Send to Quest'.")
        self.copy_files_button.setEnabled(bool(existing) and self.quest is not None)

    def check_usb(self) -> None:
        """Look for adb and connected headsets (background thread)."""
        if self.quest is None:
            self.adb_label.setText("USB helpers are not available.")
            for b in (self.install_tools_button, self.usb_setup_button, self.usb_check_button):
                b.setEnabled(False)
            return
        self._usb_checked_once = True
        self.adb_label.setText("Looking for USB tools (adb)…")
        q = self.quest

        def work() -> tuple[str | None, list[str]]:
            adb = q.find_adb()
            return adb, (list(q.adb_devices()) if adb else [])

        self.tasks.start(
            work,
            pass_progress=False,
            pass_cancel=False,
            on_result=self._usb_checked,
            on_error=lambda e: self.adb_label.setText(f"Could not check USB: {e}"),
            name="adb-check",
        )

    def _usb_checked(self, res: tuple[str | None, list[str]]) -> None:
        self.adb_path, self.devices = res
        if not self.adb_path:
            self.adb_label.setText("USB tools (adb) are not installed.")
        elif not self.devices:
            self.adb_label.setText(
                "USB tools found. No headset connected - plug in the Quest, put it on and allow USB debugging."
            )
        else:
            self.adb_label.setText(f"Headset connected: {', '.join(self.devices)}")
        self.install_tools_button.setVisible(not self.adb_path)
        self.usb_setup_button.setEnabled(bool(self.adb_path))

    def install_tools(self) -> None:
        if self.quest is None:
            return
        if not util.ask_yes_no(
            self,
            "Install USB tools",
            "Download Google's Android SDK Platform-Tools (adb, about 7 MB)?",
            informative="By downloading you accept the Android SDK Platform-Tools license terms "
            "(https://developer.android.com/studio/releases/platform-tools).",
            default_yes=True,
        ):
            return
        self.usb_progress.setVisible(True)
        self.usb_progress.setValue(0)
        self.install_tools_button.setEnabled(False)
        self._long_tasks.add("install")

        def finished() -> None:
            self._long_tasks.discard("install")
            self.install_tools_button.setEnabled(True)

        self.tasks.start(
            self.quest.install_platform_tools,
            on_progress=self._usb_progress,
            on_result=self._tools_installed,
            on_error=self._usb_error,
            on_cancelled=lambda: self.usb_progress.setVisible(False),
            on_finished=finished,
            name="install-adb",
        )

    def _usb_progress(self, frac: float, msg: str) -> None:
        self.usb_progress.setValue(int(frac * 1000))
        if msg:
            self.usb_result.setText(msg)

    def _tools_installed(self, path: Any) -> None:
        self.usb_progress.setVisible(False)
        self.usb_result.setText(f"USB tools installed ({path}).")
        self.check_usb()

    def _usb_error(self, exc: BaseException) -> None:
        self.usb_progress.setVisible(False)
        msg, hint = util.error_text(exc)
        self.usb_result.setText(f"Failed: {msg}" + (f" {hint}" if hint else ""))

    def setup_usb(self) -> None:
        if self.quest is None:
            return
        if not self.ctl.running:  # the page must be served before the headset opens it
            self._usb_after_start = True
            self.usb_result.setText("Starting the live viewer first…")
            self.start_server()
            return
        if self._show_http_problem():  # the http (USB) listener did not start
            return
        port = self._http_port()
        q = self.quest
        scene = self.focus_scene

        def work() -> str:
            if not q.adb_reverse(port):
                return (
                    "Could not connect over USB. Check the cable, that Developer Mode is on and that you "
                    "allowed USB debugging in the headset."
                )
            url = f"http://localhost:{port}/"
            if scene:
                from urllib.parse import quote

                url += f"view.html?scene={quote(scene)}"
            if q.open_url_on_quest(url):
                return f"Done! The viewer is opening in the Quest Browser ({url})."
            return f"USB connection ready. In the Quest Browser open {url}"

        self.usb_result.setText("Setting up the USB connection…")
        self.tasks.start(
            work,
            pass_progress=False,
            pass_cancel=False,
            on_result=self.usb_result.setText,
            on_error=self._usb_error,
            name="adb-reverse",
        )

    def copy_files(self) -> None:
        if self.quest is None:
            return
        files = [f for f in self.files if f.is_file()]
        if not files:
            return
        q = self.quest
        adb = self.adb_path

        def work(progress=None, cancel=None) -> str:  # noqa: ANN001
            if sys.platform == "win32":
                if q.mtp_copy(files, "Movies", progress=progress, cancel=cancel):
                    return f"Copied {len(files)} file(s) to the Quest's Movies folder."
            if adb or q.find_adb():
                ok = all(q.adb_push(f, "/sdcard/Movies", progress=progress, cancel=cancel) for f in files)
                if ok:
                    return f"Copied {len(files)} file(s) to the Quest's Movies folder (USB debugging)."
            reason = ""
            last_error = getattr(q, "last_error", None)
            if callable(last_error):
                try:
                    reason = str(last_error() or "").strip()
                except Exception:  # pragma: no cover - defensive
                    reason = ""
            explorer = (
                "You can also copy the files in File Explorer: This PC → Quest 3 → Internal shared "
                "storage → Movies."
            )
            if reason:
                return f"Could not copy the files: {reason}\n{explorer}"
            return (
                "Could not copy the files. Connect the Quest with a USB data cable, put it on and choose "
                f"'Allow' when it asks to allow access to data. {explorer}"
            )

        self.usb_progress.setVisible(True)
        self.usb_progress.setValue(0)
        self.copy_files_button.setEnabled(False)
        self.usb_result.setText("Copying…")
        self._long_tasks.add("copy")

        def finished() -> None:
            self._long_tasks.discard("copy")
            self._update_files_label()

        def cancelled() -> None:
            self.usb_progress.setVisible(False)
            self.usb_result.setText("Copying was stopped.")

        self.tasks.start(
            work,
            on_progress=self._usb_progress,
            on_result=lambda msg: (self.usb_progress.setVisible(False), self.usb_result.setText(msg)),
            on_error=self._usb_error,
            on_cancelled=cancelled,
            on_finished=finished,
            name="copy-to-quest",
        )

    # ================================================================= library tab
    def _build_library(self) -> QWidget:
        w = QWidget()
        info = QLabel(
            "Scenes in the live viewer's gallery (3D scenes, objects, VR pictures). Deleting a scene "
            "here does not delete the files in your output folder."
        )
        info.setWordWrap(True)
        info.setObjectName("Muted")
        self.scene_list = QListWidget()
        self.scene_list.setIconSize(QSize(96, 64))
        self.scene_list.currentItemChanged.connect(lambda *_: self._update_scene_buttons())
        self.show_scene_button = QPushButton("Show address / QR")
        self.show_scene_button.clicked.connect(self._show_selected_scene)
        self.open_scene_button = QPushButton("Open on this PC")
        self.open_scene_button.clicked.connect(self._open_selected_scene)
        self.delete_scene_button = QPushButton("Delete")
        self.delete_scene_button.setObjectName("DangerButton")
        self.delete_scene_button.clicked.connect(self.delete_selected_scene)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_library)
        row = QHBoxLayout()
        row.addWidget(self.show_scene_button)
        row.addWidget(self.open_scene_button)
        row.addStretch(1)
        row.addWidget(refresh)
        row.addWidget(self.delete_scene_button)
        lay = QVBoxLayout(w)
        lay.addWidget(info)
        lay.addWidget(self.scene_list, 1)
        lay.addLayout(row)
        if self.scene_mod is None:
            lay.addWidget(dc.unavailable_label("The scene library", self._scene_err or ""))
        return w

    def scenes(self) -> list[dict]:
        if self.scene_mod is None:
            return []
        try:
            return list(self.scene_mod.list_scenes(self.ctl.library()))
        except Exception as e:
            log.warning("Cannot list scenes: %s", e)
            return []

    def refresh_library(self) -> None:
        self.scene_list.clear()
        lib = self.ctl.library()
        for s in self.scenes():
            sid = s.get("id", "")
            created = str(s.get("created", ""))[:16].replace("T", " ")
            item = QListWidgetItem(f"{s.get('title') or sid}\n{s.get('mode', '')} · {created}")
            item.setData(Qt.ItemDataRole.UserRole, sid)
            thumb = s.get("thumbnail")
            if thumb:
                local = lib / sid / Path(str(thumb)).name
                if local.is_file():
                    try:
                        item.setIcon(QIcon(util.np_to_pixmap(util.load_thumbnail(local, 96))))
                    except Exception as e:  # unreadable thumbnail: list the scene anyway
                        log.debug("Thumbnail of %s: %s", sid, e)
            self.scene_list.addItem(item)
        if self.scene_list.count() == 0:
            item = QListWidgetItem("No scenes yet. Create a 3D Scene, Person/Object or VR picture first.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.scene_list.addItem(item)
        self._select_scene(self.focus_scene)
        self._update_scene_buttons()

    def _selected_scene(self) -> str | None:
        it = self.scene_list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    def _select_scene(self, scene_id: str | None) -> None:
        if not scene_id:
            return
        for i in range(self.scene_list.count()):
            if self.scene_list.item(i).data(Qt.ItemDataRole.UserRole) == scene_id:
                self.scene_list.setCurrentRow(i)
                break

    def _update_scene_buttons(self) -> None:
        on = bool(self._selected_scene())
        for b in (self.show_scene_button, self.open_scene_button, self.delete_scene_button):
            b.setEnabled(on)

    def _show_selected_scene(self) -> None:
        sid = self._selected_scene()
        if sid:
            if not self.ctl.running:
                self.start_server()
            self.set_focus_scene(sid)

    def _open_selected_scene(self) -> None:
        sid = self._selected_scene()
        if not sid:
            return
        if not self.ctl.running:
            self.start_server()
            return
        url = self.ctl.scene_url(sid, base=self.ctl.local_url() or None)
        if url:
            util.open_url(url)

    def delete_selected_scene(self) -> bool:
        sid = self._selected_scene()
        if not sid or self.scene_mod is None:
            return False
        if not util.ask_yes_no(
            self,
            "Delete scene",
            f"Delete the scene '{sid}' from the viewer library?",
            informative="Files in your output folder are not affected.",
        ):
            return False
        try:
            self.scene_mod.delete_scene(sid, self.ctl.library())
        except Exception as e:
            util.show_error(self, "Delete scene", e)
            return False
        if self.focus_scene == sid:
            self.focus_scene = None
            if self.ctl.running:
                self._update_focus()
        self.refresh_library()
        return True

    # ================================================================= closing
    def busy(self) -> bool:
        """True while files are copied to the headset or the USB tools are installed."""
        return bool(self._long_tasks)

    def done(self, result: int) -> None:  # noqa: D401 - QDialog API
        if self._long_tasks:
            what = "Copying files to the Quest" if "copy" in self._long_tasks else "Installing the USB tools"
            if not util.ask_yes_no(
                self, "Still working", f"{what} is not finished yet.", informative="Stop it and close?"
            ):
                return
            self.tasks.cancel_all()
            self._long_tasks.clear()
        for sig, slot in (
            (self.ctl.started, self._server_started),
            (self.ctl.stopped, self._server_stopped),
            (self.ctl.failed, self._server_failed),
            (self.ctl.busyChanged, self._update_server_ui),
        ):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self.tasks.shutdown(DIALOG_CLOSE_WAIT_MS)
        super().done(result)
