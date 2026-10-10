"""Every dialog opens and closes with its backend monkeypatched (no network)."""

from __future__ import annotations

import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gui


# =========================================================================== Models
@pytest.fixture
def fake_download(monkeypatch):
    """Patch models.download so 'downloads' are instant and local."""
    from vrphoto.models import download

    state: dict[str, bool] = {}
    started = threading.Event()
    gate = threading.Event()
    gate.set()

    def ensure_model(model_id, *, progress=None, cancel=None, token=None, variant="default"):
        started.set()
        for i in range(5):
            if cancel is not None:
                cancel.check()
            if progress is not None:
                progress(i / 5, f"Downloading part {i + 1}")
            gate.wait(5)
        state[model_id] = True
        return Path("/fake") / model_id

    monkeypatch.setattr(download, "ensure_model", ensure_model)
    monkeypatch.setattr(
        download, "is_downloaded", lambda model_id, variant="default": state.get(model_id, False)
    )
    monkeypatch.setattr(download, "delete_model", lambda model_id: state.pop(model_id, None))
    return types.SimpleNamespace(state=state, started=started, gate=gate)


def test_models_dialog_lists_registry_grouped(qtbot, settings, fake_download):
    from vrphoto.gui.dialogs.models_dialog import COL_STATUS, ModelsDialog
    from vrphoto.models.registry import list_specs

    dlg = ModelsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    visible = [s for s in list_specs() if not s.experimental]
    assert set(dlg.items) == {s.id for s in visible}
    tasks = {dlg.tree.topLevelItem(i).text(0) for i in range(dlg.tree.topLevelItemCount())}
    assert any("Depth" in t for t in tasks) and any("Look behind" in t for t in tasks)
    assert dlg.items["da2-small"].text(COL_STATUS) == "Not downloaded"
    nc = [s.id for s in visible if s.noncommercial]
    assert nc, "registry has non-commercial models"
    lic = dlg.tree.itemWidget(dlg.items[nc[0]], 2)
    assert "NON-COMMERCIAL" in " ".join(w.text() for w in lic.findChildren(type(dlg.progress_label)))
    dlg.close()


def test_models_dialog_download_and_delete(qtbot, settings, fake_download, prompts):
    from vrphoto.gui.dialogs.models_dialog import COL_STATUS, ModelsDialog

    dlg = ModelsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    dl, rm = dlg.buttons["da2-small"]
    assert dl.isEnabled() and not rm.isEnabled()
    dl.click()
    qtbot.waitUntil(lambda: dlg.active_id is None, timeout=5000)
    assert fake_download.state["da2-small"]
    assert dlg.items["da2-small"].text(COL_STATUS) == "Downloaded"
    assert rm.isEnabled() and not dl.isEnabled()
    assert dlg.delete("da2-small")
    assert dlg.items["da2-small"].text(COL_STATUS) == "Not downloaded"
    assert prompts.licenses == []  # Apache-2.0 model: no prompt
    dlg.close()


def test_models_dialog_noncommercial_needs_acceptance(qtbot, settings, fake_download, prompts):
    from vrphoto.gui.dialogs.models_dialog import ModelsDialog

    dlg = ModelsDialog(settings=settings)
    qtbot.addWidget(dlg)
    prompts.license_answer = False
    assert not dlg.download("da2-large")
    assert prompts.licenses and not settings.license_accepted("da2-large")
    prompts.license_answer = True
    assert dlg.download("da2-large")
    qtbot.waitUntil(lambda: dlg.active_id is None, timeout=5000)
    assert settings.license_accepted("da2-large")
    from vrphoto.core.config import Settings

    assert "da2-large" in Settings.load().accepted_licenses  # persisted
    dlg.close()


def test_models_dialog_cancel_download(qtbot, settings, fake_download, prompts):
    from vrphoto.gui.dialogs.models_dialog import ModelsDialog

    fake_download.gate.clear()
    dlg = ModelsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    assert dlg.download("lama")
    assert fake_download.started.wait(5)
    assert dlg.cancel_button.isVisible()
    dlg.cancel_download()
    fake_download.gate.set()
    qtbot.waitUntil(lambda: dlg.active_id is None, timeout=5000)
    assert "cancelled" in dlg.progress_label.text()
    assert not fake_download.state.get("lama")
    dlg.close()


def test_models_dialog_close_while_downloading_asks(qtbot, settings, fake_download, prompts):
    from vrphoto.gui.dialogs.models_dialog import ModelsDialog

    fake_download.gate.clear()
    dlg = ModelsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.download("migan")
    assert fake_download.started.wait(5)
    prompts.answer = False
    dlg.close()
    assert dlg.isVisible()
    prompts.answer = True
    fake_download.gate.set()
    dlg.close()
    assert not dlg.isVisible()


def test_models_dialog_backend_unavailable(qtbot, settings, monkeypatch):
    from vrphoto.gui.dialogs import common as dc
    from vrphoto.gui.dialogs.models_dialog import ModelsDialog

    monkeypatch.setattr(dc, "try_backend", lambda name: (None, "ImportError: broken install"))
    dlg = ModelsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    assert dlg.items == {}
    labels = [w.text() for w in dlg.findChildren(type(dlg.progress_label))]
    assert any("not available" in t for t in labels)
    dlg.close()


# =========================================================================== Add-ons
@dataclass
class FakeManifest:
    id: str
    title: str
    description: str = "Makes 3D models."
    experimental: bool = False
    license: str = "MIT"
    noncommercial: bool = False
    region_restricted: list = field(default_factory=list)
    min_vram_gb: float = 8.0
    disk_gb: float = 2.0
    homepage: str = "https://example.org"
    hf_gated: bool = False

    def notices(self):
        out = []
        if self.noncommercial:
            out.append(f"Non-commercial use only ({self.license}).")
        if self.region_restricted:
            out.append("The license does not apply in: the European Union.")
        return out

    def requires_license_ack(self):
        return bool(self.noncommercial or self.region_restricted or self.hf_gated)


def make_fake_manager(tmp_path):
    mods = {
        "trellis": FakeManifest("trellis", "TRELLIS", noncommercial=True, license="MIT + nvdiffrast NC"),
        "capture": FakeManifest("capture", "Capture to 3D", min_vram_gb=4),
        "hunyuan": FakeManifest("hunyuan", "Hunyuan3D", experimental=True, region_restricted=["DE", "FR"]),
        "gated": FakeManifest("gated", "Gated", hf_gated=True),
    }
    state: dict[str, bool] = {}
    calls: list[tuple] = []
    gate = threading.Event()
    gate.set()
    m = types.ModuleType("vrphoto.addons.manager")

    def list_addons(include_experimental=False):
        return [x for x in mods.values() if include_experimental or not x.experimental]

    def get_manifest(aid):
        from vrphoto.core.errors import AddonError

        if aid not in mods:
            raise AddonError(f"Unknown add-on '{aid}'.")
        return mods[aid]

    def status(aid):
        d = tmp_path / "addons" / aid
        return {
            "installed": state.get(aid, False),
            "path": str(d),
            "size_mb": 1500.0 if state.get(aid) else 0,
            "supported": True,
            "partial": False,
            "update_available": False,
        }

    def install(aid, *, progress=None, cancel=None, hf_token=None):
        calls.append(("install", aid, hf_token))
        (tmp_path / "addons" / aid).mkdir(parents=True, exist_ok=True)
        (tmp_path / "addons" / aid / "install.log").write_text("step 1\nstep 2\n")
        for i in range(4):
            if cancel is not None:
                cancel.check()
            if progress is not None:
                progress(i / 4, f"Step {i + 1}")
            gate.wait(5)
        state[aid] = True

    def uninstall(aid):
        calls.append(("uninstall", aid))
        state[aid] = False

    m.list_addons = list_addons
    m.get_manifest = get_manifest
    m.status = status
    m.install = install
    m.uninstall = uninstall
    m.install_log_path = lambda aid: tmp_path / "addons" / aid / "install.log"
    m.state, m.calls, m.gate = state, calls, gate  # type: ignore[attr-defined]
    return m


@pytest.fixture
def fake_manager(monkeypatch, tmp_path):
    m = make_fake_manager(tmp_path)
    monkeypatch.setitem(sys.modules, "vrphoto.addons.manager", m)
    from vrphoto.gui.dialogs import addons_dialog

    monkeypatch.setattr(
        addons_dialog,
        "detect_gpu",
        lambda: {"name": "NVIDIA GeForce RTX 4070", "vram_mb": 12282, "nvidia": True},
    )
    return m


def test_addons_dialog_cards_and_requirements(qtbot, settings, fake_manager):
    from vrphoto.gui.dialogs.addons_dialog import AddonsDialog

    dlg = AddonsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    assert set(dlg.cards) == {"trellis", "capture", "gated"}  # experimental hidden
    qtbot.waitUntil(lambda: dlg.gpu is not None, timeout=5000)
    assert "RTX 4070" in dlg.gpu_label.text()
    req = dlg.cards["trellis"].req_label.text()
    assert "12 GB" in req and "needs 8 GB" in req
    assert dlg.cards["trellis"].notes_label.isVisible()
    dlg.exp_check.setChecked(True)
    assert "hunyuan" in dlg.cards and settings.show_experimental
    dlg.close()


def test_addons_dialog_install_and_uninstall(qtbot, settings, fake_manager, prompts):
    from vrphoto.gui.dialogs.addons_dialog import AddonsDialog

    settings.hf_token = "hf_secret"
    dlg = AddonsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    card = dlg.cards["capture"]
    assert card.install_button.isVisible() and not card.uninstall_button.isVisible()
    card.install_button.click()
    qtbot.waitUntil(lambda: dlg.active_id is None, timeout=5000)
    assert fake_manager.calls[0] == ("install", "capture", "hf_secret")
    assert "Installed" in card.state_label.text()
    assert card.uninstall_button.isVisible()
    assert "step 2" in card.log_view.toPlainText()
    card.uninstall_button.click()
    qtbot.waitUntil(lambda: dlg.active_id is None and fake_manager.calls[-1][0] == "uninstall", timeout=5000)
    qtbot.waitUntil(lambda: card.install_button.isVisible(), timeout=5000)
    dlg.close()


def test_addons_dialog_license_prompt(qtbot, settings, fake_manager, prompts):
    from vrphoto.gui.dialogs.addons_dialog import AddonsDialog

    dlg = AddonsDialog(settings=settings)
    qtbot.addWidget(dlg)
    prompts.license_answer = False
    assert not dlg.install("trellis")
    assert prompts.licenses == ["TRELLIS"] and fake_manager.calls == []
    prompts.license_answer = True
    assert dlg.install("trellis")
    qtbot.waitUntil(lambda: dlg.active_id is None, timeout=5000)
    assert settings.license_accepted("trellis")
    dlg.close()


def test_addons_dialog_gated_needs_token(qtbot, settings, fake_manager, prompts):
    from vrphoto.gui.dialogs.addons_dialog import AddonsDialog

    settings.hf_token = ""
    dlg = AddonsDialog(settings=settings)
    qtbot.addWidget(dlg)
    assert not dlg.install("gated")
    assert any("Hugging Face" in e for e in prompts.errors)
    dlg.close()


def test_addons_dialog_cancel_install(qtbot, settings, fake_manager, prompts):
    from vrphoto.gui.dialogs.addons_dialog import AddonsDialog

    fake_manager.gate.clear()
    dlg = AddonsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    assert dlg.install("capture")
    qtbot.waitUntil(lambda: dlg.cards["capture"].cancel_button.isVisible(), timeout=5000)
    dlg.cards["capture"].cancel_button.click()
    fake_manager.gate.set()
    qtbot.waitUntil(lambda: dlg.active_id is None, timeout=5000)
    assert "cancelled" in dlg.cards["capture"].progress_label.text()
    assert not fake_manager.state.get("capture")
    dlg.close()


def test_addons_dialog_low_vram_warns(qtbot, settings, fake_manager, prompts, monkeypatch):
    from vrphoto.gui.dialogs import addons_dialog

    monkeypatch.setattr(
        addons_dialog, "detect_gpu", lambda: {"name": "Small GPU", "vram_mb": 4096, "nvidia": True}
    )
    dlg = addons_dialog.AddonsDialog(settings=settings)
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: dlg.gpu is not None, timeout=5000)
    prompts.answer = False
    settings.accept_license("trellis")
    assert not dlg.install("trellis")
    assert any("needs 8 GB" in q for q in prompts.questions)
    dlg.close()


def test_requirement_notes():
    from vrphoto.gui.dialogs.addons_dialog import requirement_notes

    m = FakeManifest("x", "X", min_vram_gb=12, disk_gb=20)
    lv = dict(
        (lvl, t) for lvl, t in requirement_notes(m, {"name": "AMD", "vram_mb": 16000, "nvidia": False}, 100)
    )
    assert "no NVIDIA GPU" in lv["warn"]
    levels = [lvl for lvl, _ in requirement_notes(m, {"name": "RTX", "vram_mb": 12288, "nvidia": True}, 5.0)]
    assert levels == ["ok", "warn"]
    assert [lvl for lvl, _ in requirement_notes(m, None, None)] == ["info", "info"]


def test_requirement_notes_binary_addons_accept_any_gpu():
    from vrphoto.gui.dialogs.addons_dialog import license_notices, requirement_notes

    m = FakeManifest("capture", "Capture", min_vram_gb=4, disk_gb=2)
    m.kind = "binary"  # type: ignore[attr-defined]
    notes = requirement_notes(m, {"name": "AMD Radeon", "vram_mb": 8192, "nvidia": False}, 50)
    assert [lvl for lvl, _ in notes] == ["ok", "ok"]
    notes = requirement_notes(m, {"name": None, "vram_mb": None, "nvidia": False}, 50)
    assert notes[0][0] == "info" and "could be detected" in notes[0][1]

    class WithHw(FakeManifest):
        def notices(self):
            return [
                "Needs a GPU with at least 4 GB VRAM.",
                "Download/disk space: about 2 GB.",
                "Non-commercial use only (X).",
            ]

    assert license_notices(WithHw("x", "X")) == ["Non-commercial use only (X)."]


def test_addons_dialog_with_real_manifests(qtbot, settings, monkeypatch):
    """The real manager lists the bundled manifests (nothing is installed)."""
    pytest.importorskip("vrphoto.addons.manager")
    from vrphoto.gui.dialogs import addons_dialog

    monkeypatch.setattr(addons_dialog, "detect_gpu", lambda: {"name": None, "vram_mb": None, "nvidia": False})
    dlg = addons_dialog.AddonsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    assert "capture" in dlg.cards
    assert all(not c.status.get("installed") for c in dlg.cards.values())
    dlg.close()


def test_addons_dialog_backend_unavailable(qtbot, settings, monkeypatch):
    monkeypatch.setitem(sys.modules, "vrphoto.addons.manager", None)
    from vrphoto.gui.dialogs import addons_dialog

    monkeypatch.setattr(addons_dialog, "detect_gpu", lambda: {})
    dlg = addons_dialog.AddonsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    assert dlg.manager is None and dlg.cards == {}
    dlg.close()


# =========================================================================== Send to Quest
@pytest.fixture
def fake_quest(monkeypatch):
    from vrphoto.webxr import net, quest

    calls: list[tuple] = []
    monkeypatch.setattr(quest, "find_adb", lambda: "/tools/adb")
    monkeypatch.setattr(quest, "adb_devices", lambda: ["2G0YC1ZF8B07WD"])
    monkeypatch.setattr(
        quest, "adb_reverse", lambda port, serial=None: calls.append(("reverse", port)) or True
    )
    monkeypatch.setattr(
        quest, "open_url_on_quest", lambda url, serial=None: calls.append(("open", url)) or True
    )
    monkeypatch.setattr(
        quest, "mtp_copy", lambda files, sub="Movies", **kw: calls.append(("mtp", len(files))) or True
    )
    monkeypatch.setattr(
        quest,
        "adb_push",
        lambda f, d="/sdcard/Movies", serial=None, **kw: calls.append(("push", Path(f).name)) or True,
    )
    monkeypatch.setattr(
        quest,
        "install_platform_tools",
        lambda progress=None, cancel=None: (progress and progress(1.0, "done"), "/tools/adb")[1],
    )
    monkeypatch.setattr(
        net, "firewall_report", lambda port: {"status": "blocked", "reason": "No rule.", "profile": ""}
    )
    monkeypatch.setattr(net, "add_firewall_rule", lambda ports: calls.append(("fw", tuple(ports))) or True)
    return calls


@pytest.fixture
def controller(qtbot, settings, fake_server_factory):
    from vrphoto.gui.server_ctl import ServerController

    ctl = ServerController(settings, server_factory=fake_server_factory)
    yield ctl
    ctl.shutdown()


@pytest.fixture
def library_scene(settings):
    from vrphoto.webxr.scene import export_scene

    img = np.zeros((64, 128, 3), np.uint8)
    return export_scene(
        {
            "title": "Beach",
            "mode": "stereo",
            "stereo": {"media": "beach_LR.jpg", "kind": "image", "layout": "sbs"},
        },
        {"beach_LR.jpg": b"\xff\xd8\xff\xd9"},
        library=settings.library_dir,
        thumbnail=img,
    ).name


def test_quest_dialog_wifi(qtbot, settings, controller, fake_quest, library_scene, fake_server_factory):
    from vrphoto.gui.dialogs.quest_dialog import QuestDialog

    dlg = QuestDialog(controller=controller, settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitUntil(lambda: controller.running, timeout=5000)  # auto-started
    qtbot.waitUntil(lambda: dlg.url_label.text() == "https://192.168.1.5:8443/", timeout=5000)
    assert dlg.qr_label.pixmap() is not None and not dlg.qr_label.pixmap().isNull()
    assert "Proceed to 192.168.1.5" in dlg.steps.text()
    assert "10.0.0.2" in dlg.other_urls.text()
    qtbot.waitUntil(lambda: "BLOCKED" in dlg.fw_label.text(), timeout=5000)
    dlg.fw_button.click()
    qtbot.waitUntil(lambda: ("fw", (8443, 8080)) in fake_quest, timeout=5000)
    dlg.copy_url()
    from PySide6.QtGui import QGuiApplication

    assert QGuiApplication.clipboard().text() == "https://192.168.1.5:8443/"
    dlg.toggle_server()  # stop
    qtbot.waitUntil(lambda: not controller.running and dlg.url_label.text() == "", timeout=5000)
    assert fake_server_factory.instances[-1].stops == 1
    dlg.close()


def test_quest_dialog_focus_scene_and_library(
    qtbot, settings, controller, fake_quest, library_scene, prompts
):
    from vrphoto.gui.dialogs.quest_dialog import QuestDialog

    dlg = QuestDialog(controller=controller, settings=settings, focus_scene=library_scene)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitUntil(lambda: controller.running, timeout=5000)
    qtbot.waitUntil(lambda: library_scene in dlg.scene_label.text(), timeout=5000)
    assert dlg.current_url() == f"https://192.168.1.5:8443/view.html?scene={library_scene}"
    assert dlg.scene_list.count() == 1
    assert dlg._selected_scene() == library_scene
    assert not dlg.scene_list.item(0).icon().isNull()  # thumbnail
    prompts.answer = True
    assert dlg.delete_selected_scene()
    assert not (Path(settings.library_dir) / library_scene).exists()
    assert dlg.focus_scene is None
    assert dlg.scene_list.item(0).flags() == dlg.scene_list.item(0).flags().NoItemFlags
    dlg.close()


def test_quest_dialog_usb(qtbot, settings, controller, fake_quest, tmp_path, monkeypatch):
    from vrphoto.gui.dialogs import quest_dialog

    out = tmp_path / "x_LR.jpg"
    out.write_bytes(b"jpg")
    dlg = quest_dialog.QuestDialog(controller=controller, settings=settings, files=[out], focus_scene="beach")
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitUntil(lambda: "2G0YC1ZF8B07WD" in dlg.adb_label.text(), timeout=5000)
    assert dlg.usb_setup_button.isEnabled()
    qtbot.waitUntil(lambda: controller.running and not controller.busy, timeout=5000)
    controller.stop()
    qtbot.waitUntil(lambda: not controller.running and not controller.busy, timeout=5000)
    dlg.setup_usb()  # starts the viewer first, then sets up USB
    qtbot.waitUntil(lambda: "opening" in dlg.usb_result.text(), timeout=5000)
    port = int(settings.http_port)
    assert ("reverse", 8080) in fake_quest or ("reverse", port) in fake_quest
    assert any(c[0] == "open" and c[1].endswith("view.html?scene=beach") for c in fake_quest)
    assert dlg.copy_files_button.isEnabled()
    monkeypatch.setattr(quest_dialog.sys, "platform", "linux")
    dlg.copy_files()
    qtbot.waitUntil(lambda: "Copied 1 file" in dlg.usb_result.text(), timeout=5000)
    assert ("push", "x_LR.jpg") in fake_quest
    dlg.close()


def test_quest_dialog_install_usb_tools(qtbot, settings, controller, fake_quest, monkeypatch, prompts):
    from vrphoto.webxr import quest

    monkeypatch.setattr(quest, "find_adb", lambda: None)
    from vrphoto.gui.dialogs.quest_dialog import QuestDialog

    dlg = QuestDialog(controller=controller, settings=settings, autostart=False)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitUntil(lambda: "not installed" in dlg.adb_label.text(), timeout=5000)
    assert not dlg.install_tools_button.isHidden()  # (on a background tab)
    assert not dlg.usb_setup_button.isEnabled()
    monkeypatch.setattr(quest, "find_adb", lambda: "/tools/adb")
    dlg.install_tools()
    assert any("Platform-Tools" in q for q in prompts.questions)
    qtbot.waitUntil(lambda: "installed" in dlg.usb_result.text(), timeout=5000)
    qtbot.waitUntil(lambda: "Headset connected" in dlg.adb_label.text(), timeout=5000)
    assert not controller.running  # autostart=False
    dlg.close()


def test_quest_dialog_server_failure(qtbot, settings, fake_quest):
    from vrphoto.core.errors import VRPhotoError
    from vrphoto.gui.dialogs.quest_dialog import QuestDialog
    from vrphoto.gui.server_ctl import ServerController

    class Broken:
        running = False

        def start(self):
            raise VRPhotoError("Port 8443 is already in use.", hint="Close the other program.")

        def stop(self):
            pass

    ctl = ServerController(settings, server_factory=lambda s: Broken())
    dlg = QuestDialog(controller=ctl, settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitUntil(lambda: "could not start" in dlg.status_label.text(), timeout=5000)
    assert (
        "already in use" in dlg.status_label.text() and "Close the other program" in dlg.status_label.text()
    )
    assert not ctl.running and ctl.last_error
    dlg.close()
    ctl.shutdown()


def test_quest_dialog_backends_unavailable(qtbot, settings, controller, monkeypatch):
    from vrphoto.gui.dialogs import common as dc
    from vrphoto.gui.dialogs.quest_dialog import QuestDialog

    monkeypatch.setattr(dc, "try_backend", lambda name: (None, f"{name} missing"))
    dlg = QuestDialog(controller=controller, settings=settings, autostart=False)
    qtbot.addWidget(dlg)
    dlg.show()
    assert dlg.quest is None and dlg.net is None and dlg.scene_mod is None
    assert "not available" in dlg.adb_label.text()
    dlg.close()


def test_server_controller_real_server(qtbot, settings, tmp_path):
    """The default factory starts the real ViewerServer on free ports."""
    pytest.importorskip("vrphoto.webxr.server")
    from vrphoto.gui.server_ctl import ServerController

    settings.https_port = 0
    settings.http_port = 0
    ctl = ServerController(settings)
    with qtbot.waitSignals([ctl.started], timeout=30000, raising=False) as blocker:
        ctl.start()
    if not blocker.signal_triggered:
        pytest.skip(f"viewer server could not start here: {ctl.last_error}")
    try:
        urls = ctl.urls()
        assert urls["local"].startswith("http://localhost:")
        assert ctl.scene_url("abc", base=urls["local"]).endswith("/view.html?scene=abc")
    finally:
        ctl.shutdown()
    assert not ctl.running


# =========================================================================== Settings
def test_settings_dialog_saves(qtbot, settings, isolated_home):
    from vrphoto.core.config import Settings, get_settings
    from vrphoto.gui.dialogs.settings_dialog import SettingsDialog

    dlg = SettingsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    assert dlg.hf_token.echoMode() == dlg.hf_token.EchoMode.Password
    dlg.fp16.setChecked(True)
    dlg.max_px.setValue(2048)
    dlg.hf_token.setText("  hf_abc123  ")
    dlg.quest_model.setCurrentIndex(dlg.quest_model.findData("quest2"))
    dlg.video_encoder.setCurrentIndex(dlg.video_encoder.findData("nvenc"))
    dlg.theme.setCurrentIndex(dlg.theme.findData("light"))
    dlg.show_experimental.setChecked(True)
    dlg.device.setCurrentIndex(dlg.device.findData("cpu"))
    dlg.https_port.setValue(9443)
    dlg.output_dir.setText(str(isolated_home / "out"))
    with qtbot.waitSignal(dlg.saved, timeout=2000) as blocker:
        assert dlg.save()
    changed = blocker.args[0]
    assert {
        "use_fp16",
        "max_process_px",
        "hf_token",
        "quest_model",
        "video_encoder",
        "theme",
        "show_experimental",
        "device",
        "https_port",
        "output_dir",
    } <= changed
    on_disk = Settings.load()
    assert on_disk.use_fp16 and on_disk.max_process_px == 2048 and on_disk.hf_token == "hf_abc123"
    assert on_disk.quest_model == "quest2" and on_disk.video_encoder == "nvenc"
    assert on_disk.theme == "light" and on_disk.show_experimental and on_disk.device == "cpu"
    assert on_disk.https_port == 9443
    assert get_settings() is settings
    assert not dlg.isVisible()


def test_settings_dialog_validation(qtbot, settings):
    from vrphoto.gui.dialogs.settings_dialog import SettingsDialog

    dlg = SettingsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.http_port.setValue(dlg.https_port.value())
    assert not dlg.save()
    assert "different" in dlg.error_label.text()
    assert dlg.isVisible()
    dlg.reject()


def test_settings_dialog_token_toggle_and_devices(qtbot, settings, monkeypatch):
    from vrphoto.gui.dialogs import settings_dialog

    monkeypatch.setattr(
        settings_dialog.SettingsDialog,
        "_detect_devices",
        staticmethod(lambda: [("0", "NVIDIA GeForce RTX 4070 (12 GB)")]),
    )
    dlg = settings_dialog.SettingsDialog(settings=settings)
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.show_token.click()
    assert dlg.hf_token.echoMode() == dlg.hf_token.EchoMode.Normal
    qtbot.waitUntil(lambda: dlg.device.findData("0") >= 0, timeout=5000)
    assert "RTX 4070" in dlg.gpu_info.text()
    dlg.device.setCurrentIndex(dlg.device.findData("0"))
    assert dlg.values()["device"] == "0"
    dlg.reject()


def test_main_window_applies_saved_settings(make_window, qtbot, settings):
    from PySide6.QtWidgets import QApplication

    w, _ = make_window()
    dlg = w.open_settings()
    dlg.theme.setCurrentIndex(dlg.theme.findData("light"))
    dlg.show_experimental.setChecked(True)
    dlg.save()
    assert QApplication.instance().property("vrphoto_theme") == "light"
    w.modes.set_mode("object")
    assert "hunyuan" in w.modes.current_panel().form.editors["method"].values()
    from vrphoto.gui.theme import apply_theme

    apply_theme(QApplication.instance(), "dark")


# =========================================================================== About / Log
def test_about_dialog_shows_notices(qtbot):
    from vrphoto import __version__
    from vrphoto.gui.dialogs.about_dialog import AboutDialog, load_notices

    text, _ = load_notices()
    assert text.strip()
    dlg = AboutDialog()
    qtbot.addWidget(dlg)
    dlg.show()
    assert __version__ in dlg.windowTitle() or __version__ in dlg.system.toPlainText()
    assert dlg.notices.toPlainText().strip()
    qtbot.waitUntil(lambda: "Python" in dlg.system.toPlainText(), timeout=10000)  # filled in the background
    dlg.close()


def test_about_dialog_fallback_text(qtbot, monkeypatch, tmp_path):
    from vrphoto.gui.dialogs import about_dialog

    monkeypatch.setattr(about_dialog, "notices_candidates", lambda: [tmp_path / "missing.md"])
    text, md = about_dialog.load_notices()
    assert "THIRD_PARTY_NOTICES" in text and md


def test_notices_candidates_when_frozen(monkeypatch, tmp_path):
    from vrphoto.gui.dialogs import about_dialog

    exe = tmp_path / "VRPhotoStudio.exe"
    exe.write_bytes(b"")
    (tmp_path / "THIRD_PARTY_NOTICES.txt").write_text("Frozen notices", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)
    assert tmp_path / "THIRD_PARTY_NOTICES.txt" in about_dialog.notices_candidates()
    text, _ = about_dialog.load_notices()
    assert text == "Frozen notices"


def test_log_dialog_tails_file(qtbot, tmp_path):
    from vrphoto.gui.dialogs.log_dialog import LogDialog

    log = tmp_path / "vrphoto.log"
    log.write_text("".join(f"line {i}\n" for i in range(5000)), encoding="utf-8")
    dlg = LogDialog(path=log)
    qtbot.addWidget(dlg)
    dlg.show()
    text = dlg.view.toPlainText()
    assert "line 4999" in text and "line 0\n" not in text
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("NEW MESSAGE\n")
    dlg.refresh()
    assert "NEW MESSAGE" in dlg.view.toPlainText()
    dlg.auto.setChecked(False)
    assert not dlg.timer.isActive()
    dlg.open_folder()
    dlg.close()


def test_log_dialog_missing_file(qtbot, tmp_path):
    from vrphoto.gui.dialogs.log_dialog import LogDialog

    dlg = LogDialog(path=tmp_path / "nope.log")
    qtbot.addWidget(dlg)
    assert "empty" in dlg.view.toPlainText()
    dlg.close()


def test_tail_text(tmp_path):
    from vrphoto.gui.dialogs.common import tail_text

    p = tmp_path / "x.log"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    assert tail_text(p) == "a\nb\nc"
    assert tail_text(p, max_lines=2) == "b\nc"
    assert tail_text(tmp_path / "missing") == ""
