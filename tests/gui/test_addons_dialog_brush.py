"""Add-ons dialog: "Test Brush on this PC" on the installed Capture card (SHARP 360 §4.5)."""

from __future__ import annotations

import sys
import threading
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

pytestmark = pytest.mark.gui


@dataclass
class _Manifest:
    id: str
    title: str
    description: str = "An add-on."
    experimental: bool = False
    license: str = "MIT"
    noncommercial: bool = False
    region_restricted: list = field(default_factory=list)
    min_vram_gb: float = 4.0
    disk_gb: float = 2.0
    homepage: str = ""
    hf_gated: bool = False
    kind: str = "binary"

    def notices(self) -> list[str]:
        return []

    def requires_license_ack(self) -> bool:
        return False


def _fake_manager(tmp_path, installed: set[str]) -> types.ModuleType:
    mods = {"capture": _Manifest("capture", "Capture to 3D"), "trellis": _Manifest("trellis", "TRELLIS")}
    m = types.ModuleType("vrphoto.addons.manager")
    m.list_addons = lambda include_experimental=False: list(mods.values())
    m.get_manifest = lambda aid: mods[aid]
    m.status = lambda aid: {
        "installed": aid in installed,
        "path": str(tmp_path / aid),
        "supported": True,
        "partial": False,
        "update_available": False,
    }
    m.install = lambda aid, **kw: installed.add(aid)
    m.uninstall = lambda aid: installed.discard(aid)
    m.install_log_path = lambda aid: tmp_path / aid / "install.log"
    return m


@pytest.fixture
def make_dialog(qtbot, settings, monkeypatch, tmp_path):
    from vrphoto.gui.dialogs import addons_dialog

    monkeypatch.setattr(
        addons_dialog, "detect_gpu", lambda: {"name": "GTX 1080", "vram_mb": 8192, "nvidia": True}
    )

    def make(installed: set[str]):
        monkeypatch.setitem(sys.modules, "vrphoto.addons.manager", _fake_manager(tmp_path, installed))
        dlg = addons_dialog.AddonsDialog(settings=settings)
        qtbot.addWidget(dlg)
        dlg.show()
        qtbot.waitUntil(lambda: not dlg.status_pending, timeout=5000)
        return dlg

    return make


def _result(ok: bool) -> dict[str, Any]:
    r: dict[str, Any] = {
        "ok": ok,
        "sh0": ok,
        "masks": ok,
        "steps_per_s": 24.0 if ok else None,
        "seconds": 12.5,
        "splats": 2000 if ok else 0,
        "steps": 300,
        "brush_version": "0.3.0",
        "attempts": [],
        "date": "2026-10-12T10:00:00",
        "log": "brush_test.log",
    }
    if not ok:
        r["error"] = "Brush training failed (exit code 101)."
    return r


def test_test_button_only_on_the_installed_capture_card(make_dialog):
    dlg = make_dialog(set())
    assert not dlg.cards["capture"].test_button.isVisible()
    assert not dlg.cards["trellis"].test_button.isVisible()
    dlg.close()
    dlg = make_dialog({"capture", "trellis"})
    card = dlg.cards["capture"]
    assert card.test_button.isVisible() and card.test_button.isEnabled()
    assert card.test_button.text() == "Test Brush on this PC"
    assert "Not tested yet" in card.test_button.toolTip()
    assert not dlg.cards["trellis"].test_button.isVisible()
    dlg.close()


@pytest.mark.parametrize("ok", [True, False])
def test_brush_test_shows_the_verdict_and_remembers_it(
    make_dialog, monkeypatch, prompts, settings, qtbot, ok
):
    from vrphoto.addons import capture
    from vrphoto.core.config import Settings

    gate = threading.Event()
    seen: dict[str, Any] = {}

    def fake_selftest(progress=None, cancel=None):
        seen["thread"] = threading.current_thread() is not threading.main_thread()
        progress(0.5, "Testing Brush (step 150/300)")
        gate.wait(5)
        return _result(ok)

    monkeypatch.setattr(capture, "brush_selftest", fake_selftest)
    dlg = make_dialog({"capture"})
    card = dlg.cards["capture"]
    card.test_button.click()
    assert dlg.active_id == "capture" and dlg.testing
    assert card.cancel_button.isVisible() and not card.uninstall_button.isEnabled()
    assert not card.test_button.isEnabled()  # one task at a time
    assert not dlg.install("trellis")
    qtbot.waitUntil(lambda: "step 150/300" in card.progress_label.text(), timeout=5000)
    gate.set()
    qtbot.waitUntil(lambda: dlg.active_id is None, timeout=5000)
    assert seen["thread"]  # off the UI thread
    shown = prompts.infos if ok else prompts.errors
    assert len(shown) == 1 and "Brush test" in shown[0]
    assert ("Brush works on this PC" in shown[0]) is ok and ("could not train" in shown[0]) is (not ok)
    assert settings.brush_test["ok"] is ok
    assert Settings.load().brush_test == _result(ok)  # saved to settings.json
    assert ("Brush works" in card.test_button.toolTip()) is ok
    assert card.test_button.isEnabled() and not card.cancel_button.isVisible()
    dlg.close()


def test_brush_test_can_be_cancelled(make_dialog, monkeypatch, prompts, settings, qtbot):
    from vrphoto.addons import capture

    started = threading.Event()

    def slow_selftest(progress=None, cancel=None):
        started.set()
        for _ in range(500):
            cancel.check()
            threading.Event().wait(0.01)
        raise AssertionError("not cancelled")

    monkeypatch.setattr(capture, "brush_selftest", slow_selftest)
    dlg = make_dialog({"capture"})
    assert dlg.test_brush("capture")
    assert started.wait(5)
    dlg.cards["capture"].cancel_button.click()
    qtbot.waitUntil(lambda: dlg.active_id is None, timeout=5000)
    assert "cancelled" in dlg.cards["capture"].progress_label.text()
    assert settings.brush_test == {} and not prompts.errors and not prompts.infos
    dlg.close()


def test_brush_test_errors_are_shown(make_dialog, monkeypatch, prompts, settings, qtbot):
    from vrphoto.addons import capture
    from vrphoto.core.errors import AddonNotInstalled

    def missing(progress=None, cancel=None):
        raise AddonNotInstalled("brush was not found.", hint="Install the Capture add-on.")

    monkeypatch.setattr(capture, "brush_selftest", missing)
    dlg = make_dialog({"capture"})
    assert dlg.test_brush("capture")
    qtbot.waitUntil(lambda: dlg.active_id is None, timeout=5000)
    assert any("brush was not found" in e for e in prompts.errors)
    assert settings.brush_test == {}  # nothing is remembered for an error
    dlg.close()


def test_brush_test_summary():
    from vrphoto.gui.dialogs.addons_dialog import brush_test_summary

    assert brush_test_summary(None) == "Not tested yet."
    assert brush_test_summary({}) == "Not tested yet."
    assert brush_test_summary(_result(True)) == "Last test 2026-10-12 10:00: Brush works, 24.0 steps/s."
    assert "could not train" in brush_test_summary(_result(False))
