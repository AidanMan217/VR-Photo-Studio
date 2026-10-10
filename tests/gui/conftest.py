"""Fixtures for GUI tests: offscreen Qt, a contract-conformant fake JobQueue, fake
runners, message-box stubs (nothing ever blocks), sample media and fake backends.

Helpers live in ``gui_helpers.py`` (importable from the test modules)."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from gui_helpers import FakeJobQueue, FakeServer, Prompts, fake_runner, make_image, make_video

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def settings(tmp_path, isolated_home):
    """Settings in the isolated data root with outputs in tmp (never ~/Pictures)."""
    from vrphoto.core.config import Settings, set_settings

    s = Settings()
    s.output_dir = str(tmp_path / "outputs")
    s.https_port = 18443
    s.http_port = 18080
    s.save()
    set_settings(s)
    return s


@pytest.fixture(autouse=True)
def prompts(monkeypatch) -> Prompts:
    """Never block on a modal dialog in tests."""
    from vrphoto.gui import util
    from vrphoto.gui.dialogs import common as dc

    p = Prompts()
    monkeypatch.setattr(util, "ask_yes_no", p.ask)
    monkeypatch.setattr(util, "show_error", p.error)
    monkeypatch.setattr(util, "show_info", p.info)
    monkeypatch.setattr(util, "open_path", lambda path: True)
    monkeypatch.setattr(util, "open_url", lambda url: True)
    monkeypatch.setattr(dc, "confirm_license", p.license)
    return p


@pytest.fixture
def heuristic_engine():
    """Preview engine using the instant no-model depth estimator."""
    from vrphoto.depth.base import get_estimator
    from vrphoto.gui.preview import PreviewEngine

    return PreviewEngine(lambda name, device: get_estimator("heuristic"))


@pytest.fixture
def fake_server_factory():
    FakeServer.instances.clear()
    return FakeServer


@pytest.fixture
def media(tmp_path) -> dict[str, Path]:
    d = tmp_path / "media"
    out = {
        "a": make_image(d / "a.jpg", seed=1),
        "b": make_image(d / "b.png", seed=2),
        "c": make_image(d / "sub" / "c.jpg", seed=3),
        "txt": d / "notes.txt",
        "dir": d,
    }
    out["txt"].write_text("not an image")
    vid = make_video(d / "clip.mp4")
    if vid is not None:
        out["video"] = vid
    return out


@pytest.fixture
def make_window(qtbot, settings, heuristic_engine, fake_server_factory):
    """Factory: MainWindow wired to a FakeJobQueue with the given runner."""
    from vrphoto.gui.main_window import MainWindow
    from vrphoto.gui.state import GuiState

    created: list[Any] = []

    def factory(
        runner: Callable[..., Any] = fake_runner, *, state: GuiState | None = None, queue: Any = None
    ):
        q = queue if queue is not None else FakeJobQueue(runner)
        w = MainWindow(
            settings,
            queue=q,
            preview_engine=heuristic_engine,
            server_factory=fake_server_factory,
            state=state or GuiState(),
        )
        qtbot.addWidget(w)
        w.show()
        created.append((w, q))
        return w, q

    yield factory
    for w, q in created:
        try:
            if not w._shut_down:
                w.shutdown()
        except RuntimeError:
            pass
        q.shutdown(wait=True)
