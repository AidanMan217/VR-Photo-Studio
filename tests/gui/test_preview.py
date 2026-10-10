"""Preview engine (depth / stereo / wiggle / parallax) and the preview tabs."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent

from vrphoto.gui.panels.preview import (
    TAB_DEPTH,
    TAB_ORIGINAL,
    TAB_PARALLAX,
    TAB_STEREO,
    TAB_WIGGLE,
    PreviewPanel,
)
from vrphoto.gui.preview import PREVIEW_LONG, LRUCache, PreviewEngine

pytestmark = pytest.mark.gui


class CountingEstimator:
    """Wraps the heuristic estimator and counts calls."""

    def __init__(self):
        from vrphoto.depth.base import get_estimator

        self.inner = get_estimator("heuristic")
        self.calls = 0
        self.id = "counting"

    def estimate(self, img, *, progress=None, cancel=None):
        self.calls += 1
        return self.inner.estimate(img, progress=progress, cancel=cancel)


class FailingEstimator:
    id = "broken"

    def estimate(self, img, *, progress=None, cancel=None):
        from vrphoto.core.errors import ModelUnavailable

        raise ModelUnavailable("Depth model could not be downloaded", hint="Check your internet connection.")


def test_lru_cache():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)  # evicts b (least recently used)
    assert "b" not in c and "a" in c and len(c) == 2
    c.clear()
    assert len(c) == 0 and c.get("a") is None


def test_engine_depth_is_cached_per_model(media):
    est = CountingEstimator()
    made: list[tuple[str, str]] = []

    def factory(name, device):
        made.append((name, device))
        return est

    eng = PreviewEngine(factory)
    dp = eng.depth(media["a"], "auto", "cpu")
    assert dp.disparity.shape == dp.image.shape[:2]
    assert dp.disparity.dtype == np.float32 and 0 <= dp.disparity.min() <= dp.disparity.max() <= 1
    assert eng.depth(media["a"], "auto", "cpu") is dp
    assert est.calls == 1
    eng.depth(media["a"], "da2-small", "cpu")
    assert est.calls == 2 and made == [("auto", "cpu"), ("da2-small", "cpu")]
    assert eng.cached_depth(media["a"], "auto") is dp
    col = eng.colorized(dp)
    assert col.shape == dp.image.shape and col.dtype == np.uint8


def test_engine_renders(media):
    from vrphoto.depth.base import get_estimator

    eng = PreviewEngine(lambda n, d: get_estimator("heuristic"))
    dp = eng.depth(media["a"])
    li = eng.layered(dp, near_m=1.0, far_m=10.0)
    assert max(li.K.width, li.K.height) <= PREVIEW_LONG
    assert eng.layered(dp, near_m=1.0, far_m=10.0) is li  # cached
    params = eng.stereo_params({"ipd_m": 0.065, "strength": 1.5, "convergence_m": None, "swap_eyes": True})
    assert params.ipd_m == 0.065 and params.strength == 1.5 and params.swap_eyes and params.mode == "flat"
    ana = eng.anaglyph(li, params)
    assert ana.shape == (li.K.height, li.K.width, 3)
    frames = eng.wiggle(li, params, 4)
    assert len(frames) == 4
    f0 = eng.parallax_frame(li, 0.0, 0.0)
    f1 = eng.parallax_frame(li, 1.0, 0.0, 0.05)
    assert f0.shape == f1.shape and not np.array_equal(f0, f1)


def test_stereo_params_ignore_junk():
    p = PreviewEngine.stereo_params({"ipd_m": "x", "strength": True, "convergence_m": 0})
    assert p.ipd_m == 0.063 and p.strength == 1.0 and p.convergence_m is None


def test_engine_video_first_frame(media):
    if "video" not in media:
        pytest.skip("OpenCV cannot write MP4 here")
    from vrphoto.depth.base import get_estimator

    eng = PreviewEngine(lambda n, d: get_estimator("heuristic"))
    src = eng.source(media["video"])
    assert src.ndim == 3 and src.shape[2] == 3


@pytest.fixture
def panel(qtbot, heuristic_engine):
    opts = {
        "common": {"depth_model": "auto", "device": "auto", "near_m": 1.2, "far_m": 15.0},
        "strength": 1.0,
    }
    p = PreviewPanel(engine=heuristic_engine, options_provider=lambda: opts)
    p.opts = opts  # type: ignore[attr-defined]
    qtbot.addWidget(p)
    p.resize(800, 600)
    p.show()
    yield p
    p.shutdown()


def test_preview_tabs_render(panel, media, qtbot):
    panel.set_path(media["a"])
    qtbot.waitUntil(lambda: panel.views[TAB_ORIGINAL].pixmap() is not None, timeout=5000)
    assert panel.views[TAB_DEPTH].pixmap() is None  # depth only on demand
    panel.set_current_tab(TAB_DEPTH)
    qtbot.waitUntil(lambda: panel.views[TAB_DEPTH].pixmap() is not None, timeout=10000)
    panel.set_current_tab(TAB_STEREO)
    qtbot.waitUntil(lambda: panel.views[TAB_STEREO].pixmap() is not None, timeout=10000)
    assert panel.views[TAB_STEREO].message() == ""
    panel.set_current_tab(TAB_WIGGLE)
    qtbot.waitUntil(lambda: len(panel._wiggle_frames) == 8, timeout=10000)
    assert panel.wiggle_timer.isActive()
    first = panel._wiggle_i
    qtbot.waitUntil(lambda: panel._wiggle_i != first, timeout=3000)
    panel.set_current_tab(TAB_PARALLAX)
    assert not panel.wiggle_timer.isActive()
    qtbot.waitUntil(lambda: panel.views[TAB_PARALLAX].pixmap() is not None, timeout=10000)
    assert panel.auto_timer.isActive()


def test_parallax_follows_mouse(panel, media, qtbot):
    panel.set_path(media["a"])
    panel.set_current_tab(TAB_PARALLAX)
    qtbot.waitUntil(lambda: panel.views[TAB_PARALLAX].pixmap() is not None, timeout=10000)
    view = panel.views[TAB_PARALLAX]
    got: list[tuple[float, float]] = []
    view.moved.connect(lambda dx, dy: got.append((dx, dy)))
    r = view.image_rect()
    pos = QPointF(r.right() - 1, r.center().y())
    ev = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        pos,
        pos,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mouseMoveEvent(ev)
    assert got and got[-1][0] == pytest.approx(1.0, abs=0.02) and abs(got[-1][1]) < 0.05
    assert view.hovering
    qtbot.waitUntil(
        lambda: panel._parallax_drawn is not None and panel._parallax_drawn[0] > 0.9, timeout=5000
    )
    assert panel.parallax_fps > 0
    view.leaveEvent(None)
    assert not view.hovering


def test_parallax_never_queues_more_than_one_render(panel, media, qtbot):
    panel.set_path(media["a"])
    panel.set_current_tab(TAB_PARALLAX)
    qtbot.waitUntil(lambda: panel._li is not None, timeout=10000)
    panel.auto_timer.stop()
    qtbot.waitUntil(lambda: not panel._parallax_busy, timeout=5000)
    for i in range(50):
        panel._request_parallax(i / 50, 0.0)
    assert len([w for w in panel.tasks.active() if w.name == "parallax"]) <= 1
    qtbot.waitUntil(lambda: panel._parallax_drawn == (49 / 50, 0.0), timeout=5000)


def test_options_change_invalidates_stereo(panel, media, qtbot):
    panel.set_path(media["a"])
    panel.set_current_tab(TAB_STEREO)
    qtbot.waitUntil(lambda: panel.views[TAB_STEREO].pixmap() is not None, timeout=10000)
    depth = panel._depth
    panel.opts["strength"] = 1.8
    panel._apply_options()
    qtbot.waitUntil(lambda: panel._anaglyph_sig is not None and panel._anaglyph_sig[1] == 1.8, timeout=10000)
    assert panel._depth is depth  # depth was kept
    panel.opts["common"]["depth_model"] = "heuristic"
    panel._apply_options()
    assert panel._depth is None or panel._depth is not depth
    qtbot.waitUntil(lambda: panel._depth is not None and panel._depth_sig[0] == "heuristic", timeout=10000)


def test_switching_input_drops_stale_results(panel, media, qtbot):
    panel.set_current_tab(TAB_DEPTH)
    panel.set_path(media["a"])
    panel.set_path(media["b"])
    qtbot.waitUntil(lambda: panel.views[TAB_DEPTH].pixmap() is not None, timeout=10000)
    qtbot.waitUntil(lambda: not panel.tasks.busy(), timeout=10000)
    assert panel.current_path() == media["b"]
    assert panel._depth.image.shape[:2] == panel.engine.source(media["b"]).shape[:2]
    panel.set_path(None)
    assert "Add a photo" in panel.views[TAB_DEPTH].message()


def test_depth_failure_shows_friendly_message(qtbot, media):
    p = PreviewPanel(engine=PreviewEngine(lambda n, d: FailingEstimator()))
    qtbot.addWidget(p)
    p.show()
    msgs: list[str] = []
    p.statusMessage.connect(msgs.append)
    p.set_current_tab(TAB_DEPTH)
    p.set_path(media["a"])
    qtbot.waitUntil(lambda: "not available" in p.views[TAB_DEPTH].message(), timeout=10000)
    assert "could not be downloaded" in p.views[TAB_DEPTH].message()
    assert "internet" in p.views[TAB_DEPTH].message()
    assert any("Preview failed" in m for m in msgs)
    p.shutdown()


def test_preview_runs_in_background_thread(qtbot, media):
    main = threading.get_ident()
    threads: list[int] = []

    class Spy(CountingEstimator):
        def estimate(self, img, *, progress=None, cancel=None):
            threads.append(threading.get_ident())
            return super().estimate(img, progress=progress, cancel=cancel)

    p = PreviewPanel(engine=PreviewEngine(lambda n, d: Spy()))
    qtbot.addWidget(p)
    p.show()
    p.set_current_tab(TAB_DEPTH)
    p.set_path(media["a"])
    qtbot.waitUntil(lambda: p.views[TAB_DEPTH].pixmap() is not None, timeout=10000)
    assert threads and all(t != main for t in threads)
    p.shutdown()


def test_image_view_message_and_rect(qtbot):
    from vrphoto.gui.panels.preview import ImageView

    v = ImageView()
    qtbot.addWidget(v)
    v.resize(400, 300)
    assert v.image_rect().isEmpty()
    v.set_image(np.zeros((100, 200, 3), np.uint8))
    r = v.image_rect()
    assert r.width() == pytest.approx(400) and r.height() == pytest.approx(200)
    v.set_message("Working")
    v.show()
    v.repaint()
    assert v.message() == "Working"
