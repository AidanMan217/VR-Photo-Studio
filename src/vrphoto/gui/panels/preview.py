"""Centre panel: preview tabs for the selected input.

Tabs: Original, Depth (computed on demand with the selected depth model, colourised and
cached), 3D Preview (red/cyan anaglyph from a quick low-res stereo pair), Wiggle (animated
stereo wiggle) and Parallax (the view follows the mouse; rendered with
``geometry.warp.render_view`` on a ~640 px layered image at 10-20 fps).

All computation happens in background threads (:class:`TaskRunner`); results from a
previous input or outdated options are dropped using a generation counter.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QTabWidget, QVBoxLayout, QWidget

from ...core.log import get_logger
from ..preview import PreviewEngine
from ..util import error_text, np_to_pixmap
from ..workers import TaskRunner

log = get_logger(__name__)

TAB_ORIGINAL, TAB_DEPTH, TAB_STEREO, TAB_WIGGLE, TAB_PARALLAX = range(5)
TAB_TITLES = ("Original", "Depth", "3D Preview", "Wiggle", "Parallax")
TAB_HINTS = (
    "Your photo as it will be processed.",
    "Depth map from the selected depth model: warm/bright = near, cool/dark = far.",
    "Quick 3D preview for red/cyan glasses (the real result is full-colour stereo).",
    "The picture rocks between the left and right eye views - a quick way to judge the depth.",
    "Move the mouse over the picture to look around it, like moving your head in VR.",
)


class ImageView(QWidget):
    """Shows a pixmap scaled to fit (keeping aspect ratio) with an optional message overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._message = ""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(640, 480)

    def set_pixmap(self, pm: QPixmap | None) -> None:
        self._pixmap = pm
        self.update()

    def set_image(self, img) -> None:  # noqa: ANN001 - numpy array
        self.set_pixmap(np_to_pixmap(img) if img is not None else None)

    def pixmap(self) -> QPixmap | None:
        return self._pixmap

    def set_message(self, text: str) -> None:
        self._message = text or ""
        self.update()

    def message(self) -> str:
        return self._message

    def image_rect(self) -> QRectF:
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        s = min(self.width() / pw, self.height() / ph)
        w, h = pw * s, ph * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, e) -> None:  # noqa: N802, ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if self._pixmap is not None and not self._pixmap.isNull():
            p.drawPixmap(self.image_rect(), self._pixmap, QRectF(self._pixmap.rect()))
        if self._message:
            fm = p.fontMetrics()
            r = fm.boundingRect(
                self.rect().adjusted(40, 0, -40, 0), int(Qt.TextFlag.TextWordWrap), self._message
            )
            box = QRectF(r).adjusted(-14, -10, 14, 10)
            box.moveCenter(QPointF(self.rect().center()))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 160))
            p.drawRoundedRect(box, 8, 8)
            p.setPen(QColor("#ffffff"))
            p.drawText(box, int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self._message)
        p.end()


class ParallaxView(ImageView):
    """ImageView that reports the mouse position as (dx, dy) in [-1, 1]."""

    moved = Signal(float, float)
    left = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.hovering = False

    def mouseMoveEvent(self, e) -> None:  # noqa: N802, ANN001
        self.hovering = True
        r = self.image_rect()
        if r.isEmpty():
            r = QRectF(self.rect())
        pos = e.position()
        dx = (pos.x() - r.center().x()) / max(1.0, r.width() / 2)
        dy = (pos.y() - r.center().y()) / max(1.0, r.height() / 2)
        self.moved.emit(max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy)))

    def enterEvent(self, e) -> None:  # noqa: N802, ANN001
        self.hovering = True
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802, ANN001
        self.hovering = False
        self.left.emit()
        super().leaveEvent(e)


def _num(d: dict[str, Any], key: str, default: float | None) -> float | None:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    return float(v)


class PreviewPanel(QWidget):
    """The preview tabs.  Feed it a path with :meth:`set_path` and an options provider."""

    statusMessage = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        engine: PreviewEngine | None = None,
        options_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine or PreviewEngine()
        self.options_provider = options_provider or (lambda: {})
        self.tasks = TaskRunner(self, max_threads=2)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.views: list[ImageView] = []
        for i, title in enumerate(TAB_TITLES):
            v = ParallaxView() if i == TAB_PARALLAX else ImageView()
            v.setToolTip(TAB_HINTS[i])
            self.views.append(v)
            self.tabs.addTab(v, title)
            self.tabs.setTabToolTip(i, TAB_HINTS[i])
        self.hint = QLabel(TAB_HINTS[0])
        self.hint.setObjectName("Muted")
        self.hint.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(self.hint)
        self.tabs.currentChanged.connect(self._tab_changed)

        self._path: Path | None = None
        self._gen = 0
        self._depth = None
        self._depth_sig: tuple | None = None
        self._li = None
        self._layer_sig: tuple | None = None
        self._anaglyph_sig: tuple | None = None
        self._wiggle_sig: tuple | None = None
        self._wiggle_frames: list[QPixmap] = []
        self._wiggle_i = 0
        self._running: dict[str, Any] = {}
        self._parallax_busy = False
        self._parallax_target = (0.0, 0.0)
        self._parallax_drawn: tuple[float, float] | None = None
        self._parallax_t0 = time.monotonic()
        self.parallax_fps = 0.0

        self.wiggle_timer = QTimer(self)
        self.wiggle_timer.setInterval(110)
        self.wiggle_timer.timeout.connect(self._wiggle_tick)
        self.auto_timer = QTimer(self)
        self.auto_timer.setInterval(66)  # ~15 fps idle animation of the parallax view
        self.auto_timer.timeout.connect(self._auto_tick)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(400)
        self.debounce.timeout.connect(self._apply_options)

        pv: ParallaxView = self.views[TAB_PARALLAX]  # type: ignore[assignment]
        pv.moved.connect(self._parallax_mouse)
        pv.left.connect(self._parallax_left)
        self._show_empty()

    # ---------------------------------------------------------------- public API
    def current_path(self) -> Path | None:
        return self._path

    def set_path(self, path: str | Path | None) -> None:
        """Show ``path`` (or nothing).  Cancels work for the previous input."""
        p = Path(path) if path else None
        if p == self._path:
            return
        self._path = p
        self._reset_all()
        if p is None:
            self._show_empty()
            return
        self.views[TAB_ORIGINAL].set_message("Loading…")
        self._start("source", self.engine.source, p, on_result=self._source_done, pass_progress=False)
        self._refresh()

    def options_changed(self) -> None:
        """Call when job options change; previews update after a short pause."""
        self.debounce.start()

    def set_current_tab(self, index: int) -> None:
        self.tabs.setCurrentIndex(index)

    def shutdown(self) -> None:
        self.wiggle_timer.stop()
        self.auto_timer.stop()
        self.debounce.stop()
        self.tasks.shutdown(3000)

    # ---------------------------------------------------------------- option signatures
    def _opts(self) -> dict[str, Any]:
        try:
            d = self.options_provider() or {}
        except Exception:
            log.debug("Options provider failed", exc_info=True)
            d = {}
        return d if isinstance(d, dict) else {}

    def _sigs(self) -> tuple[tuple, tuple, tuple, float]:
        o = self._opts()
        c = o.get("common") if isinstance(o.get("common"), dict) else {}
        depth_sig = (str(c.get("depth_model") or "auto"), str(c.get("device") or "auto"))
        layer_sig = (
            _num(c, "near_m", 1.2),
            _num(c, "far_m", 15.0),
            _num(c, "hfov_deg", None),
            o.get("look_behind", "band") != "none",
        )
        params = PreviewEngine.stereo_params(o)
        stereo_sig = (
            params.ipd_m,
            params.strength,
            params.convergence_m,
            params.max_parallax_pct,
            params.swap_eyes,
        )
        amp = _num(o, "amplitude_m", None)
        if amp is None:
            mb = _num(o, "max_baseline_m", None)
            amp = mb / 2 if mb else 0.04
        return depth_sig, layer_sig, stereo_sig, float(min(max(amp, 0.005), 0.25))

    def _apply_options(self) -> None:
        if self._path is None:
            return
        depth_sig, layer_sig, stereo_sig, _ = self._sigs()
        if self._depth is not None and depth_sig != self._depth_sig:
            self._invalidate_depth()
        elif self._li is not None and layer_sig != self._layer_sig:
            self._invalidate_layers()
        elif stereo_sig != self._anaglyph_sig or stereo_sig != self._wiggle_sig:
            self._invalidate_stereo()
        self._parallax_drawn = None
        self._refresh()

    # ---------------------------------------------------------------- state resets
    def _reset_all(self) -> None:
        self._gen += 1
        self.tasks.cancel_all()
        self._running.clear()
        self._parallax_busy = False
        self.views[TAB_ORIGINAL].set_pixmap(None)
        self._invalidate_depth()

    def _invalidate_depth(self) -> None:
        self._depth = None
        self._depth_sig = None
        self.views[TAB_DEPTH].set_pixmap(None)
        self._invalidate_layers()

    def _invalidate_layers(self) -> None:
        self._li = None
        self._layer_sig = None
        self.views[TAB_PARALLAX].set_pixmap(None)
        self._parallax_drawn = None
        self._invalidate_stereo()

    def _invalidate_stereo(self) -> None:
        self._anaglyph_sig = None
        self._wiggle_sig = None
        self._wiggle_frames = []
        self.wiggle_timer.stop()
        self.views[TAB_STEREO].set_pixmap(None)
        self.views[TAB_WIGGLE].set_pixmap(None)

    def _show_empty(self) -> None:
        for v in self.views:
            v.set_pixmap(None)
            v.set_message("Add a photo on the left to see a preview here.")

    # ---------------------------------------------------------------- task helpers
    def _start(
        self,
        key: str,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None],
        pass_progress: bool = True,
        **kwargs: Any,
    ) -> None:
        if key in self._running:
            return
        gen = self._gen

        def ok(value: Any) -> None:
            if gen == self._gen:
                on_result(value)

        def err(exc: BaseException) -> None:
            if gen == self._gen:
                self._failed(key, exc)

        def prog(frac: float, msg: str) -> None:
            if gen == self._gen:
                self._progress(key, frac, msg)

        def fin() -> None:
            if gen == self._gen:
                self._running.pop(key, None)

        w = self.tasks.start(
            fn,
            *args,
            on_result=ok,
            on_error=err,
            on_finished=fin,
            on_progress=prog if pass_progress else None,
            pass_progress=pass_progress,
            name=f"preview-{key}",
            **kwargs,
        )
        self._running[key] = w

    def _dependent_views(self, key: str) -> list[int]:
        return {
            "source": [TAB_ORIGINAL],
            "depth": [TAB_DEPTH, TAB_STEREO, TAB_WIGGLE, TAB_PARALLAX],
            "layers": [TAB_STEREO, TAB_WIGGLE, TAB_PARALLAX],
            "anaglyph": [TAB_STEREO],
            "wiggle": [TAB_WIGGLE],
            "parallax": [TAB_PARALLAX],
        }.get(key, [])

    def _progress(self, key: str, frac: float, msg: str) -> None:
        text = f"{msg or 'Working'}… {int(frac * 100)}%" if 0 < frac < 1 else f"{msg or 'Working'}…"
        for i in self._dependent_views(key):
            if self.views[i].pixmap() is None:
                self.views[i].set_message(text)

    def _failed(self, key: str, exc: BaseException) -> None:
        self._running.pop(key, None)
        msg, hint = error_text(exc)
        log.warning("Preview (%s) failed: %s", key, msg)
        text = f"Preview not available:\n{msg}" + (f"\n{hint}" if hint else "")
        for i in self._dependent_views(key):
            self.views[i].set_message(text)
        if key == "parallax":
            self._parallax_busy = False
        self.statusMessage.emit(f"Preview failed: {msg}")

    # ---------------------------------------------------------------- pipeline
    def _tab_changed(self, index: int) -> None:
        self.hint.setText(TAB_HINTS[index] if 0 <= index < len(TAB_HINTS) else "")
        if index != TAB_WIGGLE:
            self.wiggle_timer.stop()
        if index != TAB_PARALLAX:
            self.auto_timer.stop()
        self._refresh()

    def _refresh(self) -> None:
        """Start whatever the visible tab still needs (depth -> layers -> render)."""
        if self._path is None:
            return
        tab = self.tabs.currentIndex()
        if tab == TAB_ORIGINAL:
            return
        depth_sig, layer_sig, stereo_sig, _amp = self._sigs()
        if self._depth is None:
            model, device = depth_sig
            for i in (TAB_DEPTH, TAB_STEREO, TAB_WIGGLE, TAB_PARALLAX):
                if self.views[i].pixmap() is None:
                    self.views[i].set_message("Estimating depth…")
            self._start(
                "depth",
                self.engine.depth,
                self._path,
                model,
                device,
                on_result=lambda dp, s=depth_sig: self._depth_done(dp, s),
            )
            return
        if tab == TAB_DEPTH:
            return
        if self._li is None:
            near, far, hfov, look = layer_sig
            for i in (TAB_STEREO, TAB_WIGGLE, TAB_PARALLAX):
                if self.views[i].pixmap() is None:
                    self.views[i].set_message("Preparing 3D…")
            self._start(
                "layers",
                self.engine.layered,
                self._depth,
                near_m=near or 1.2,
                far_m=far or 15.0,
                hfov_deg=hfov,
                look_behind=look,
                pass_progress=False,
                on_result=lambda li, s=layer_sig: self._layers_done(li, s),
            )
            return
        params = PreviewEngine.stereo_params(self._opts())
        if tab == TAB_STEREO and self._anaglyph_sig != stereo_sig:
            self.views[TAB_STEREO].set_message("Rendering 3D preview…")
            self._start(
                "anaglyph",
                self.engine.anaglyph,
                self._li,
                params,
                pass_progress=False,
                on_result=lambda img, s=stereo_sig: self._anaglyph_done(img, s),
            )
        elif tab == TAB_WIGGLE:
            if self._wiggle_sig != stereo_sig:
                self.views[TAB_WIGGLE].set_message("Rendering wiggle…")
                self._start(
                    "wiggle",
                    self.engine.wiggle,
                    self._li,
                    params,
                    8,
                    pass_progress=False,
                    on_result=lambda frames, s=stereo_sig: self._wiggle_done(frames, s),
                )
            elif self._wiggle_frames and not self.wiggle_timer.isActive():
                self.wiggle_timer.start()
        elif tab == TAB_PARALLAX:
            if self._parallax_drawn is None:
                self._request_parallax(*self._parallax_target)
            if not self.auto_timer.isActive():
                self.auto_timer.start()

    def _source_done(self, img) -> None:  # noqa: ANN001
        v = self.views[TAB_ORIGINAL]
        v.set_image(img)
        v.set_message("")
        name = self._path.name if self._path else ""
        self.statusMessage.emit(f"{name}: {img.shape[1]} × {img.shape[0]} preview")

    def _clear_pending_messages(self, tabs: tuple[int, ...]) -> None:
        for i in tabs:
            if self.views[i].pixmap() is None:
                self.views[i].set_message("")

    def _depth_done(self, dp, sig: tuple) -> None:  # noqa: ANN001
        self._depth = dp
        self._depth_sig = sig
        v = self.views[TAB_DEPTH]
        v.set_image(self.engine.colorized(dp))
        v.set_message("")
        self._clear_pending_messages((TAB_STEREO, TAB_WIGGLE, TAB_PARALLAX))
        self._refresh()

    def _layers_done(self, li, sig: tuple) -> None:  # noqa: ANN001
        self._li = li
        self._layer_sig = sig
        self._clear_pending_messages((TAB_STEREO, TAB_WIGGLE, TAB_PARALLAX))
        self._refresh()

    def _anaglyph_done(self, img, sig: tuple) -> None:  # noqa: ANN001
        self._anaglyph_sig = sig
        v = self.views[TAB_STEREO]
        v.set_image(img)
        v.set_message("")

    def _wiggle_done(self, frames, sig: tuple) -> None:  # noqa: ANN001
        self._wiggle_sig = sig
        self._wiggle_frames = [np_to_pixmap(f) for f in frames]
        self._wiggle_i = 0
        v = self.views[TAB_WIGGLE]
        v.set_message("")
        if self._wiggle_frames:
            v.set_pixmap(self._wiggle_frames[0])
            if self.tabs.currentIndex() == TAB_WIGGLE:
                self.wiggle_timer.start()

    def _wiggle_tick(self) -> None:
        if not self._wiggle_frames:
            self.wiggle_timer.stop()
            return
        self._wiggle_i = (self._wiggle_i + 1) % len(self._wiggle_frames)
        self.views[TAB_WIGGLE].set_pixmap(self._wiggle_frames[self._wiggle_i])

    # ---------------------------------------------------------------- parallax
    def _parallax_mouse(self, dx: float, dy: float) -> None:
        self._request_parallax(dx, dy)

    def _parallax_left(self) -> None:
        self._parallax_t0 = time.monotonic()

    def _auto_tick(self) -> None:
        pv: ParallaxView = self.views[TAB_PARALLAX]  # type: ignore[assignment]
        if pv.hovering or self._li is None or not self.isVisible():
            return
        t = time.monotonic() - self._parallax_t0
        a = 2 * math.pi * t / 4.0  # one slow circle every 4 s
        self._request_parallax(0.7 * math.sin(a), 0.5 * math.cos(a))

    def _request_parallax(self, dx: float, dy: float) -> None:
        """Render the latest requested viewpoint; never queues more than one render."""
        self._parallax_target = (dx, dy)
        if self._li is None or self._parallax_busy:
            return
        _, _, _, amp = self._sigs()
        self._parallax_busy = True
        target = (dx, dy)
        started = time.monotonic()
        gen = self._gen

        def done(img) -> None:  # noqa: ANN001
            if gen != self._gen:
                return
            self._parallax_busy = False
            self._parallax_drawn = target
            dt = max(1e-3, time.monotonic() - started)
            self.parallax_fps = 0.8 * self.parallax_fps + 0.2 * (1.0 / dt) if self.parallax_fps else 1.0 / dt
            v = self.views[TAB_PARALLAX]
            v.set_image(img)
            v.set_message("")
            if self._parallax_target != target:
                self._request_parallax(*self._parallax_target)

        def failed(exc: BaseException) -> None:
            if gen == self._gen:
                self._failed("parallax", exc)

        def cancelled() -> None:
            if gen == self._gen:
                self._parallax_busy = False

        self.tasks.start(
            self.engine.parallax_frame,
            self._li,
            dx,
            dy,
            amp,
            pass_progress=False,
            on_result=done,
            on_error=failed,
            on_cancelled=cancelled,
            name="parallax",
        )
