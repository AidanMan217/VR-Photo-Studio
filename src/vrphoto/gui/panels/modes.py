"""Right panel: "What do you want to make?" mode cards, the option panel of the selected
mode and the big "Create" button."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.log import get_logger
from .. import options_meta as om
from ..util import media_exts
from .options import JobOptionsPanel

log = get_logger(__name__)


class ModeCard(QFrame):
    """A clickable card with a title and a one-line description."""

    clicked = Signal(str)

    def __init__(self, mode: om.ModeInfo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mode = mode
        self.setObjectName("ModeCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("selected", False)
        self.setToolTip(mode.description)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(mode.title)
        self.setAccessibleDescription(mode.description)
        title = QLabel(mode.title)
        title.setObjectName("ModeTitle")
        desc = QLabel(mode.tagline)
        desc.setObjectName("ModeDesc")
        desc.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(1)
        lay.addWidget(title)
        lay.addWidget(desc)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)

    @property
    def selected(self) -> bool:
        return bool(self.property("selected"))

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.mode.kind)
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e) -> None:  # noqa: N802, ANN001
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit(self.mode.kind)
        else:
            super().keyPressEvent(e)


class ModePanel(QWidget):
    """Mode cards + per-mode options + Create button.

    Signals:
        modeChanged(str): the selected job kind changed.
        optionsChanged(): any option of the current mode changed.
        createRequested(): the user clicked "Create".
    """

    modeChanged = Signal(str)
    optionsChanged = Signal()
    createRequested = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        common_defaults: dict[str, Any] | None = None,
        show_experimental: bool = False,
        remembered: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.common_defaults = dict(common_defaults or {})
        self.show_experimental = show_experimental
        self.remembered: dict[str, dict[str, Any]] = dict(remembered or {})
        self.panels: dict[str, JobOptionsPanel] = {}
        self._kind = ""

        title = QLabel("What do you want to make?")
        title.setObjectName("SectionTitle")
        self.cards: dict[str, ModeCard] = {}
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, m in enumerate(om.MODES):
            card = ModeCard(m)
            card.clicked.connect(self.set_mode)
            self.cards[m.kind] = card
            grid.addWidget(card, i // 2, i % 2)
        cards_box = QWidget()
        cards_box.setLayout(grid)

        self.options_title = QLabel("")
        self.options_title.setObjectName("SectionTitle")
        self.options_desc = QLabel("")
        self.options_desc.setObjectName("Muted")
        self.options_desc.setWordWrap(True)
        self.stack = QStackedWidget()
        self._placeholder = QLabel("")
        self.stack.addWidget(self._placeholder)
        opt_box = QWidget()
        ol = QVBoxLayout(opt_box)
        ol.setContentsMargins(2, 0, 8, 0)
        ol.addWidget(self.options_title)
        ol.addWidget(self.options_desc)
        ol.addWidget(self.stack)
        ol.addStretch(1)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(title)
        cl.addWidget(cards_box)
        cl.addSpacing(6)
        cl.addWidget(opt_box)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(content)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.inputs_label = QLabel("")
        self.inputs_label.setObjectName("Muted")
        self.inputs_label.setWordWrap(True)
        self.create_button = QPushButton("Create")
        self.create_button.setObjectName("PrimaryButton")
        self.create_button.setToolTip("Start making the selected kind of result for the selected photos.")
        self.create_button.clicked.connect(self.createRequested)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.scroll, 1)
        lay.addWidget(self.inputs_label)
        lay.addWidget(self.create_button)
        self.set_mode(om.MODES[0].kind)

    # ---------------------------------------------------------------- modes
    def current_kind(self) -> str:
        return self._kind

    def current_mode(self) -> om.ModeInfo:
        return om.MODE_BY_KIND[self._kind]

    def panel_for(self, kind: str) -> JobOptionsPanel:
        """The option panel of ``kind`` (created on first use, restoring remembered options)."""
        p = self.panels.get(kind)
        if p is None:
            p = JobOptionsPanel(
                kind, common_defaults=self.common_defaults, show_experimental=self.show_experimental
            )
            remembered = self.remembered.get(kind)
            if remembered:
                p.load_options(remembered)
            p.changed.connect(self._panel_changed)
            self.panels[kind] = p
            self.stack.addWidget(p)
        return p

    def current_panel(self) -> JobOptionsPanel:
        return self.panel_for(self._kind)

    def set_mode(self, kind: str) -> None:
        if kind not in om.MODE_BY_KIND:
            log.warning("Unknown mode %r", kind)
            return
        changed = kind != self._kind
        self._kind = kind
        for k, card in self.cards.items():
            card.set_selected(k == kind)
        m = om.MODE_BY_KIND[kind]
        panel = self.panel_for(kind)
        self.stack.setCurrentWidget(panel)
        self.options_title.setText(f"{m.title} options")
        self.options_desc.setText(m.description)
        self.create_button.setText(f"Create {m.title}")
        if changed:
            self.modeChanged.emit(kind)

    def _panel_changed(self) -> None:
        if self.sender() is self.panels.get(self._kind):
            self.optionsChanged.emit()

    # ---------------------------------------------------------------- options memory
    def remember_all(self) -> dict[str, dict[str, Any]]:
        """Snapshot of options of every mode that was opened (for GuiState)."""
        for kind, p in self.panels.items():
            try:
                self.remembered[kind] = p.options_dict(strict=False)  # invalid entries are skipped
            except Exception as e:  # pragma: no cover - defensive: keep the previous snapshot
                log.debug("Not remembering options of %s: %s", kind, e)
        return dict(self.remembered)

    def current_options(self) -> dict[str, Any]:
        """Options of the current mode for previews (never raises; invalid entries skipped)."""
        try:
            return self.current_panel().options_dict(strict=False)
        except Exception:  # pragma: no cover - defensive
            log.debug("current_options failed", exc_info=True)
            return {}

    def refresh_choices(self, show_experimental: bool) -> None:
        self.show_experimental = show_experimental
        for p in self.panels.values():
            p.refresh_choices(show_experimental)

    def apply_common_defaults(self, defaults: dict[str, Any], names: set[str] | None = None) -> None:
        """New Settings defaults: update future panels, and set ``names`` in all existing ones."""
        self.common_defaults.update(defaults)
        upd = {k: v for k, v in defaults.items() if names is None or k in names}
        for p in self.panels.values():
            p.set_common_defaults(defaults, upd)
        for d in self.remembered.values():
            if isinstance(d.get("common"), dict):
                d["common"].update(upd)

    # ---------------------------------------------------------------- inputs / specs
    def accepted(self, paths: list[Path], kind: str | None = None) -> tuple[list[Path], list[Path]]:
        """Split ``paths`` into (usable, skipped) for the mode's accepted media types."""
        m = om.MODE_BY_KIND[kind or self._kind]
        images, videos = media_exts()
        ok, skipped = [], []
        for p in paths:
            ext = p.suffix.lower()
            kind_ = "video" if ext in videos else "image" if ext in images else ""
            (ok if kind_ in m.accepts else skipped).append(p)
        return ok, skipped

    def build_specs(self, paths: list[Path], kind: str | None = None) -> list[Any]:
        """JobSpecs for ``paths``: one per input, or one with all inputs for batch modes.

        Raises :class:`~vrphoto.core.errors.InputError` for invalid options.
        """
        kind = kind or self._kind
        m = om.MODE_BY_KIND[kind]
        panel = self.panel_for(kind)
        if not paths:
            return []
        if m.batch:
            return [panel.build_spec([str(p) for p in paths])]
        return [panel.build_spec([str(p)]) for p in paths]

    def update_inputs_hint(self, paths: list[Path], total: int) -> None:
        """Explain what "Create" will do with the current selection."""
        m = self.current_mode()
        ok, skipped = self.accepted(paths)
        if total == 0:
            text = "Add photos on the left, choose what to make, then click Create."
            enabled = False
        elif not ok:
            need = "videos" if m.accepts == frozenset({"video"}) else "photos"
            text = f"{m.title} needs {need}. Select some on the left."
            enabled = False
        elif m.batch:
            text = f"{len(ok)} file(s) → one {m.title} job."
            enabled = len(ok) >= m.min_inputs
        else:
            text = f"{len(ok)} file(s) → {len(ok)} job{'s' if len(ok) != 1 else ''}."
            enabled = True
        if skipped and ok:
            text += f" ({len(skipped)} not suitable for {m.title} will be skipped.)"
        self.inputs_label.setText(text)
        self.create_button.setEnabled(enabled)
