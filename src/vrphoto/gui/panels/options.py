"""Option panels built generically from the job dataclasses + :mod:`..options_meta`.

* :class:`OptionsForm` - one form for a list of dataclass fields (basic rows on top, an
  expandable "Advanced" section below).
* :class:`JobOptionsPanel` - the full panel for one job kind: presets, the job's own
  options and the "Common settings" (from :class:`CommonOptions`); builds a JobSpec.

Values shown by the widgets are in display units (see ``FieldMeta.scale``); the forms
convert to/from the stored dataclass values, so ``values()`` is always JSON-able and
``build_spec()`` round-trips through ``job_to_dict`` / ``job_from_dict``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import types
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.errors import InputError
from ...core.log import get_logger
from .. import options_meta as om

log = get_logger(__name__)

NONE_DATA = "__none__"


# --------------------------------------------------------------------------- type helpers
@dataclasses.dataclass(frozen=True)
class FieldType:
    base: type  # float | int | bool | str | list | dict
    optional: bool


def field_types(cls: type) -> dict[str, FieldType]:
    """Resolved base type + optional flag of every dataclass field of ``cls``."""
    hints = typing.get_type_hints(cls)
    out: dict[str, FieldType] = {}
    for f in dataclasses.fields(cls):
        tp = hints.get(f.name, str)
        optional = False
        origin = typing.get_origin(tp)
        if origin in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(tp) if a is not type(None)]
            optional = len(args) != len(typing.get_args(tp))
            tp = args[0] if args else str
            origin = typing.get_origin(tp)
        base = origin or tp
        if not isinstance(base, type):
            base = str
        out[f.name] = FieldType(base, optional)
    return out


def default_values(cls: type) -> dict[str, Any]:
    """Dataclass defaults as a dict (default factories evaluated)."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            out[f.name] = f.default_factory()  # type: ignore[misc]
    return out


def _clean_float(v: float) -> float:
    return float(round(float(v), 9))


# --------------------------------------------------------------------------- dynamic choices
_FALLBACK_CHOICES: dict[str, list[tuple[Any, str]]] = {
    "depth_models": [
        ("auto", "Automatic (Depth Anything V2 Small)"),
        ("da2-small", "Depth Anything V2 Small (fast)"),
        ("da2-base", "Depth Anything V2 Base (non-commercial)"),
        ("da2-large", "Depth Anything V2 Large (non-commercial)"),
        ("moge2", "MoGe-2 (real distances + field of view)"),
        ("da3", "Depth Anything 3 (experimental)"),
        ("heuristic", "Simple (no AI)"),
    ],
    "inpaint_models": [
        ("auto", "Automatic (best available)"),
        ("lama", "LaMa (best quality)"),
        ("migan", "MI-GAN (fast)"),
        ("opencv", "OpenCV (no download)"),
        ("pushpull", "Smooth fill (instant, blurry)"),
    ],
    "segment_models": [
        ("auto", "Automatic (best available)"),
        ("birefnet-lite", "BiRefNet lite"),
        ("birefnet-portrait", "BiRefNet portrait (people)"),
        ("isnet", "IS-Net"),
        ("u2net", "U2-Net (fast)"),
        ("grabcut", "GrabCut (no download)"),
    ],
    "devices": [("auto", "Automatic (fastest GPU)"), ("cpu", "CPU only")],
}


_device_choices: list[tuple[Any, str]] | None = None


def detect_device_choices() -> list[tuple[Any, str]]:
    """Enumerate GPUs (may take a second on Windows: run it in a worker thread) and cache
    the result for :func:`dynamic_choices`."""
    global _device_choices
    try:
        from ...models.gpu import device_choices

        _device_choices = list(device_choices())
    except Exception as e:
        log.debug("GPU list unavailable: %s", e)
        _device_choices = list(_FALLBACK_CHOICES["devices"])
    return list(_device_choices)


def dynamic_choices(name: str) -> list[tuple[Any, str]]:
    """Choices from the live backends (fail soft to a static list).

    ``"devices"`` never enumerates GPUs itself (that can block on Windows); it returns the
    list cached by :func:`detect_device_choices`, or Automatic/CPU until that has run.
    """
    if name == "devices":
        return list(_device_choices) if _device_choices is not None else list(_FALLBACK_CHOICES["devices"])
    try:
        if name == "depth_models":
            from ...depth.base import list_estimators

            return list(list_estimators())
        if name == "inpaint_models":
            from ...inpaint.base import list_inpainters

            return list(list_inpainters())
        if name == "segment_models":
            from ...segment.base import list_segmenters

            return list(list_segmenters())
    except Exception as e:
        log.debug("Dynamic choices %s unavailable (%s); using defaults", name, e)
    return list(_FALLBACK_CHOICES.get(name, []))


# --------------------------------------------------------------------------- editors
class FieldEditor(QWidget):
    """Base class: one widget editing one field (values in STORED units)."""

    changed = Signal()

    def __init__(self, name: str, meta: om.FieldMeta, ftype: FieldType, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self.meta = meta
        self.ftype = ftype
        self.setToolTip(meta.tooltip)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._layout = lay

    def value(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def set_value(self, v: Any) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # numeric conversion helpers
    def _to_display(self, v: float) -> float:
        return float(v) * self.meta.scale

    def _to_stored(self, d: float) -> Any:
        v = d / self.meta.scale
        if self.ftype.base is int:
            return int(round(v))
        return _clean_float(v)


class _NoWheelMixin:
    """Ignore the mouse wheel unless focused, so scrolling the panel never changes values."""

    def wheelEvent(self, e):  # noqa: N802 - Qt API
        if self.hasFocus():  # type: ignore[attr-defined]
            super().wheelEvent(e)  # type: ignore[misc]
        else:
            e.ignore()


class NoWheelSpin(_NoWheelMixin, QDoubleSpinBox):
    pass


class NoWheelIntSpin(_NoWheelMixin, QSpinBox):
    pass


class NoWheelSlider(_NoWheelMixin, QSlider):
    pass


class NoWheelCombo(_NoWheelMixin, QComboBox):
    pass


class SpinEditor(FieldEditor):
    """Spin box; optional fields use the minimum as an "Auto" special value."""

    def __init__(self, name, meta, ftype, parent=None):
        super().__init__(name, meta, ftype, parent)
        lo = meta.minimum if meta.minimum is not None else 0.0
        hi = meta.maximum if meta.maximum is not None else 1e9
        self._sentinel: float | None = None
        is_int = ftype.base is int and meta.scale == 1 and meta.decimals == 0
        if is_int:
            spin: QDoubleSpinBox | QSpinBox = NoWheelIntSpin()
            lo_i, hi_i = int(lo), int(min(hi, 2**31 - 1))
            if ftype.optional:
                self._sentinel = lo_i - max(1, int(meta.step))
                spin.setMinimum(int(self._sentinel))
            else:
                spin.setMinimum(lo_i)
            spin.setMaximum(hi_i)
            spin.setSingleStep(max(1, int(meta.step)))
            spin.setGroupSeparatorShown(True)
        else:
            spin = NoWheelSpin()
            spin.setDecimals(meta.decimals)
            if ftype.optional:
                self._sentinel = lo - meta.step
                spin.setMinimum(self._sentinel)
            else:
                spin.setMinimum(lo)
            spin.setMaximum(hi)
            spin.setSingleStep(meta.step)
            spin.setGroupSeparatorShown(meta.decimals == 0)
        spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if ftype.optional:
            spin.setSpecialValueText(meta.none_label or "Auto")
        elif meta.zero_label and lo == 0:
            spin.setSpecialValueText(meta.zero_label)
        if meta.unit:
            spin.setSuffix(f" {meta.unit}")
        spin.setKeyboardTracking(False)
        spin.setMinimumWidth(110)
        self.spin = spin
        self._layout.addWidget(spin, 1)
        spin.valueChanged.connect(lambda *_: self.changed.emit())

    def value(self) -> Any:
        d = self.spin.value()
        if self._sentinel is not None and d <= self._sentinel + 1e-9:
            return None
        return self._to_stored(d)

    def set_value(self, v: Any) -> None:
        self.spin.blockSignals(True)
        try:
            if v is None:
                if self._sentinel is not None:
                    self.spin.setValue(self._sentinel)
            else:
                d = self._to_display(v)
                if isinstance(self.spin, QSpinBox):
                    self.spin.setValue(int(round(d)))
                else:
                    self.spin.setValue(d)
        finally:
            self.spin.blockSignals(False)


class SliderEditor(FieldEditor):
    """Slider (snaps to ``step``) + spin box (exact value, keyboard entry)."""

    def __init__(self, name, meta, ftype, parent=None):
        super().__init__(name, meta, ftype, parent)
        self.lo = float(meta.minimum if meta.minimum is not None else 0.0)
        self.hi = float(meta.maximum if meta.maximum is not None else 1.0)
        self.step = float(meta.step or 1.0)
        n = max(1, int(round((self.hi - self.lo) / self.step)))
        self.slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, n)
        self.slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.slider.setMinimumWidth(90)
        self.spin = NoWheelSpin()
        self.spin.setDecimals(meta.decimals)
        self.spin.setRange(self.lo, self.hi)
        self.spin.setSingleStep(self.step)
        self.spin.setKeyboardTracking(False)
        self.spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.spin.setGroupSeparatorShown(meta.decimals == 0)
        if meta.unit:
            self.spin.setSuffix(f" {meta.unit}")
        if meta.zero_label and self.lo == 0:
            self.spin.setSpecialValueText(meta.zero_label)
        self.spin.setFixedWidth(104)
        self._layout.addWidget(self.slider, 1)
        self._layout.addWidget(self.spin)
        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_changed)

    def _pos_for(self, d: float) -> int:
        return int(round((min(max(d, self.lo), self.hi) - self.lo) / self.step))

    def _slider_moved(self, pos: int) -> None:
        d = self.lo + pos * self.step
        self.spin.blockSignals(True)
        self.spin.setValue(d)
        self.spin.blockSignals(False)
        self.changed.emit()

    def _spin_changed(self, d: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._pos_for(d))
        self.slider.blockSignals(False)
        self.changed.emit()

    def value(self) -> Any:
        return self._to_stored(self.spin.value())

    def set_value(self, v: Any) -> None:
        if v is None:
            return
        d = self._to_display(v)
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        try:
            self.spin.setValue(d)
            self.slider.setValue(self._pos_for(d))
        finally:
            for w in (self.slider, self.spin):
                w.blockSignals(False)


class ComboEditor(FieldEditor):
    """Combo box over ``(value, label)`` choices (values kept in a Python list)."""

    def __init__(
        self,
        name,
        meta,
        ftype,
        parent=None,
        *,
        choices: list[tuple[Any, str]] | None = None,
        show_experimental: bool = False,
    ):
        super().__init__(name, meta, ftype, parent)
        self.combo = NoWheelCombo()
        self.combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(14)
        self.combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._values: list[Any] = []
        self._layout.addWidget(self.combo, 1)
        self.set_choices(choices if choices is not None else list(meta.choices), show_experimental)
        self.combo.currentIndexChanged.connect(lambda *_: self.changed.emit())

    def set_choices(self, choices: list[tuple[Any, str]], show_experimental: bool = False) -> None:
        current = self.value() if self._values else None
        self.combo.blockSignals(True)
        try:
            self.combo.clear()
            self._values = []
            for v, label in choices:
                if v in self.meta.experimental_choices and not show_experimental:
                    continue
                self._values.append(v)
                self.combo.addItem(label)
        finally:
            self.combo.blockSignals(False)
        if current is not None:
            self.set_value(current)

    def values(self) -> list[Any]:
        return list(self._values)

    @staticmethod
    def _same(a: Any, b: Any) -> bool:
        """Equality that keeps None / True / False / 1 / 1.0 apart (but 30 == 30.0)."""
        if a is None or b is None:
            return a is b
        if isinstance(a, bool) or isinstance(b, bool):
            return type(a) is type(b) and a == b
        try:
            return bool(a == b)
        except Exception:  # pragma: no cover - exotic values
            return False

    def _index_of(self, v: Any) -> int:
        for i, x in enumerate(self._values):
            if self._same(x, v):
                return i
        return -1

    def value(self) -> Any:
        i = self.combo.currentIndex()
        if i < 0 or i >= len(self._values):
            return None
        return self._values[i]

    def set_value(self, v: Any) -> None:
        i = self._index_of(v)
        self.combo.blockSignals(True)
        try:
            if i < 0:  # keep unknown / hidden (e.g. experimental) values instead of losing them
                self._values.append(v)
                self.combo.addItem("Automatic" if v is None else str(v))
                i = len(self._values) - 1
            self.combo.setCurrentIndex(i)
        finally:
            self.combo.blockSignals(False)


class CheckEditor(FieldEditor):
    def __init__(self, name, meta, ftype, parent=None):
        super().__init__(name, meta, ftype, parent)
        self.check = QCheckBox()
        self.check.setToolTip(meta.tooltip)
        self._layout.addWidget(self.check)
        self._layout.addStretch(1)
        self.check.toggled.connect(lambda *_: self.changed.emit())

    def value(self) -> Any:
        return bool(self.check.isChecked())

    def set_value(self, v: Any) -> None:
        self.check.blockSignals(True)
        self.check.setChecked(bool(v))
        self.check.blockSignals(False)


class TextEditor(FieldEditor):
    """Line edit; empty means None for optional fields."""

    def __init__(self, name, meta, ftype, parent=None):
        super().__init__(name, meta, ftype, parent)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(meta.placeholder)
        self.edit.setClearButtonEnabled(True)
        self._layout.addWidget(self.edit, 1)
        self.edit.textChanged.connect(lambda *_: self.changed.emit())

    def value(self) -> Any:
        t = self.edit.text().strip()
        if not t:
            return None if self.ftype.optional else ""
        return t

    def set_value(self, v: Any) -> None:
        self.edit.blockSignals(True)
        self.edit.setText("" if v is None else str(v))
        self.edit.blockSignals(False)


class PathEditor(TextEditor):
    """Line edit + "Browse..." button (``kind`` "file" or "folder").

    A file field's dialog lists the files named by ``FieldMeta.file_filter`` ("images" =
    the photo formats the app reads, plus "All files" for odd extensions).
    """

    def __init__(self, name, meta, ftype, parent=None):
        super().__init__(name, meta, ftype, parent)
        self.button = QPushButton("Browse…")
        self.button.setToolTip("Choose a folder" if meta.kind == "folder" else "Choose a file")
        self._layout.addWidget(self.button)
        self.button.clicked.connect(self.browse)

    def name_filter(self) -> str:
        """QFileDialog name filter of a file field ("" = all files)."""
        if self.meta.file_filter == "images":
            from ..util import image_dialog_filter

            return image_dialog_filter()
        if self.meta.file_filter:
            log.debug("Unknown file filter %r for %s", self.meta.file_filter, self.name)
        return ""

    def browse(self) -> None:
        start = self.edit.text().strip() or str(Path.home())
        if self.meta.kind == "folder":
            path = QFileDialog.getExistingDirectory(self, self.meta.label, start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, self.meta.label, start, self.name_filter())
        if path:
            self.edit.setText(str(Path(path)))


class JsonEditor(FieldEditor):
    """Small JSON text box (e.g. custom key frames).  Invalid JSON raises InputError."""

    def __init__(self, name, meta, ftype, parent=None):
        super().__init__(name, meta, ftype, parent)
        self.edit = QPlainTextEdit()
        self.edit.setPlaceholderText(meta.placeholder)
        self.edit.setFixedHeight(64)
        self._layout.addWidget(self.edit, 1)
        self.edit.textChanged.connect(self._validate)

    def _parse(self) -> Any:
        t = self.edit.toPlainText().strip()
        if not t:
            return None if self.ftype.optional else []
        try:
            v = json.loads(t)
        except ValueError as e:
            raise InputError(f"'{self.meta.label}' is not valid JSON: {e}") from e
        if self.ftype.base is list and not isinstance(v, list):
            raise InputError(f"'{self.meta.label}' must be a JSON list [...].")
        return v

    def _validate(self) -> None:
        try:
            self._parse()
            ok = True
        except InputError:
            ok = False
        self.edit.setProperty("invalid", not ok)
        self.edit.style().unpolish(self.edit)
        self.edit.style().polish(self.edit)
        self.changed.emit()

    def value(self) -> Any:
        return self._parse()

    def set_value(self, v: Any) -> None:
        self.edit.blockSignals(True)
        self.edit.setPlainText("" if v is None else json.dumps(v))
        self.edit.blockSignals(False)
        self._validate()


def make_editor(
    name: str,
    meta: om.FieldMeta,
    ftype: FieldType,
    *,
    choice_provider: Callable[[str], list[tuple[Any, str]]] = dynamic_choices,
    show_experimental: bool = False,
) -> FieldEditor:
    """Create the editor widget for one field."""
    kind = meta.kind
    if kind == "combo":
        choices = choice_provider(meta.choices_from) if meta.choices_from else list(meta.choices)
        return ComboEditor(name, meta, ftype, choices=choices, show_experimental=show_experimental)
    if kind == "check":
        return CheckEditor(name, meta, ftype)
    if kind == "slider" and not ftype.optional:
        return SliderEditor(name, meta, ftype)
    if kind in ("slider", "spin"):
        return SpinEditor(name, meta, ftype)
    if kind in ("file", "folder"):
        return PathEditor(name, meta, ftype)
    if kind == "json":
        return JsonEditor(name, meta, ftype)
    return TextEditor(name, meta, ftype)


# --------------------------------------------------------------------------- collapsible section
class Section(QWidget):
    """A titled section whose body can be collapsed (used for "Advanced" / "Common")."""

    toggled = Signal(bool)

    def __init__(self, title: str, *, expanded: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self._title = title
        self.button = QToolButton()
        self.button.setObjectName("SectionToggle")
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.button.setText(title)
        self.button.toggled.connect(self.set_expanded)
        self.body = QWidget()
        self.body.setVisible(expanded)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.button)
        lay.addWidget(self.body)

    def set_expanded(self, on: bool) -> None:
        self.button.blockSignals(True)
        self.button.setChecked(on)
        self.button.blockSignals(False)
        self.button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.body.setVisible(on)
        self.toggled.emit(on)

    @property
    def expanded(self) -> bool:
        return self.button.isChecked()


# --------------------------------------------------------------------------- forms
class OptionsForm(QWidget):
    """Form for a set of dataclass fields; basic rows first, advanced rows collapsible."""

    changed = Signal()

    def __init__(
        self,
        cls: type,
        meta: dict[str, om.FieldMeta],
        *,
        names: list[str] | None = None,
        show_experimental: bool = False,
        choice_provider: Callable[[str], list[tuple[Any, str]]] = dynamic_choices,
        advanced_title: str = "Advanced",
        kind: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.cls = cls
        self.meta = meta
        self.kind = kind  # job kind ("common" for CommonOptions): for options_meta.field_applies
        self.types = field_types(cls)
        self.defaults = default_values(cls)
        self.editors: dict[str, FieldEditor] = {}
        self.labels: dict[str, QLabel] = {}
        all_names = [f.name for f in dataclasses.fields(cls) if f.name not in om.STRUCTURAL_FIELDS]
        self.names = [n for n in (names if names is not None else all_names) if n in meta]
        missing = [n for n in all_names if n not in meta]
        if missing:
            log.warning("No GUI metadata for %s fields: %s", cls.__name__, ", ".join(missing))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        basic = QFormLayout()
        self._setup_form(basic)
        outer.addLayout(basic)
        adv_names = [n for n in self.names if meta[n].advanced]
        self.advanced: Section | None = None
        adv_form: QFormLayout | None = None
        if adv_names:
            self.advanced = Section(advanced_title, expanded=False)
            adv_form = QFormLayout(self.advanced.body)
            adv_form.setContentsMargins(0, 0, 0, 0)
            self._setup_form(adv_form)
            outer.addWidget(self.advanced)

        for n in self.names:
            m = meta[n]
            ed = make_editor(
                n, m, self.types[n], choice_provider=choice_provider, show_experimental=show_experimental
            )
            ed.set_value(self.defaults.get(n))
            ed.changed.connect(self._on_changed)
            lab = QLabel(m.label)
            lab.setToolTip(m.tooltip)
            lab.setBuddy(ed)
            self.editors[n] = ed
            self.labels[n] = lab
            (adv_form if (m.advanced and adv_form is not None) else basic).addRow(lab, ed)
        self._update_enabled()

    @staticmethod
    def _setup_form(form: QFormLayout) -> None:
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

    def _on_changed(self) -> None:
        self._update_enabled()
        self.changed.emit()

    def _update_enabled(self) -> None:
        vals = self._lenient_values()
        for n, ed in self.editors.items():
            if not self.meta[n].conditions():
                continue
            on = self._applies(n, vals)
            ed.setEnabled(on)
            self.labels[n].setEnabled(on)

    # ---------------------------------------------------------------- values
    def values(self, *, strict: bool = True) -> dict[str, Any]:
        """Current values (stored units) for the edited fields.

        With ``strict`` an invalid entry (e.g. malformed JSON) raises InputError - unless
        the field is greyed out (its ``enabled_if`` condition is not met), then its default
        is used.  Without ``strict`` invalid fields are left out (used for previews and
        for remembering options).
        """
        out: dict[str, Any] = {}
        for n, ed in self.editors.items():
            try:
                out[n] = ed.value()
            except InputError:
                if strict and self._applies(n):
                    raise
                if strict:  # greyed out (e.g. key frames of a stock movement): not used anyway
                    out[n] = self.defaults.get(n)
        return out

    def _lenient_values(self) -> dict[str, Any]:
        """Current values, leaving out invalid entries (which then grey nothing out)."""
        out: dict[str, Any] = {}
        for n, ed in self.editors.items():
            try:
                out[n] = ed.value()
            except InputError:
                continue
        return out

    def _applies(self, name: str, values: dict[str, Any] | None = None) -> bool:
        """False if field ``name`` is greyed out (``options_meta.field_applies``, the same rule
        the add-on check uses)."""
        if not self.meta[name].conditions():
            return True
        vals = self._lenient_values() if values is None else values
        return om.field_applies(self.kind, name, vals, meta=self.meta)

    def set_values(self, values: dict[str, Any]) -> None:
        for n, v in values.items():
            ed = self.editors.get(n)
            if ed is not None:
                try:
                    ed.set_value(v)
                except Exception:  # bad remembered value -> keep the current one
                    log.debug("Ignoring value %r for %s", v, n, exc_info=True)
        self._update_enabled()

    def reset(self) -> None:
        self.set_values(self.defaults)

    def set_choices(self, name: str, choices: list[tuple[Any, str]], show_experimental: bool = False) -> None:
        ed = self.editors.get(name)
        if isinstance(ed, ComboEditor):
            ed.set_choices(choices, show_experimental)


class JobOptionsPanel(QWidget):
    """All options for one job kind: preset picker, job options, Common settings."""

    changed = Signal()

    def __init__(
        self,
        kind: str,
        *,
        common_defaults: dict[str, Any] | None = None,
        show_experimental: bool = False,
        choice_provider: Callable[[str], list[tuple[Any, str]]] = dynamic_choices,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        from ...pipeline.jobs import JOB_TYPES, CommonOptions

        if kind not in JOB_TYPES:
            raise InputError(f"Unknown job kind {kind!r}")
        self.kind = kind
        self.cls = JOB_TYPES[kind]
        self.mode = om.MODE_BY_KIND.get(kind)
        self.show_experimental = show_experimental
        self._choice_provider = choice_provider
        self._applying = False
        self.common_defaults = dict(default_values(CommonOptions))
        self.common_defaults.update(common_defaults or {})

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # presets
        prow = QHBoxLayout()
        plabel = QLabel("Preset")
        plabel.setToolTip("Quick starting points. You can still change every option afterwards.")
        self.preset_combo = NoWheelCombo()
        self.preset_combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.preset_combo.setToolTip(plabel.toolTip())
        self.preset_names = [n for n in om.PRESET_NAMES if n in om.PRESETS.get(kind, {})]
        self.preset_combo.addItem("Default")
        for n in self.preset_names:
            self.preset_combo.addItem(n)
        self.preset_combo.addItem("Custom")
        self._custom_index = self.preset_combo.count() - 1
        self.preset_combo.currentIndexChanged.connect(self._preset_chosen)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Back to the recommended default settings for this mode.")
        self.reset_button.clicked.connect(lambda: self.apply_preset("Default"))
        prow.addWidget(plabel)
        prow.addWidget(self.preset_combo, 1)
        prow.addWidget(self.reset_button)
        lay.addLayout(prow)

        # job options
        self.form = OptionsForm(
            self.cls,
            om.JOB_META.get(kind, {}),
            show_experimental=show_experimental,
            choice_provider=choice_provider,
            kind=kind,
        )
        self.form.changed.connect(self._on_user_change)
        lay.addWidget(self.form)

        # common options
        common_names = [n for n, m in om.COMMON_META.items() if m.only_modes is None or kind in m.only_modes]
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("Separator")
        lay.addWidget(line)
        self.common_section = Section("Depth, AI models and output", expanded=False)
        cl = QVBoxLayout(self.common_section.body)
        cl.setContentsMargins(0, 0, 0, 0)
        self.common = OptionsForm(
            CommonOptions,
            om.COMMON_META,
            names=common_names,
            show_experimental=show_experimental,
            choice_provider=choice_provider,
            advanced_title="More",
            kind="common",
        )
        self.common.set_values(self.common_defaults)
        self.common.changed.connect(self._on_user_change)
        cl.addWidget(self.common)
        lay.addWidget(self.common_section)
        lay.addStretch(1)
        self._hidden_common: dict[str, Any] = {
            n: self.common_defaults.get(n) for n in self.common_defaults if n not in self.common.editors
        }

    # ---------------------------------------------------------------- presets
    def _preset_chosen(self, index: int) -> None:
        if self._applying or index == self._custom_index:
            return
        name = self.preset_combo.itemText(index)
        self.apply_preset(name)

    def apply_preset(self, name: str) -> None:
        """Reset to defaults and apply a named preset ("Default" = defaults only)."""
        overrides = om.PRESETS.get(self.kind, {}).get(name, {}) if name != "Default" else {}
        job_vals = dict(self.form.defaults)
        common_vals = dict(self.common_defaults)
        for k, v in overrides.items():
            if k.startswith("common."):
                common_vals[k.split(".", 1)[1]] = v
            else:
                job_vals[k] = v
        self._applying = True
        try:
            self.form.set_values(job_vals)
            self.common.set_values(common_vals)
            idx = self.preset_combo.findText(name)
            self.preset_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self._applying = False
        self.changed.emit()

    def current_preset(self) -> str:
        return self.preset_combo.currentText()

    def _on_user_change(self) -> None:
        if not self._applying:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(self._custom_index)
            self.preset_combo.blockSignals(False)
        self.changed.emit()

    # ---------------------------------------------------------------- values / specs
    def job_values(self, *, strict: bool = True) -> dict[str, Any]:
        return self.form.values(strict=strict)

    def common_values(self, *, strict: bool = True) -> dict[str, Any]:
        vals = dict(self._hidden_common)
        vals.update(self.common.values(strict=strict))
        return vals

    def options_dict(self, *, strict: bool = True) -> dict[str, Any]:
        """JSON-able ``{field: value, "common": {...}}`` (no inputs / kind).

        ``strict=False`` never raises: invalid entries are left out.
        """
        d = self.job_values(strict=strict)
        d["common"] = self.common_values(strict=strict)
        return d

    def load_options(self, d: dict[str, Any]) -> None:
        """Restore values from :meth:`options_dict` output (unknown keys ignored)."""
        if not isinstance(d, dict):
            return
        self._applying = True
        try:
            common = d.get("common") or {}
            if isinstance(common, dict):
                for k, v in common.items():
                    if k in self._hidden_common:
                        self._hidden_common[k] = v
                self.common.set_values(common)
            self.form.set_values({k: v for k, v in d.items() if k != "common"})
            self.preset_combo.setCurrentIndex(self._custom_index)
        finally:
            self._applying = False

    def build_spec(self, inputs: list[str]) -> Any:
        """Build the JobSpec dataclass for ``inputs`` (raises InputError on invalid values)."""
        from ...pipeline.jobs import job_from_dict

        d = self.options_dict()
        d["kind"] = self.kind
        d["inputs"] = [str(p) for p in inputs]
        self._validate(d)
        return job_from_dict(d)

    def _validate(self, d: dict[str, Any]) -> None:
        c = d.get("common", {})
        near, far = c.get("near_m"), c.get("far_m")
        if isinstance(near, (int, float)) and isinstance(far, (int, float)) and near >= far:
            raise InputError(
                "'Nearest distance' must be smaller than 'Farthest distance'.",
                hint="Open 'Depth, AI models and output' and fix the two distances.",
            )
        for k, v in list(d.items()) + list(c.items()):
            if isinstance(v, float) and not math.isfinite(v):
                raise InputError(f"Option {k} has an invalid value.")
        if self.kind == "parallax" and d.get("path") == "custom" and not d.get("keyframes"):
            raise InputError(
                "The 'Custom' camera movement needs key frames.",
                hint="Open 'Advanced' and enter key frames, or pick another camera movement.",
            )
        # SHARP 360 'Your own back photo': the job would stop on a missing picture anyway
        # (before any work); saying so here keeps it out of the queue
        if self.kind == "object" and "back_images" in d and om.field_applies(self.kind, "back_images", d):
            back = d.get("back_images")
            if not back:
                raise InputError(
                    "'Your own back photo' needs a picture of the back.",
                    hint="Choose it with 'Browse…' next to 'Back photo', or pick another 'Back made by'.",
                )
            from ...pipeline.sharp360 import _back_photo  # the job's own check (Qt-free, header only)

            _back_photo(back)  # missing, not a picture, no view from behind, unreadable; folders are fine

    def set_common_defaults(self, defaults: dict[str, Any], apply: dict[str, Any] | None = None) -> None:
        """Change the Common defaults (used by Reset/presets) and set ``apply`` values now."""
        self.common_defaults.update(defaults)
        if apply:
            for k, v in apply.items():
                if k in self._hidden_common:
                    self._hidden_common[k] = v
            self.common.set_values(apply)

    def refresh_choices(self, show_experimental: bool) -> None:
        """Re-query dynamic choices (models, devices) and the experimental filter."""
        self.show_experimental = show_experimental
        for form in (self.form, self.common):
            for n, ed in form.editors.items():
                if isinstance(ed, ComboEditor):
                    m = form.meta[n]
                    choices = self._choice_provider(m.choices_from) if m.choices_from else list(m.choices)
                    ed.set_choices(choices, show_experimental)
