"""Modern dark (default) and light themes: Fusion style + palette + a QSS stylesheet.

Widgets opt into special looks with object names / dynamic properties, e.g.
``QPushButton#PrimaryButton`` (accent button), ``QFrame#Card`` (rounded panel),
``QLabel#Muted`` (secondary text), ``QLabel[badge="nc"]`` (non-commercial badge).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from ..core.log import get_logger

log = get_logger(__name__)

THEMES = ("system", "dark", "light")


@dataclass(frozen=True)
class Colors:
    bg: str
    panel: str
    panel_alt: str
    border: str
    text: str
    muted: str
    accent: str
    accent_hover: str
    accent_text: str
    danger: str
    warning: str
    success: str
    selection: str


DARK = Colors(
    bg="#15171c",
    panel="#1d2027",
    panel_alt="#252932",
    border="#323845",
    text="#e8eaf0",
    muted="#9aa3b5",
    accent="#4f8cff",
    accent_hover="#6a9dff",
    accent_text="#ffffff",
    danger="#ff6b6b",
    warning="#ffb84d",
    success="#3ecf8e",
    selection="#2f4a7a",
)
LIGHT = Colors(
    bg="#f3f4f7",
    panel="#ffffff",
    panel_alt="#eceef3",
    border="#d3d7e0",
    text="#1c1f26",
    muted="#5d6577",
    accent="#2f6fec",
    accent_hover="#4a82f0",
    accent_text="#ffffff",
    danger="#d93838",
    warning="#b86e00",
    success="#14915a",
    selection="#cfe0ff",
)


def resolve_theme(name: str | None) -> str:
    """``"system"`` -> ``"dark"``/``"light"`` from the OS colour scheme (dark if unknown)."""
    name = (name or "system").lower()
    if name in ("dark", "light"):
        return name
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Light:
            return "light"
    except Exception:  # older Qt / no app yet
        pass
    return "dark"


def colors(name: str | None) -> Colors:
    return LIGHT if resolve_theme(name) == "light" else DARK


def _resource_url(name: str) -> str:
    """Quoted ``url(...)`` argument for a bundled image (forward slashes work on Windows
    too; quoting keeps paths with spaces or parentheses, e.g. ``Program Files (x86)``,
    intact)."""
    try:
        from ..core.config import resource_path

        p = resource_path("gui", "resources", name)
        if p.is_file():
            return '"' + p.resolve().as_posix().replace('"', '\\"') + '"'
    except Exception:  # pragma: no cover - defensive
        pass
    return ""


def _arrow_rules(c: Colors) -> str:
    variant = "dark" if c is LIGHT else "light"
    up, down = _resource_url(f"arrow_up_{variant}.png"), _resource_url(f"arrow_down_{variant}.png")
    if not (up and down):
        return ""
    return f"""
QAbstractSpinBox {{ padding-right: 20px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: 18px;
    border: none; background: transparent; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {c.border}; }}
QAbstractSpinBox::up-arrow {{ image: url({up}); width: 10px; height: 6px; }}
QAbstractSpinBox::down-arrow {{ image: url({down}); width: 10px; height: 6px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::down-arrow:disabled {{ image: none; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 22px;
    border: none; }}
QComboBox::down-arrow {{ image: url({down}); width: 10px; height: 6px; }}
"""


def _check_rules(c: Colors) -> str:
    """Clearly visible check boxes (Fusion's default box is nearly invisible in the dark
    theme); needs the bundled check-mark image, otherwise Fusion's own box is kept."""
    check = _resource_url("check_light.png")
    if not check:
        return ""
    return f"""
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {c.muted}; border-radius: 4px;
    background: {c.panel_alt}; }}
QCheckBox::indicator:hover {{ border-color: {c.accent_hover}; }}
QCheckBox::indicator:checked {{ background: {c.accent}; border-color: {c.accent}; image: url({check}); }}
QCheckBox::indicator:disabled {{ border-color: {c.border}; background: {c.panel}; }}
QCheckBox::indicator:checked:disabled {{ background: {c.border}; border-color: {c.border}; }}
"""


def stylesheet(c: Colors) -> str:
    """The application QSS for a colour set."""
    return (
        f"""
QWidget {{ color: {c.text}; font-size: 10pt; }}
QWidget:disabled {{ color: {c.muted}; }}
QMainWindow, QDialog {{ background: {c.bg}; }}
QToolTip {{ color: {c.text}; background: {c.panel_alt}; border: 1px solid {c.border};
    padding: 6px; border-radius: 6px; }}
QFrame#Card, QFrame#Panel {{ background: {c.panel}; border: 1px solid {c.border}; border-radius: 10px; }}
QFrame#ModeCard {{ background: {c.panel}; border: 1px solid {c.border}; border-radius: 10px; }}
QFrame#ModeCard:hover {{ border-color: {c.accent_hover}; }}
QFrame#ModeCard[selected="true"] {{ background: {c.panel_alt}; border: 2px solid {c.accent}; }}
QLabel#ModeTitle {{ font-weight: 600; font-size: 10.5pt; }}
QLabel#ModeDesc, QLabel#Muted {{ color: {c.muted}; }}
QLabel#SectionTitle {{ font-size: 12pt; font-weight: 600; }}
QLabel#Heading {{ font-size: 14pt; font-weight: 700; }}
QLabel#BigUrl {{ font-size: 20pt; font-weight: 700; color: {c.accent};
    background: {c.panel_alt}; border-radius: 8px; padding: 8px 12px; }}
QLabel#Error {{ color: {c.danger}; }}
QLabel#Warning {{ color: {c.warning}; }}
QLabel#Success {{ color: {c.success}; }}
QLabel[badge="nc"] {{ color: #ffffff; background: {c.danger}; border-radius: 4px; padding: 1px 6px;
    font-size: 8pt; font-weight: 700; }}
QLabel[badge="exp"] {{ color: #1c1f26; background: {c.warning}; border-radius: 4px; padding: 1px 6px;
    font-size: 8pt; font-weight: 700; }}
QLabel[badge="ok"] {{ color: #ffffff; background: {c.success}; border-radius: 4px; padding: 1px 6px;
    font-size: 8pt; font-weight: 700; }}
QPushButton {{ background: {c.panel_alt}; border: 1px solid {c.border}; border-radius: 7px;
    padding: 6px 14px; }}
QPushButton:hover {{ border-color: {c.accent_hover}; }}
QPushButton:pressed {{ background: {c.border}; }}
QPushButton:disabled {{ color: {c.muted}; background: {c.panel}; border-color: {c.panel_alt}; }}
QPushButton#PrimaryButton {{ background: {c.accent}; color: {c.accent_text}; border: none;
    font-weight: 600; padding: 9px 18px; font-size: 11pt; }}
QPushButton#PrimaryButton:hover {{ background: {c.accent_hover}; }}
QPushButton#PrimaryButton:disabled {{ background: {c.border}; color: {c.muted}; }}
QPushButton#DangerButton {{ color: {c.danger}; }}
QPushButton#LinkButton, QToolButton#LinkButton {{ background: transparent; border: none;
    color: {c.accent}; padding: 2px 4px; text-align: left; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 4px 8px; }}
QToolButton:hover {{ border-color: {c.border}; background: {c.panel_alt}; }}
QToolButton#SectionToggle {{ font-weight: 600; padding: 4px 2px; }}
QToolBar {{ background: {c.panel}; border: none; border-bottom: 1px solid {c.border}; spacing: 4px; padding: 3px; }}
QMenuBar {{ background: {c.panel}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {c.selection}; }}
QMenu {{ background: {c.panel}; border: 1px solid {c.border}; }}
QStatusBar {{ background: {c.panel}; border-top: 1px solid {c.border}; }}
QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {c.panel_alt}; border: 1px solid {c.border}; border-radius: 6px; padding: 4px 6px;
    selection-background-color: {c.selection}; }}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {c.accent}; }}
QLineEdit[invalid="true"], QPlainTextEdit[invalid="true"] {{ border-color: {c.danger}; }}
QComboBox QAbstractItemView {{ background: {c.panel}; border: 1px solid {c.border};
    selection-background-color: {c.selection}; }}
QListWidget, QTreeWidget, QTableWidget {{ background: {c.panel}; border: 1px solid {c.border};
    border-radius: 8px; alternate-background-color: {c.panel_alt}; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {c.selection}; color: {c.text}; }}
QHeaderView::section {{ background: {c.panel_alt}; border: none; border-bottom: 1px solid {c.border};
    padding: 4px 6px; }}
QTabWidget::pane {{ border: 1px solid {c.border}; border-radius: 8px; background: {c.panel}; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 6px 14px; border: none; color: {c.muted}; }}
QTabBar::tab:selected {{ color: {c.text}; border-bottom: 2px solid {c.accent}; }}
QTabBar::tab:hover {{ color: {c.text}; }}
QProgressBar {{ background: {c.panel_alt}; border: 1px solid {c.border}; border-radius: 5px;
    text-align: center; min-height: 14px; }}
QProgressBar::chunk {{ background: {c.accent}; border-radius: 4px; }}
QProgressBar[state="failed"]::chunk {{ background: {c.danger}; }}
QProgressBar[state="done"]::chunk {{ background: {c.success}; }}
QSlider::groove:horizontal {{ height: 4px; background: {c.border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {c.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {c.accent_text}; border: 2px solid {c.accent}; width: 12px;
    height: 12px; margin: -6px 0; border-radius: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QSplitter::handle {{ background: {c.bg}; }}
QSplitter::handle:hover {{ background: {c.border}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {c.border}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {c.muted}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {c.border}; border-radius: 5px; min-width: 30px; }}
QScrollBar::handle:horizontal:hover {{ background: {c.muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QGroupBox {{ border: 1px solid {c.border}; border-radius: 8px; margin-top: 14px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; font-weight: 600; }}
"""
        + _arrow_rules(c)
        + _check_rules(c)
    )


def palette(c: Colors) -> QPalette:
    p = QPalette()
    role = QPalette.ColorRole
    p.setColor(role.Window, QColor(c.bg))
    p.setColor(role.WindowText, QColor(c.text))
    p.setColor(role.Base, QColor(c.panel_alt))
    p.setColor(role.AlternateBase, QColor(c.panel))
    p.setColor(role.ToolTipBase, QColor(c.panel_alt))
    p.setColor(role.ToolTipText, QColor(c.text))
    p.setColor(role.Text, QColor(c.text))
    p.setColor(role.Button, QColor(c.panel_alt))
    p.setColor(role.ButtonText, QColor(c.text))
    p.setColor(role.BrightText, QColor(c.danger))
    p.setColor(role.Highlight, QColor(c.accent))
    p.setColor(role.HighlightedText, QColor(c.accent_text))
    p.setColor(role.Link, QColor(c.accent))
    p.setColor(role.PlaceholderText, QColor(c.muted))
    for r in (role.Text, role.WindowText, role.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, r, QColor(c.muted))
    return p


def _set_color_scheme(name: str | None) -> None:
    """Tell Qt (>= 6.8) which scheme the app uses, so e.g. the Windows title bars match the
    theme; ``"system"`` follows the OS again."""
    want = {"dark": Qt.ColorScheme.Dark, "light": Qt.ColorScheme.Light}.get(
        (name or "system").lower(), Qt.ColorScheme.Unknown
    )
    try:
        hints = QGuiApplication.styleHints()
        if hints.colorScheme() != want or want == Qt.ColorScheme.Unknown:
            hints.setColorScheme(want)
    except Exception:  # older Qt / platform without support
        log.debug("Cannot set the colour scheme", exc_info=True)


def follow_system_theme(app: QApplication, theme_name: Callable[[], str | None]) -> Callable[[], None]:
    """Re-apply the theme when the OS switches between dark and light mode while the
    chosen theme (``theme_name()``, e.g. from Settings) is ``"system"``.

    Returns a function that stops following.
    """

    def changed(*_: object) -> None:
        try:
            name = (theme_name() or "system").lower()
        except Exception:  # pragma: no cover - defensive
            return
        if name == "system" and app.property("vrphoto_theme") != resolve_theme("system"):
            apply_theme(app, "system")

    hints = QGuiApplication.styleHints()
    try:
        hints.colorSchemeChanged.connect(changed)
    except Exception:  # older Qt
        log.debug("colorSchemeChanged not available", exc_info=True)
        return lambda: None

    def stop() -> None:
        try:
            hints.colorSchemeChanged.disconnect(changed)
        except (RuntimeError, TypeError):
            pass

    return stop


def apply_theme(app: QApplication, name: str | None = "system") -> str:
    """Apply a theme to the whole application.  Returns the resolved name ("dark"/"light")."""
    _set_color_scheme(name)
    resolved = resolve_theme(name)
    c = LIGHT if resolved == "light" else DARK
    try:
        app.setStyle("Fusion")
    except Exception:  # pragma: no cover - style plugin missing
        log.debug("Fusion style not available", exc_info=True)
    app.setPalette(palette(c))
    app.setStyleSheet(stylesheet(c))
    app.setProperty("vrphoto_theme", resolved)
    return resolved


def current_colors() -> Colors:
    app = QApplication.instance()
    name = app.property("vrphoto_theme") if app is not None else None
    return LIGHT if name == "light" else DARK
