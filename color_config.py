"""
color_config.py – Element color defaults and the SwatchButton widget.

Per-trajectory colors are managed inside app.py (cycling palette + row widget).
This module only covers the fixed plot-element colors (nullclines, vector field, etc.)

Color format
------------
All colors are stored and passed as 9-character hex strings in RGBA order:
    "#RRGGBBAA"   e.g.  "#00d4ffff"  = fully opaque cyan
                        "#00d4ff80"  = 50 % transparent cyan

This matches CSS / web convention and is unambiguous.
QColor is constructed via explicit r,g,b,a integer channels - never via the
string constructor - so Qt's ARGB-string parser never gets involved.
"""

from PyQt6.QtWidgets import QPushButton, QColorDialog
from PyQt6.QtGui import QColor

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _hex_to_qcolor(hex_color: str) -> QColor:
    """
    Accept "#RRGGBB" (6 chars) or "#RRGGBBAA" (8 chars after #).
    Returns a QColor with alpha correctly set.
    """
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return QColor(r, g, b, 255)
    elif len(h) == 8:
        r, g, b, a = (
            int(h[0:2], 16),
            int(h[2:4], 16),
            int(h[4:6], 16),
            int(h[6:8], 16),
        )
        return QColor(r, g, b, a)
    else:
        # Unexpected format - try Qt's parser as last resort (no alpha)
        c = QColor(hex_color)
        return c if c.isValid() else QColor(128, 128, 128, 255)


def _qcolor_to_hex(c: QColor) -> str:
    """Return "#RRGGBBAA" string (9 chars, alpha last)."""
    return "#{:02x}{:02x}{:02x}{:02x}".format(c.red(), c.green(), c.blue(), c.alpha())


def _qcolor_to_css(c: QColor) -> str:
    """Return CSS rgba() string suitable for Qt stylesheets."""
    return "rgba({},{},{},{})".format(c.red(), c.green(), c.blue(), c.alpha())


# ---------------------------------------------------------------------------
# Fixed plot-element colors  (#RRGGBBAA - fully opaque by default)
# ---------------------------------------------------------------------------

DEFAULT_ELEMENT_COLORS: dict[str, str] = {
    "iso_u": "#00d4ffff",  # du/dt = 0 nullcline  – cyan
    "iso_v": "#ff6666ff",  # dv/dt = 0 nullcline  – coral
    "vector_field": "#666666ff",  # arrow field          – grey
    "fixed_point": "#ff3333ff",  # equilibrium marker   – red
}

ELEMENT_COLOR_LABELS: dict[str, str] = {
    "iso_u": "Nullcline  du/dt = 0",
    "iso_v": "Nullcline  dv/dt = 0",
    "vector_field": "Vector field",
    "fixed_point": "Fixed points",
}

# Cycling palette for new trajectories (fully opaque, #RRGGBBAA)
TRAJECTORY_PALETTE: list[str] = [
    "#f9e04bff",  # yellow
    "#4bbbf9ff",  # sky blue
    "#f97b4bff",  # orange
    "#7bf97bff",  # lime
    "#d07bf9ff",  # violet
    "#f94b91ff",  # pink
    "#4bf9d0ff",  # teal
    "#f9c44bff",  # amber
    "#a0a0ffff",  # lavender
    "#ff9f9fff",  # salmon
]


# ---------------------------------------------------------------------------
# SwatchButton
# ---------------------------------------------------------------------------


class SwatchButton(QPushButton):
    """
    A fixed-size button that paints itself as a colored swatch (with alpha).
    Clicking opens QColorDialog with the alpha channel slider visible.

    Colors are stored and emitted as "#RRGGBBAA" hex strings.
    6-char "#RRGGBB" strings are treated as fully opaque.
    """

    def __init__(self, color: str, callback, parent=None):
        super().__init__(parent)
        self._callback = callback
        self.setFixedSize(28, 20)
        self.set_color(color)
        self.clicked.connect(self._open_dialog)

    def set_color(self, hex_color: str):
        self._qcolor = _hex_to_qcolor(hex_color)
        self._color = _qcolor_to_hex(self._qcolor)  # normalise to 9-char RRGGBBAA
        css_bg = _qcolor_to_css(self._qcolor)
        self.setStyleSheet(
            f"background-color:{css_bg};" "border:1px solid #777;" "border-radius:3px;"
        )

    @property
    def color(self) -> str:
        """Current color as "#RRGGBBAA"."""
        return self._color

    @property
    def qcolor(self) -> QColor:
        return self._qcolor

    def _open_dialog(self):
        dlg = QColorDialog(self._qcolor)
        dlg.setWindowTitle("Pick color")
        dlg.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        dlg.setStyleSheet(
            "QDialog, QWidget { background: palette(window); color: palette(windowText); }"
            "QPushButton { background: palette(button); color: palette(buttonText);"
            "  border: 1px solid palette(mid); border-radius: 3px; padding: 4px 10px; }"
            "QPushButton:hover { background: palette(light); }"
        )
        if dlg.exec():
            chosen = dlg.currentColor()
            if chosen.isValid():
                self.set_color(_qcolor_to_hex(chosen))
                self._callback(self._color)


# ---------------------------------------------------------------------------
# Convenience: convert any stored color string to QColor for pg.mkPen / mkBrush
# ---------------------------------------------------------------------------


def to_qcolor(hex_color: str) -> QColor:
    """
    Convert "#RRGGBB" or "#RRGGBBAA" to QColor.
    Pass the result directly to pg.mkPen / pg.mkBrush / symbolBrush etc.
    """
    return _hex_to_qcolor(hex_color)
