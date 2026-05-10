"""
themes.py – UI theme definitions.

Each theme dict now includes:
  tick_color      font/tick-number color on plot axes
  grid_alpha      alpha for plot grid lines
  (all previous keys still present)
"""

_BASE_SLIDER = """
QSlider::groove:horizontal {{
    background: {groove};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {handle};
    width: 12px; height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
"""

_BASE_QSS = """
QMainWindow, QWidget {{
    background-color: {bg};
    color: {text};
}}
QPushButton {{
    background-color: {btn};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 8px;
}}
QPushButton:hover   {{ background-color: {btn_hover}; }}
QPushButton:pressed {{ background-color: {btn_press}; }}
QComboBox {{
    background-color: {btn};
    color: {text};
    border: 1px solid {border};
    border-radius: 3px;
    padding: 2px 4px;
}}
QComboBox QAbstractItemView {{
    background-color: {btn};
    color: {text};
    selection-background-color: {btn_hover};
}}
QCheckBox {{ spacing: 6px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: {panel};
}}
QLabel {{ color: {text}; }}
"""


def _make(
    bg,
    panel,
    text,
    btn,
    btn_hover,
    btn_press,
    border,
    plot_bg,
    axis_color,
    zero_line_color,
    info_bg,
    info_text,
    header_color,
    groove,
    handle,
    tick_color=None,
    grid_alpha=0.28,
):

    if tick_color is None:
        tick_color = axis_color

    qss = (_BASE_QSS + _BASE_SLIDER).format(
        bg=bg,
        panel=panel,
        text=text,
        btn=btn,
        btn_hover=btn_hover,
        btn_press=btn_press,
        border=border,
        groove=groove,
        handle=handle,
    )
    return {
        "stylesheet": qss,
        "plot_bg": plot_bg,
        "axis_color": axis_color,
        "tick_color": tick_color,
        "grid_alpha": grid_alpha,
        "zero_line_color": zero_line_color,
        "info_bg": info_bg,
        "info_text": info_text,
        "header_color": header_color,
    }


THEMES = {
    "Dark": _make(
        bg="#121212",
        panel="#1e1e1e",
        text="#e0e0e0",
        btn="#2a2a2a",
        btn_hover="#3a3a3a",
        btn_press="#1a1a1a",
        border="#555",
        plot_bg="#0d0d0d",
        axis_color="#cccccc",
        tick_color="#cccccc",
        zero_line_color="#444444",
        info_bg="#1e1e1e",
        info_text="#e0e0e0",
        header_color="#5ec4ff",
        groove="#333333",
        handle="#888888",
        grid_alpha=0.25,
    ),
    "Grey": _make(
        bg="#3c3c3c",
        panel="#4a4a4a",
        text="#e0e0e0",
        btn="#5a5a5a",
        btn_hover="#6a6a6a",
        btn_press="#4a4a4a",
        border="#888",
        plot_bg="#2e2e2e",
        axis_color="#dddddd",
        tick_color="#dddddd",
        zero_line_color="#666666",
        info_bg="#4a4a4a",
        info_text="#e0e0e0",
        header_color="#5ec4ff",
        groove="#5a5a5a",
        handle="#aaaaaa",
        grid_alpha=0.28,
    ),
    "White": _make(
        bg="#f5f5f5",
        panel="#e8e8e8",
        text="#1a1a1a",
        btn="#dcdcdc",
        btn_hover="#cccccc",
        btn_press="#c0c0c0",
        border="#aaaaaa",
        plot_bg="#ffffff",
        axis_color="#333333",
        tick_color="#222222",
        zero_line_color="#bbbbbb",
        info_bg="#e8e8e8",
        info_text="#1a1a1a",
        header_color="#0066cc",
        groove="#cccccc",
        handle="#777777",
        grid_alpha=0.30,
    ),
}
