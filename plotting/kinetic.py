"""
plotting/kinetic.py – Kinetic plot showing all visible trajectories.

Displays u(t) as a solid line and v(t) as a dashed line for every visible
trajectory.

Viewport policy:
  - Zoom and pan are fully locked. The view is always fitted to the data.
"""

import numpy as np
import pyqtgraph as pg
from color_config import to_qcolor

_SOLID = pg.QtCore.Qt.PenStyle.SolidLine
_DASH = pg.QtCore.Qt.PenStyle.DashLine


class KineticPlot:
    def __init__(self):
        self.plot = pg.PlotWidget(title="Kinetics  (all visible trajectories)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "t")
        self.plot.setLabel("left", "u, v")

        # Persistent legend explaining line-style encoding.
        # Two named invisible reference curves — pyqtgraph renders them
        # as legend swatches showing the solid / dashed distinction.
        self.plot.addLegend(offset=(10, 10))
        self._legend_u = self.plot.plot(
            [], [], pen=pg.mkPen("#aaaaaa", width=2, style=_SOLID), name="u(t)"
        )
        self._legend_v = self.plot.plot(
            [], [], pen=pg.mkPen("#aaaaaa", width=2, style=_DASH), name="v(t)"
        )

        # Completely disable user interaction
        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)

        # Disable auto-ranging so we control it ourselves
        self.plot.enableAutoRange(enable=False)

        # tid -> {t, u, v, color, visible, params, u0, v0,
        #         curve_u: PlotDataItem, curve_v: PlotDataItem}
        self._data: dict[int, dict] = {}
        self._order: list[int] = []
        self._width: int = 2

        self._x_range = (0.0, 1.0)
        self._y_range = (0.0, 1.0)
        self._locked = False

        self.plot.sigRangeChanged.connect(self._enforce_lock)

    # ------------------------------------------------------------------
    # Locking helpers
    # ------------------------------------------------------------------

    def _enforce_lock(self):
        if self._locked:
            return
        self._locked = True
        self.plot.setXRange(*self._x_range, padding=0)
        self.plot.setYRange(*self._y_range, padding=0)
        self._locked = False

    # ------------------------------------------------------------------
    # Appearance
    # ------------------------------------------------------------------

    def set_font_size(self, px: int):
        """Update all text sizes on the kinetic plot."""
        font = pg.QtGui.QFont()
        font.setPixelSize(px)
        for ax_name in ("left", "bottom"):
            ax = self.plot.getAxis(ax_name)
            ax.setTickFont(font)
            ax.setLabel(
                ax.labelText,
                **{"color": ax.labelStyle.get("color", ""), "font-size": f"{px}px"},
            )
        pi = self.plot.getPlotItem()
        pi.titleLabel.setAttr("size", f"{px}px")
        pi.titleLabel.setText(pi.titleLabel.text)

    def apply_theme(self, theme: dict):
        self.plot.setBackground(theme["plot_bg"])
        axis_color = theme["axis_color"]
        tick_color = theme.get("tick_color", axis_color)

        axis_pen = pg.mkPen(color=axis_color, width=2)
        tick_pen = pg.mkPen(color=tick_color)

        for axis_name in ("left", "bottom"):
            axis = self.plot.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(tick_pen)
            axis.setLabel(axis.labelText, **{"color": tick_color})

        # Plot title
        pi = self.plot.getPlotItem()
        pi.titleLabel.setAttr("color", tick_color)
        pi.titleLabel.setText(pi.titleLabel.text)

        # Legend: recolour every text item inside the LegendItem
        legend = pi.legend
        if legend is not None:
            for sample, label in legend.items:
                label.setAttr("color", tick_color)
                label.setText(label.text)
            legend.setBrush(pg.mkBrush(theme["plot_bg"]))
            legend.setPen(pg.mkPen(color=axis_color, width=1))

        self._update_legend_colors()

        self.plot.showGrid(x=True, y=True, alpha=theme.get("grid_alpha", 0.3))

    def set_line_width(self, width: int):
        self._width = width
        for d in self._data.values():
            color = d["color"]
            d["curve_u"].setPen(pg.mkPen(to_qcolor(color), width=width, style=_SOLID))
            d["curve_v"].setPen(pg.mkPen(to_qcolor(color), width=width, style=_DASH))

    # ------------------------------------------------------------------
    # Trajectory management
    # ------------------------------------------------------------------

    def add_trajectory(
        self,
        tid: int,
        t: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        color: str,
        visible: bool = True,
        params: dict | None = None,
        u0: float = 0.0,
        v0: float = 0.0,
    ):
        curve_u = self.plot.plot(
            t,
            u,
            pen=pg.mkPen(to_qcolor(color), width=self._width, style=_SOLID),
        )
        curve_v = self.plot.plot(
            t,
            v,
            pen=pg.mkPen(to_qcolor(color), width=self._width, style=_DASH),
        )

        self._data[tid] = dict(
            t=t,
            u=u,
            v=v,
            color=color,
            visible=visible,
            params=params or {},
            u0=u0,
            v0=v0,
            curve_u=curve_u,
            curve_v=curve_v,
        )
        if tid not in self._order:
            self._order.append(tid)

        curve_u.setVisible(visible)
        curve_v.setVisible(visible)
        self._fit_viewport()
        self._update_legend_colors()

    def remove_trajectory(self, tid: int):
        if tid in self._data:
            d = self._data.pop(tid)
            self.plot.removeItem(d["curve_u"])
            self.plot.removeItem(d["curve_v"])
        if tid in self._order:
            self._order.remove(tid)
        self._fit_viewport()
        self._update_legend_colors()

    def set_visible(self, tid: int, visible: bool):
        if tid in self._data:
            self._data[tid]["visible"] = visible
            self._data[tid]["curve_u"].setVisible(visible)
            self._data[tid]["curve_v"].setVisible(visible)
            self._fit_viewport()
            self._update_legend_colors()

    def set_color(self, tid: int, color: str):
        if tid in self._data:
            self._data[tid]["color"] = color
            w = self._width
            self._data[tid]["curve_u"].setPen(
                pg.mkPen(to_qcolor(color), width=w, style=_SOLID)
            )
            self._data[tid]["curve_v"].setPen(
                pg.mkPen(to_qcolor(color), width=w, style=_DASH)
            )
            self._update_legend_colors()

    def clear_all(self):
        for tid in list(self._data.keys()):
            self.remove_trajectory(tid)

    def _update_legend_colors(self):
        """
        Set the legend swatch lines to the color of the last visible
        trajectory. If no trajectories are visible, keep the current color.
        """
        color = None
        for tid in reversed(self._order):
            d = self._data.get(tid)
            if d and d["visible"]:
                color = d["color"]
                break
        if color is None:
            return  # nothing visible — leave lines as-is
        lw = self._width
        self._legend_u.setPen(pg.mkPen(to_qcolor(color), width=lw, style=_SOLID))
        self._legend_v.setPen(pg.mkPen(to_qcolor(color), width=lw, style=_DASH))

    # ------------------------------------------------------------------
    # Viewport: fit to data min/max, then lock
    # ------------------------------------------------------------------

    def _fit_viewport(self):
        t_all, y_all = [], []
        for d in self._data.values():
            if not d["visible"]:
                continue
            t_all.append(d["t"])
            y_all.append(d["u"])
            y_all.append(d["v"])

        if not t_all:
            self._x_range = (0.0, 1.0)
            self._y_range = (0.0, 1.0)
        else:
            t_cat = np.concatenate(t_all)
            y_cat = np.concatenate(y_all)

            t_min, t_max = float(np.nanmin(t_cat)), float(np.nanmax(t_cat))
            y_min, y_max = float(np.nanmin(y_cat)), float(np.nanmax(y_cat))

            if t_max == t_min:
                t_max = t_min + 1.0
            if y_max == y_min:
                y_max = y_min + 1.0

            pad_t = (t_max - t_min) * 0.03
            pad_y = (y_max - y_min) * 0.05

            self._x_range = (t_min, t_max + pad_t)
            self._y_range = (y_min - pad_y, y_max + pad_y)

        self._locked = True
        self.plot.setXRange(*self._x_range, padding=0)
        self.plot.setYRange(*self._y_range, padding=0)
        self._locked = False
