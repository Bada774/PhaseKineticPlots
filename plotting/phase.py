"""
plotting/phase.py – Phase-plane plot.

Features:
  - Multiple pre-computed trajectories, each with evenly-spaced directional arrows
  - Arrow count AND head-size sliders
  - Click-to-set-initial-point callback
  - Vector field (normalised arrows on a grid), scaled to current view
  - Nullclines with adjustable thickness
  - Per-trajectory fixed-point markers and start-point markers
  - Viewport controlled by text parameters (xmin, xmax, ymin, ymax)
  - Zoom clamped: user can zoom out/in by at most ZOOM_SLACK (37.5%) beyond
    the "base" range that was set via set_viewport or Apply Viewport.
"""

import numpy as np
import pyqtgraph as pg
from color_config import to_qcolor

_SOLID = pg.QtCore.Qt.PenStyle.SolidLine
_DASH = pg.QtCore.Qt.PenStyle.DashLine

DEFAULT_X = (0.0, 5.0)
DEFAULT_Y = (0.0, 10.0)

# How much extra the user is allowed to zoom beyond the base viewport (fraction)
ZOOM_SLACK = 0.375  # ±37.5 % → covers the requested "25-50 %" band


# ---------------------------------------------------------------------------
# Arrow geometry helpers
# ---------------------------------------------------------------------------


def _arc_length(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    ds = np.sqrt(np.diff(u) ** 2 + np.diff(v) ** 2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def _make_trajectory_arrows(
    u: np.ndarray,
    v: np.ndarray,
    n_arrows: int = 18,
    head_frac: float = 0.009,
    view_diag: float | None = None,
) -> tuple:
    """
    Build arrow geometry evenly spaced along arc length.
    Returns (x_arr, y_arr, connect) arrays for PlotDataItem.

        head_frac is a fraction of view_diag (viewport diagonal in data units).
    This keeps arrow heads a fixed visual size regardless of integration time.
    """
    if len(u) < 4:
        return np.array([]), np.array([]), np.array([], dtype=bool)

    s = _arc_length(u, v)
    total = s[-1]
    if total < 1e-9:
        return np.array([]), np.array([]), np.array([], dtype=bool)

    reference = view_diag if view_diag is not None else total
    head_len = reference * head_frac
    alpha = 0.45

    positions = np.linspace(total * 0.04, total * 0.96, n_arrows)

    xs, ys, conn = [], [], []

    for sp in positions:
        idx = int(np.searchsorted(s, sp))
        idx = np.clip(idx, 1, len(u) - 2)

        tip_u = float(np.interp(sp, s, u))
        tip_v = float(np.interp(sp, s, v))

        angle = np.arctan2(v[idx] - v[idx - 1], u[idx] - u[idx - 1])

        l_u = tip_u - head_len * np.cos(angle - alpha)
        l_v = tip_v - head_len * np.sin(angle - alpha)
        r_u = tip_u - head_len * np.cos(angle + alpha)
        r_v = tip_v - head_len * np.sin(angle + alpha)

        xs += [l_u, tip_u, r_u]
        ys += [l_v, tip_v, r_v]
        conn += [True, True, False]

    return np.array(xs), np.array(ys), np.array(conn, dtype=bool)


# ---------------------------------------------------------------------------
# PhasePlot
# ---------------------------------------------------------------------------


class PhasePlot:
    def __init__(self, on_click=None):
        """
        on_click – callable(u: float, v: float) or None.
                   Called when the user left-clicks inside the plot area.
        """
        self._on_click = on_click

        self.plot = pg.PlotWidget(
            title="Phase Plane  *  left-click to set initial point"
        )
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "u")
        self.plot.setLabel("left", "v")

        axis_pen = pg.mkPen("w", width=2)
        self.plot.getAxis("left").setPen(axis_pen)
        self.plot.getAxis("bottom").setPen(axis_pen)

        font = pg.QtGui.QFont()
        font.setPixelSize(13)
        self.plot.getAxis("left").setTickFont(font)
        self.plot.getAxis("bottom").setTickFont(font)

        # Zero lines
        self._zero_v = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen(to_qcolor("#777777ff"), width=1)
        )
        self._zero_h = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen(to_qcolor("#777777ff"), width=1)
        )
        self.plot.addItem(self._zero_v)
        self.plot.addItem(self._zero_h)

        # Nullclines — colors stored explicitly so width changes never corrupt them
        self._nullcline_width: int = 1
        self._nc_tick_count: int = 5  # number of tick marks per nullcline
        self._nc_tick_len: float = 2.0  # tick mark half-length in data units
        self._iso_u_color: str = "#00d4ffff"  # matches DEFAULT_ELEMENT_COLORS["iso_u"]
        self._iso_v_color: str = "#ff6666ff"  # matches DEFAULT_ELEMENT_COLORS["iso_v"]
        self._iso_u = self.plot.plot(
            pen=pg.mkPen(
                to_qcolor(self._iso_u_color), width=self._nullcline_width, style=_DASH
            )
        )
        self._iso_v = self.plot.plot(
            pen=pg.mkPen(
                to_qcolor(self._iso_v_color), width=self._nullcline_width, style=_DASH
            )
        )
        # Tick-mark items (replaced on each nullcline update)
        self._iso_u_ticks = pg.PlotDataItem(
            pen=pg.mkPen(to_qcolor(self._iso_u_color), width=self._nullcline_width)
        )
        self._iso_v_ticks = pg.PlotDataItem(
            pen=pg.mkPen(to_qcolor(self._iso_v_color), width=self._nullcline_width)
        )
        self.plot.addItem(self._iso_u_ticks)
        self.plot.addItem(self._iso_v_ticks)

        # Raw numpy caches – always valid, unlike PlotDataItem.xData which can be None
        self._iso_u_xy: tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
        self._iso_v_xy: tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))

        # Vector field item (replaced on every refresh)
        self._vf_item = None
        self._vf_color = "#666666"

        # Per-trajectory items:
        #   tid -> {curve, arrows, fp_marker, start_marker}
        self._traj_items: dict[int, dict] = {}
        self._traj_width: int = 2
        self._arrow_head_frac: float = 0.009
        self._vf_length_mult: float = 1.0  # controlled by the VF length slider

        # Base viewport (set_viewport / Apply Viewport sets this)
        self._base_x = DEFAULT_X
        self._base_y = DEFAULT_Y

        # View
        vb = self.plot.getViewBox()
        vb.setMenuEnabled(False)
        vb.setMouseEnabled(x=True, y=True)
        self.plot.setXRange(*DEFAULT_X)
        self.plot.setYRange(*DEFAULT_Y)

        # Pending initial-point marker (circle)
        self._init_marker = self.plot.plot(
            pen=None,
            symbol="o",
            symbolSize=14,
            symbolBrush="#00ff88",
            symbolPen=pg.mkPen("w", width=2),
        )

        # Click
        self.plot.scene().sigMouseClicked.connect(self._on_scene_click)

        # Disable auto-range so adding curves never shifts the viewport
        self.plot.getViewBox().disableAutoRange()

        # Clamp zoom when range changes; also rescale ticks to new viewport
        self._clamping = False
        self._rebuilding_ticks = False
        self.plot.sigRangeChanged.connect(self._on_range_changed)

    def _on_range_changed(self):
        """Handle range changes: clamp zoom only.
        Tick geometry is rebuilt by update_nullclines() after new data is
        written, so we must NOT rebuild here with stale cached coordinates."""
        self._clamp_zoom()

    # ------------------------------------------------------------------
    # Zoom clamping
    # ------------------------------------------------------------------

    def _allowed_range(self):
        """Return (x_min_allowed, x_max_allowed, y_min_allowed, y_max_allowed)."""
        bx0, bx1 = self._base_x
        by0, by1 = self._base_y
        bw = bx1 - bx0
        bh = by1 - by0
        slack_x = bw * ZOOM_SLACK
        slack_y = bh * ZOOM_SLACK
        return (bx0 - slack_x, bx1 + slack_x, by0 - slack_y, by1 + slack_y)

    def _clamp_zoom(self):
        if self._clamping or self._rebuilding_ticks:
            return
        xr, yr = self.plot.viewRange()
        ax0, ax1, ay0, ay1 = self._allowed_range()

        cx0 = max(xr[0], ax0)
        cx1 = min(xr[1], ax1)
        cy0 = max(yr[0], ay0)
        cy1 = min(yr[1], ay1)

        # Also enforce minimum zoom-in (must show at least 10% of base range)
        min_w = (self._base_x[1] - self._base_x[0]) * 0.10
        min_h = (self._base_y[1] - self._base_y[0]) * 0.10
        if cx1 - cx0 < min_w:
            mid = (cx0 + cx1) / 2
            cx0, cx1 = mid - min_w / 2, mid + min_w / 2
        if cy1 - cy0 < min_h:
            mid = (cy0 + cy1) / 2
            cy0, cy1 = mid - min_h / 2, mid + min_h / 2

        changed = (
            abs(cx0 - xr[0]) > 1e-9
            or abs(cx1 - xr[1]) > 1e-9
            or abs(cy0 - yr[0]) > 1e-9
            or abs(cy1 - yr[1]) > 1e-9
        )
        if changed:
            self._clamping = True
            self.plot.setXRange(cx0, cx1, padding=0)
            self.plot.setYRange(cy0, cy1, padding=0)
            self._clamping = False

    # ------------------------------------------------------------------
    # View
    # ------------------------------------------------------------------

    def set_viewport(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Set phase plot viewport from text fields and update base range."""
        if x_max > x_min and y_max > y_min:
            self._base_x = (x_min, x_max)
            self._base_y = (y_min, y_max)
            self._clamping = True
            self.plot.setXRange(x_min, x_max, padding=0)
            self.plot.setYRange(y_min, y_max, padding=0)
            self._clamping = False
            self._rebuild_nullcline_ticks()

    # ------------------------------------------------------------------
    # Appearance
    # ------------------------------------------------------------------

    def apply_theme(self, theme: dict):
        self.plot.setBackground(theme["plot_bg"])
        axis_color = theme["axis_color"]
        tick_color = theme.get("tick_color", axis_color)

        axis_pen = pg.mkPen(color=axis_color, width=2)
        tick_pen = pg.mkPen(color=tick_color)
        label_css = f"color: {tick_color}; font-size: 13px;"

        for axis_name in ("left", "bottom"):
            axis = self.plot.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(tick_pen)
            # Re-apply the axis label with the themed colour
            axis.setLabel(axis.labelText, **{"color": tick_color})

        # Plot title
        pi = self.plot.getPlotItem()
        pi.titleLabel.setAttr("color", tick_color)
        pi.titleLabel.setText(pi.titleLabel.text)

        zpn = pg.mkPen(color=theme["zero_line_color"], width=1)
        self._zero_v.setPen(zpn)
        self._zero_h.setPen(zpn)

        self.plot.showGrid(x=True, y=True, alpha=theme.get("grid_alpha", 0.25))

    def apply_element_colors(self, colors: dict):
        self._iso_u_color = colors["iso_u"]
        self._iso_v_color = colors["iso_v"]
        self._iso_u.setPen(
            pg.mkPen(
                to_qcolor(self._iso_u_color), width=self._nullcline_width, style=_DASH
            )
        )
        self._iso_v.setPen(
            pg.mkPen(
                to_qcolor(self._iso_v_color), width=self._nullcline_width, style=_DASH
            )
        )
        self._iso_u_ticks.setPen(
            pg.mkPen(to_qcolor(self._iso_u_color), width=self._nullcline_width)
        )
        self._iso_v_ticks.setPen(
            pg.mkPen(to_qcolor(self._iso_v_color), width=self._nullcline_width)
        )
        self._vf_color = colors["vector_field"]
        if self._vf_item is not None:
            self._vf_item.setPen(pg.mkPen(to_qcolor(self._vf_color), width=1))
        fp_color = colors.get("fixed_point", "#ff3333ff")
        for d in self._traj_items.values():
            d["fp_marker"].setSymbolBrush(fp_color)
            d["fp_marker"].setSymbolPen(pg.mkPen(to_qcolor(fp_color), width=2))

    def set_line_width(self, width: int):
        self._traj_width = width
        for d in self._traj_items.values():
            color = d["color"]
            d["curve"].setPen(pg.mkPen(to_qcolor(color), width=width, style=_SOLID))
            d["arrows"].setPen(pg.mkPen(to_qcolor(color), width=max(1, width - 1)))

    def set_nullcline_width(self, width: int):
        self._nullcline_width = width
        self._iso_u.setPen(
            pg.mkPen(to_qcolor(self._iso_u_color), width=width, style=_DASH)
        )
        self._iso_v.setPen(
            pg.mkPen(to_qcolor(self._iso_v_color), width=width, style=_DASH)
        )
        self._iso_u_ticks.setPen(pg.mkPen(to_qcolor(self._iso_u_color), width=width))
        self._iso_v_ticks.setPen(pg.mkPen(to_qcolor(self._iso_v_color), width=width))

    def set_nullcline_ticks(self, count: int, length: float):
        """count: number of tick marks; length: half-length in data units."""
        self._nc_tick_count = count
        self._nc_tick_len = length
        # Tick geometry is rebuilt on the next update_nullclines call.
        # Trigger a visual refresh on the cached data if available.
        self._rebuild_nullcline_ticks()

    def _arc_length_1d(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        return np.concatenate([[0.0], np.cumsum(ds)])

    def _make_nc_ticks(
        self,
        x: np.ndarray,
        y: np.ndarray,
        count: int,
        half_len_x: float,
        half_len_y: float,
        orientation: str,  # "h" → horizontal ticks (u-nullcline); "v" → vertical (v-nullcline)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build tick mark geometry along a nullcline.

        Tick direction is FIXED by convention (not perpendicular-to-tangent),
        so the two nullclines always look visually distinct:
          "h"  →  horizontal ticks  (du/dt=0 nullcline: flow is vertical here)
          "v"  →  vertical ticks    (dv/dt=0 nullcline: flow is horizontal here)

        half_len_x / half_len_y are viewport-relative lengths in data units.
        Returns (px, py, connect) arrays suitable for PlotDataItem.
        """
        if len(x) < 2 or count < 1:
            return np.array([]), np.array([]), np.array([], dtype=bool)

        s = self._arc_length_1d(x, y)
        total = s[-1]
        if total < 1e-9:
            return np.array([]), np.array([]), np.array([], dtype=bool)

        positions = np.linspace(total * 0.05, total * 0.95, count)
        px, py, conn = [], [], []

        for sp in positions:
            cx = float(np.interp(sp, s, x))
            cy = float(np.interp(sp, s, y))

            if orientation == "h":
                # Horizontal tick: varies in x, fixed in y
                px += [cx - half_len_x, cx + half_len_x]
                py += [cy, cy]
            else:
                # Vertical tick: fixed in x, varies in y
                px += [cx, cx]
                py += [cy - half_len_y, cy + half_len_y]

            conn += [True, False]  # draw segment, then break

        return np.array(px), np.array(py), np.array(conn, dtype=bool)

    def _rebuild_nullcline_ticks(self):
        """Re-compute tick geometry from cached nullcline data.

        Tick half-length is expressed as a fraction of the BASE viewport
        (_base_x / _base_y) rather than the live viewRange().
        """
        if self._rebuilding_ticks:
            return
        self._rebuilding_ticks = True
        try:
            vw = max(self._base_x[1] - self._base_x[0], 1e-9)
            vh = max(self._base_y[1] - self._base_y[0], 1e-9)

            # 1% of each axis per unit of user tick-length setting.
            # At nc_tick_len=3 (intended default) → 3% of view per axis, clearly visible.
            scale = 0.01 * self._nc_tick_len
            hl_x = vw * scale  # half-length in x data-units  (for horizontal ticks)
            hl_y = vh * scale  # half-length in y data-units  (for vertical ticks)

            xu, yu = self._iso_u_xy
            xv, yv = self._iso_v_xy

            # --- u-nullcline: horizontal ticks (du/dt=0 → flow is vertical here) ---
            if len(xu) > 1 and self._nc_tick_count > 0:
                px, py, conn = self._make_nc_ticks(
                    xu, yu, self._nc_tick_count, hl_x, hl_y, "h"
                )
            else:
                px, py, conn = np.array([]), np.array([]), np.array([], dtype=bool)

            self.plot.removeItem(self._iso_u_ticks)
            self._iso_u_ticks = pg.PlotDataItem(
                px,
                py,
                connect=conn if len(conn) else "all",
                pen=pg.mkPen(to_qcolor(self._iso_u_color), width=self._nullcline_width),
            )
            self._add(self._iso_u_ticks)

            # --- v-nullcline: vertical ticks (dv/dt=0 → flow is horizontal here) ---
            if len(xv) > 1 and self._nc_tick_count > 0:
                px, py, conn = self._make_nc_ticks(
                    xv, yv, self._nc_tick_count, hl_x, hl_y, "v"
                )
            else:
                px, py, conn = np.array([]), np.array([]), np.array([], dtype=bool)

            self.plot.removeItem(self._iso_v_ticks)
            self._iso_v_ticks = pg.PlotDataItem(
                px,
                py,
                connect=conn if len(conn) else "all",
                pen=pg.mkPen(to_qcolor(self._iso_v_color), width=self._nullcline_width),
            )
            self._add(self._iso_v_ticks)
        finally:
            self._rebuilding_ticks = False

    def set_arrow_head_size(self, frac: float):
        self._arrow_head_frac = frac

    def set_vf_length(self, mult: float):
        """Set the vector-field arrow length multiplier (1.0 = default)."""
        self._vf_length_mult = mult

    def _add(self, item):
        """Add an item to the plot and immediately re-disable auto-range.

        pyqtgraph's PlotItem.plot() / addItem() internally calls
        enableAutoRange() which would let new trajectory data resize the
        viewport.  We always want the viewport to stay fixed after the user
        (or set_viewport) sets it.
        """
        self.plot.addItem(item)
        self.plot.getViewBox().disableAutoRange()

    # ------------------------------------------------------------------
    # Initial-point marker
    # ------------------------------------------------------------------

    def set_init_marker(self, u: float, v: float):
        self._init_marker.setData([u], [v])

    def clear_init_marker(self):
        self._init_marker.setData([], [])

    # ------------------------------------------------------------------
    # Click handler
    # ------------------------------------------------------------------

    def _on_scene_click(self, event):
        if self._on_click is None:
            return
        if event.button() != pg.QtCore.Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        pt = vb.mapSceneToView(pos)
        self._on_click(pt.x(), pt.y())

    # ------------------------------------------------------------------
    # Trajectories
    # ------------------------------------------------------------------

    def add_trajectory(
        self,
        tid: int,
        u: np.ndarray,
        v: np.ndarray,
        color: str,
        visible: bool = True,
        n_arrows: int = 18,
        u0: float | None = None,
        v0: float | None = None,
        fixed_points: list | None = None,
    ):
        curve = pg.PlotDataItem(
            u,
            v,
            pen=pg.mkPen(to_qcolor(color), width=self._traj_width, style=_SOLID),
        )
        self._add(curve)

        xr, yr = self.plot.viewRange()
        vw = xr[1] - xr[0]
        vh = yr[1] - yr[0]
        view_diag = (vw**2 + vh**2) ** 0.5

        ax, ay, ac = _make_trajectory_arrows(
            u,
            v,
            n_arrows=n_arrows,
            head_frac=self._arrow_head_frac,
            view_diag=view_diag,
        )
        if len(ax):
            arrows = pg.PlotDataItem(
                ax,
                ay,
                connect=ac,
                pen=pg.mkPen(to_qcolor(color), width=max(1, self._traj_width - 1)),
            )
        else:
            arrows = pg.PlotDataItem([], [])
        self._add(arrows)

        # Per-trajectory fixed-point marker
        fp_marker = pg.PlotDataItem(
            pen=None,
            symbol="+",
            symbolSize=16,
            symbolBrush=to_qcolor("#ff3333ff"),
            symbolPen=pg.mkPen(to_qcolor("#ff3333ff"), width=2),
        )
        if fixed_points:
            fp_marker.setData(
                [p[0] for p in fixed_points], [p[1] for p in fixed_points]
            )
        self._add(fp_marker)

        # Per-trajectory start marker (circle)
        start_marker = pg.PlotDataItem(
            pen=None,
            symbol="o",
            symbolSize=10,
            symbolBrush=to_qcolor(color),
            symbolPen=pg.mkPen("w", width=1),
        )
        if u0 is not None and v0 is not None:
            start_marker.setData([u0], [v0])
        self._add(start_marker)

        self._traj_items[tid] = dict(
            color=color,
            curve=curve,
            arrows=arrows,
            fp_marker=fp_marker,
            start_marker=start_marker,
        )
        self.set_visible(tid, visible)

    def remove_trajectory(self, tid: int):
        if tid in self._traj_items:
            d = self._traj_items.pop(tid)
            for key, item in d.items():
                if key == "color":
                    continue
                self.plot.removeItem(item)

    def set_visible(self, tid: int, visible: bool):
        if tid in self._traj_items:
            for key, item in self._traj_items[tid].items():
                if key == "color":
                    continue
                item.setVisible(visible)

    def set_color(self, tid: int, color: str):
        if tid in self._traj_items:
            d = self._traj_items[tid]
            d["color"] = color
            d["curve"].setPen(
                pg.mkPen(to_qcolor(color), width=self._traj_width, style=_SOLID)
            )
            d["arrows"].setPen(
                pg.mkPen(to_qcolor(color), width=max(1, self._traj_width - 1))
            )
            d["start_marker"].setSymbolBrush(to_qcolor(color))

    def clear_all_trajectories(self):
        for tid in list(self._traj_items.keys()):
            self.remove_trajectory(tid)

    # ------------------------------------------------------------------
    # Fixed points (kept for backward compat)
    # ------------------------------------------------------------------

    def set_fixed_points(self, points: list):
        pass

    # ------------------------------------------------------------------
    # Vector field  – arrows scaled to current view range
    # ------------------------------------------------------------------

    def update_vector_field(self, model_func, params: dict, density: int = 15):
        if self._vf_item is not None:
            self.plot.removeItem(self._vf_item)
            self._vf_item = None

        xr, yr = self.plot.viewRange()
        x_min, x_max = xr[0], xr[1]
        y_min, y_max = yr[0], yr[1]
        if x_max <= x_min or y_max <= y_min:
            return

        x_pts = np.linspace(x_min, x_max, density)
        y_pts = np.linspace(y_min, y_max, density)
        X, Y = np.meshgrid(x_pts, y_pts)

        try:
            res = model_func(0, [X, Y], params)
        except Exception:
            return

        U = np.array(res[0], dtype=float)
        V = np.array(res[1], dtype=float)

        speed = np.sqrt(U**2 + V**2) + 1e-9
        U /= speed
        V /= speed

        # Scale arrow length relative to current view extent
        view_w = x_max - x_min
        view_h = y_max - y_min
        # Use ~60% of grid spacing as arrow length so they don't overlap
        grid_step_x = view_w / max(density - 1, 1)
        grid_step_y = view_h / max(density - 1, 1)
        scale = min(grid_step_x, grid_step_y) * 0.55 * self._vf_length_mult

        # Head length ≈ 30% of arrow body, but capped so it's visible
        head_len = scale * 0.35
        alpha = 0.38

        x0 = X.flatten()
        y0 = Y.flatten()

        # Rotate arrows to point FROM grid point TOWARD destination
        uf = U.flatten()
        vf = V.flatten()
        x1 = x0 + uf * scale
        y1 = y0 + vf * scale
        ang = np.arctan2(vf, uf)

        xl = x1 - head_len * np.cos(ang - alpha)
        yl = y1 - head_len * np.sin(ang - alpha)
        xr2 = x1 - head_len * np.cos(ang + alpha)
        yr2 = y1 - head_len * np.sin(ang + alpha)

        n = len(x0)
        allx = np.empty(n * 6)
        ally = np.empty(n * 6)
        conn = np.ones(n * 6, dtype=bool)
        for i in range(n):
            idx = i * 6
            allx[idx : idx + 6] = [x0[i], x1[i], xl[i], x1[i], xr2[i], x1[i]]
            ally[idx : idx + 6] = [y0[i], y1[i], yl[i], y1[i], yr2[i], y1[i]]
            conn[idx + 5] = False

        self._vf_item = pg.PlotDataItem(
            allx,
            ally,
            connect=conn,
            pen=pg.mkPen(to_qcolor(self._vf_color), width=1),
        )
        self._add(self._vf_item)

    # ------------------------------------------------------------------
    # Nullclines
    # ------------------------------------------------------------------

    def update_nullclines(self, isocline_func, params: dict):
        xr, yr = self.plot.viewRange()
        u_min, u_max = xr[0], xr[1]
        v_min, v_max = yr[0], yr[1]
        if u_max <= u_min:
            self.clear_nullclines()
            return

        u_arr = np.linspace(u_min, u_max, 500)
        (x1, y1), (x2, y2) = isocline_func(u_arr, params, v_range=(v_min, v_max))
        _xu, _yu = np.asarray(x1), np.asarray(y1)
        _xv, _yv = np.asarray(x2), np.asarray(y2)
        self._iso_u_xy = (_xu, _yu)
        self._iso_v_xy = (_xv, _yv)
        self._iso_u.setData(_xu, _yu)
        self._iso_v.setData(_xv, _yv)
        self._rebuild_nullcline_ticks()

    def clear_nullclines(self):
        self._iso_u_xy = (np.array([]), np.array([]))
        self._iso_v_xy = (np.array([]), np.array([]))
        self._iso_u.setData([], [])
        self._iso_v.setData([], [])
        self._rebuild_nullcline_ticks()
