"""
app.py – Main application window (trajectory-based, no live simulation).

Workflow:
  1. Choose a model and set its parameters.
  2. Set an initial point by clicking the phase plot or typing coordinates.
  3. Click "Add Trajectory" to compute and render the full solution.
  4. Repeat with different initial points / parameter values.
  5. Each trajectory has its own color, visibility toggle, and delete button.
  6. Fixed points and start marker are computed per trajectory and shown
     on the phase plot bound to that trajectory.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QSlider,
    QLabel,
    QComboBox,
    QCheckBox,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QLineEdit,
    QSpinBox,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QLocale, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QDoubleValidator

# ---------------------------------------------------------------------------
# Locale-safe float parsing
# ---------------------------------------------------------------------------

# Force C locale for all validators so the decimal separator is always '.'
# regardless of the OS regional settings (e.g. German/Russian locale uses ',')
_C_LOCALE = QLocale(QLocale.Language.C)
_C_LOCALE.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)


def _make_validator(
    lo: float = -1e18, hi: float = 1e18, decimals: int = 8
) -> QDoubleValidator:
    """Return a QDoubleValidator locked to the C locale (dot as decimal)."""
    v = QDoubleValidator(lo, hi, decimals)
    v.setLocale(_C_LOCALE)
    v.setNotation(QDoubleValidator.Notation.StandardNotation)
    return v


def _parse_float(text: str) -> float:
    """
    Parse a float from user input, accepting both '.' and ',' as the decimal
    separator.  Raises ValueError if the string cannot be converted.
    """
    return float(text.replace(",", "."))


def _install_dot_filter(field: "QLineEdit"):
    """
    Silently replace any comma the user types with a dot in real-time.
    This guarantees dot-as-decimal regardless of OS locale.
    """

    def _on_changed(text: str):
        if "," in text:
            fixed = text.replace(",", ".")
            field.blockSignals(True)
            field.setText(fixed)
            field.blockSignals(False)

    field.textChanged.connect(_on_changed)


from models import MODELS
from analysis import build_jacobian_func, find_fixed_points, classify_fixed_point
from plotting.kinetic import KineticPlot
from plotting.phase import PhasePlot
from themes import THEMES
from color_config import (
    DEFAULT_ELEMENT_COLORS,
    ELEMENT_COLOR_LABELS,
    TRAJECTORY_PALETTE,
    SwatchButton,
    to_qcolor,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T_MAX_DEFAULT = 50.0
N_EVAL = 3000  # integration evaluation points


# ---------------------------------------------------------------------------
# Background integration worker
# ---------------------------------------------------------------------------


class IntegrationWorker(QThread):
    """
    Runs solve_ivp and find_fixed_points in a background thread so the UI
    stays fully responsive even when an integration diverges or takes long.

    Emits ``finished(sol, fixed_pts)`` on the main thread when done.
    sol is the scipy OdeSolution object (or None on exception).
    fixed_pts is a (possibly empty) list of [u*, v*] equilibria.
    """

    finished = pyqtSignal(object, object)

    def __init__(
        self,
        model_func,
        params: dict,
        guesses_fn,
        jac_func,
        u0: float,
        v0: float,
        t_max: float,
        parent=None,
    ):
        super().__init__(parent)
        self._model_func = model_func
        self._params = dict(params)  # snapshot – immune to slider changes
        self._guesses_fn = guesses_fn
        self._jac_func = jac_func
        self._u0, self._v0 = u0, v0
        self._t_max = t_max

    def run(self):
        t_eval = np.linspace(0.0, self._t_max, N_EVAL)

        # --- Termination events ---
        # 1. State-magnitude limit: fires when |u| or |v| exceeds threshold.
        #    Catches trajectories that blow up toward ±∞.
        _STATE_LIMIT = 1e4

        def _state_event(t, y):
            return _STATE_LIMIT - max(abs(y[0]), abs(y[1]))

        _state_event.terminal = True
        _state_event.direction = -1  # fire on downward crossing (value → 0)

        # 2. Derivative-rate limit: fires when the ODE rhs grows very large.
        #    Catches divergent trajectories *earlier* than the state event,
        #    before the integrator is forced into tiny steps.
        _RATE_LIMIT = 1e6

        def _rate_event(t, y):
            try:
                dy = self._model_func(t, y, self._params)
                return _RATE_LIMIT - max(abs(dy[0]), abs(dy[1]))
            except Exception:
                return 0.0  # treat any model error as "rate exceeded"

        _rate_event.terminal = True
        _rate_event.direction = -1

        # --- Integrate ---
        sol = None
        try:
            sol = solve_ivp(
                lambda t, y: self._model_func(t, y, self._params),
                (0.0, self._t_max),
                [self._u0, self._v0],
                t_eval=t_eval,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
                dense_output=False,
                events=[_state_event, _rate_event],
                max_step=self._t_max / 200,
            )
        except Exception:
            pass  # sol stays None; caller will handle it

        # --- Fixed-point search (pure numpy/scipy, thread-safe) ---
        fixed_pts = []
        if self._jac_func is not None:
            try:
                fixed_pts = find_fixed_points(
                    self._model_func,
                    self._params,
                    self._guesses_fn(self._params),
                )
            except Exception:
                pass

        self.finished.emit(sol, fixed_pts)


# ---------------------------------------------------------------------------
# Trajectory data model
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    tid: int
    label: str
    params: dict
    u0: float
    v0: float
    color: str
    visible: bool = True
    t_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    u_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    v_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    fixed_points: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# TrajectoryRow widget
# ---------------------------------------------------------------------------


class TrajectoryRow(QWidget):
    """One row in the trajectory list: [swatch] [label] [eye] [x]"""

    def __init__(
        self,
        traj: Trajectory,
        on_color_change,
        on_visibility_change,
        on_delete,
        parent=None,
    ):
        super().__init__(parent)
        self._tid = traj.tid

        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(4)

        # Color swatch
        self._swatch = SwatchButton(
            traj.color,
            callback=lambda c: on_color_change(self._tid, c),
        )
        row.addWidget(self._swatch)

        # Label
        lbl = QLabel(traj.label)
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lbl.setToolTip(
            f"u0={traj.u0:.4g}  v0={traj.v0:.4g}\n"
            + "\n".join(f"{k}={v:.4g}" for k, v in traj.params.items())
        )
        row.addWidget(lbl)

        # Visibility checkbox
        self._eye = QCheckBox()
        self._eye.setChecked(traj.visible)
        self._eye.setToolTip("Show / hide")
        self._eye.stateChanged.connect(
            lambda s: on_visibility_change(self._tid, bool(s))
        )
        row.addWidget(self._eye)

        # Delete button
        btn_del = QPushButton("x")
        btn_del.setFixedSize(22, 22)
        btn_del.setToolTip("Remove trajectory")
        btn_del.clicked.connect(lambda: on_delete(self._tid))
        row.addWidget(btn_del)

    def update_color(self, hex_color: str):
        self._swatch.set_color(hex_color)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phase & Kinetic Plotter")
        self.setMinimumSize(1100, 720)

        # --- state ---
        self._theme_name = "Grey"
        self._elem_colors = dict(DEFAULT_ELEMENT_COLORS)
        self._traj_list: list[Trajectory] = []
        self._traj_rows: dict[int, TrajectoryRow] = {}
        self._next_tid = 0
        self._palette_idx = 0
        self._integ_pending: dict | None = None  # set while worker runs
        self._worker: IntegrationWorker | None = None
        self._pending_u0 = 1.0
        self._pending_v0 = 1.0
        self._t_max = T_MAX_DEFAULT
        self._show_nullclines = False

        # model state (populated by _load_model)
        self._model_name = ""
        self.model_func = None
        self.params_spec = {}
        self.params: dict = {}
        self._jac_func = None

        # --- build UI ---
        self._build_ui()

        first = next(iter(MODELS))
        self._load_model(first, update_selector=False)
        self._apply_theme(self._theme_name)
        self._on_arrow_head_changed()
        self._refresh_phase_field()

        # Debounce timer: pan/zoom events fire on every pixel of mouse movement.
        # Instead of recalculating the vector field and nullclines on every event,
        # we wait 50 ms after the last range change before doing heavy work.
        self._refresh_timer = QTimer()
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(50)
        self._refresh_timer.timeout.connect(self._refresh_phase_field)

        self.phase.plot.sigRangeChanged.connect(self._on_range_changed)

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_model(self, name: str, update_selector: bool = True):
        cfg = MODELS[name]
        self._model_name = name
        self.model_func = cfg["func"]
        self.params_spec = cfg["params_spec"]
        self.params = {k: v["default"] for k, v in self.params_spec.items()}

        if update_selector:
            self.model_selector.blockSignals(True)
            self.model_selector.setCurrentText(name)
            self.model_selector.blockSignals(False)

        init = cfg["initial_state"]
        self._pending_u0 = float(init[0])
        self._pending_v0 = float(init[1])
        self._sync_coord_fields()

        self._rebuild_param_sliders()

        param_names = list(self.params_spec.keys())
        self._jac_func = build_jacobian_func(self.model_func, param_names)

        # Apply per-model viewport if defined in META
        vp = cfg["module"].META.get("viewport")
        if vp and hasattr(self, "phase"):
            xmin, xmax, ymin, ymax = vp
            self._vp_xmin.setText(f"{xmin:.3g}")
            self._vp_xmax.setText(f"{xmax:.3g}")
            self._vp_ymin.setText(f"{ymin:.3g}")
            self._vp_ymax.setText(f"{ymax:.3g}")
            self.phase.set_viewport(xmin, xmax, ymin, ymax)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        self.kinetic = KineticPlot()
        self.phase = PhasePlot(on_click=self._on_phase_click)

        plots = QVBoxLayout()
        plots.addWidget(self.kinetic.plot)
        plots.addWidget(self.phase.plot)

        self._ctrl_layout = QVBoxLayout()
        self._ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._ctrl_layout.setSpacing(5)
        self._build_controls()

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(self._ctrl_layout)

        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setWidget(ctrl_widget)
        ctrl_scroll.setMinimumWidth(250)
        ctrl_scroll.setMaximumWidth(320)
        ctrl_scroll.setFrameShape(QFrame.Shape.NoFrame)

        root = QHBoxLayout()
        root.addLayout(plots, stretch=4)
        root.addWidget(ctrl_scroll, stretch=1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    def _build_controls(self):
        cl = self._ctrl_layout

        # -- Model ----------------------------------------------------------
        cl.addWidget(self._section("Model"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(list(MODELS.keys()))
        self.model_selector.currentTextChanged.connect(self._on_model_changed)
        cl.addWidget(self.model_selector)

        # -- Parameters (dynamic) -------------------------------------------
        cl.addWidget(self._section("Parameters"))
        self._param_container = QWidget()
        pcl = QVBoxLayout(self._param_container)
        pcl.setContentsMargins(0, 0, 0, 0)
        pcl.setSpacing(2)
        cl.addWidget(self._param_container)

        # -- Integration ----------------------------------------------------
        cl.addWidget(self._section("Integration"))
        self._t_max_label = QLabel()
        self._t_max_slider = self._make_slider(
            1, 500, int(self._t_max), self._on_t_max_changed
        )
        cl.addWidget(self._t_max_label)
        cl.addWidget(self._t_max_slider)
        self._update_t_max_label()

        # -- Initial Point --------------------------------------------------
        cl.addWidget(self._section("Initial Point"))
        cl.addWidget(QLabel("Click the phase plot  -or-  enter manually:"))

        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("u0"))
        self._field_u0 = self._make_coord_field(self._pending_u0)
        coord_row.addWidget(self._field_u0)
        coord_row.addWidget(QLabel("v0"))
        self._field_v0 = self._make_coord_field(self._pending_v0)
        coord_row.addWidget(self._field_v0)
        cl.addLayout(coord_row)

        self._field_u0.editingFinished.connect(self._on_coords_typed)
        self._field_v0.editingFinished.connect(self._on_coords_typed)

        self._btn_add = QPushButton("Add Trajectory")
        self._btn_add.clicked.connect(self._add_trajectory)
        cl.addWidget(self._btn_add)

        btn_clear_all = QPushButton("Clear All Trajectories")
        btn_clear_all.clicked.connect(self._clear_all_trajectories)
        cl.addWidget(btn_clear_all)

        # -- Trajectory List ------------------------------------------------
        cl.addWidget(self._section("Trajectories"))
        self._traj_container = QWidget()
        self._traj_container.setLayout(QVBoxLayout())
        self._traj_container.layout().setContentsMargins(0, 0, 0, 0)
        self._traj_container.layout().setSpacing(2)

        traj_scroll = QScrollArea()
        traj_scroll.setWidgetResizable(True)
        traj_scroll.setWidget(self._traj_container)
        traj_scroll.setMinimumHeight(80)
        traj_scroll.setMaximumHeight(200)
        traj_scroll.setFrameShape(QFrame.Shape.NoFrame)
        cl.addWidget(traj_scroll)

        # Line width
        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Line width:"))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 6)
        self._width_spin.setValue(2)
        self._width_spin.valueChanged.connect(self._on_width_changed)
        width_row.addWidget(self._width_spin)
        width_row.addStretch()
        cl.addLayout(width_row)

        # -- Phase Plane ----------------------------------------------------
        cl.addWidget(self._section("Phase Plane"))

        self._density_label = QLabel()
        self._density_slider = self._make_slider(4, 30, 14, self._on_density_changed)
        cl.addWidget(self._density_label)
        cl.addWidget(self._density_slider)
        self._update_density_label()

        self._arrows_label = QLabel()
        self._arrows_slider = self._make_slider(0, 40, 12, self._on_arrows_changed)
        cl.addWidget(self._arrows_label)
        cl.addWidget(self._arrows_slider)
        self._update_arrows_label()

        self._arrow_head_label = QLabel()
        self._arrow_head_slider = self._make_slider(
            1, 20, 4, self._on_arrow_head_changed
        )
        cl.addWidget(self._arrow_head_label)
        cl.addWidget(self._arrow_head_slider)
        self._update_arrow_head_label()

        self._vf_length_label = QLabel()
        self._vf_length_slider = self._make_slider(
            10, 300, 100, self._on_vf_length_changed
        )
        cl.addWidget(self._vf_length_label)
        cl.addWidget(self._vf_length_slider)
        self._update_vf_length_label()

        self._check_nullclines = QCheckBox("Show Nullclines")
        self._check_nullclines.stateChanged.connect(self._on_nullclines_toggled)
        cl.addWidget(self._check_nullclines)

        # Container for nullcline appearance controls - shown only when nullclines are on
        self._nc_controls = QWidget()
        nc_layout = QVBoxLayout(self._nc_controls)
        nc_layout.setContentsMargins(0, 0, 0, 0)
        nc_layout.setSpacing(3)

        nc_row = QHBoxLayout()
        nc_row.addWidget(QLabel("Nullcline width:"))
        self._nc_width_spin = QSpinBox()
        self._nc_width_spin.setRange(1, 6)
        self._nc_width_spin.setValue(1)
        self._nc_width_spin.valueChanged.connect(self._on_nc_width_changed)
        nc_row.addWidget(self._nc_width_spin)
        nc_row.addStretch()
        nc_layout.addLayout(nc_row)

        # Nullcline tick marks
        self._nc_tick_count_label = QLabel()
        self._nc_tick_count_slider = self._make_slider(
            0, 20, 5, self._on_nc_tick_count_changed
        )
        nc_layout.addWidget(self._nc_tick_count_label)
        nc_layout.addWidget(self._nc_tick_count_slider)
        self._update_nc_tick_count_label()

        self._nc_tick_len_label = QLabel()
        self._nc_tick_len_slider = self._make_slider(
            1, 5, 2, self._on_nc_tick_len_changed
        )
        nc_layout.addWidget(self._nc_tick_len_label)
        nc_layout.addWidget(self._nc_tick_len_slider)
        self._update_nc_tick_len_label()

        self._nc_controls.setVisible(False)
        cl.addWidget(self._nc_controls)

        # -- Phase Plot Viewport --------------------------------------------
        cl.addWidget(self._section("Phase Plot Viewport"))
        vp_grid = QGridLayout()
        vp_grid.setSpacing(3)

        vp_grid.addWidget(QLabel("x min:"), 0, 0)
        self._vp_xmin = QLineEdit("0.0")
        self._vp_xmin.setLocale(_C_LOCALE)
        self._vp_xmin.setFixedWidth(60)
        _install_dot_filter(self._vp_xmin)
        vp_grid.addWidget(self._vp_xmin, 0, 1)

        vp_grid.addWidget(QLabel("x max:"), 0, 2)
        self._vp_xmax = QLineEdit("5.0")
        self._vp_xmax.setLocale(_C_LOCALE)
        self._vp_xmax.setFixedWidth(60)
        _install_dot_filter(self._vp_xmax)
        vp_grid.addWidget(self._vp_xmax, 0, 3)

        vp_grid.addWidget(QLabel("y min:"), 1, 0)
        self._vp_ymin = QLineEdit("0.0")
        self._vp_ymin.setLocale(_C_LOCALE)
        self._vp_ymin.setFixedWidth(60)
        _install_dot_filter(self._vp_ymin)
        vp_grid.addWidget(self._vp_ymin, 1, 1)

        vp_grid.addWidget(QLabel("y max:"), 1, 2)
        self._vp_ymax = QLineEdit("10.0")
        self._vp_ymax.setLocale(_C_LOCALE)
        self._vp_ymax.setFixedWidth(60)
        _install_dot_filter(self._vp_ymax)
        vp_grid.addWidget(self._vp_ymax, 1, 3)

        cl.addLayout(vp_grid)

        btn_apply_vp = QPushButton("Apply Viewport")
        btn_apply_vp.clicked.connect(self._apply_viewport)
        cl.addWidget(btn_apply_vp)

        # -- Appearance -----------------------------------------------------
        cl.addWidget(self._section("Appearance"))

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Theme:"))
        self._theme_selector = QComboBox()
        self._theme_selector.addItems(list(THEMES.keys()))
        self._theme_selector.setCurrentText(self._theme_name)
        self._theme_selector.currentTextChanged.connect(self._on_theme_changed)
        theme_row.addWidget(self._theme_selector)
        cl.addLayout(theme_row)

        cl.addWidget(QLabel("Plot element colors:"))
        self._elem_swatch: dict[str, SwatchButton] = {}
        for key, label in ELEMENT_COLOR_LABELS.items():
            row = QHBoxLayout()
            row.setContentsMargins(0, 1, 0, 1)
            lbl = QLabel(label)
            lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )

            def make_cb(k):
                return lambda c: self._on_elem_color_changed(k, c)

            sw = SwatchButton(self._elem_colors[key], make_cb(key))
            self._elem_swatch[key] = sw
            row.addWidget(lbl)
            row.addWidget(sw)
            cl.addLayout(row)

        btn_reset_colors = QPushButton("Reset Element Colors")
        btn_reset_colors.clicked.connect(self._reset_elem_colors)
        cl.addWidget(btn_reset_colors)

        # -- Analysis -------------------------------------------------------
        cl.addWidget(self._section("Stability Analysis"))
        self._info_label = QLabel("Add a trajectory to see fixed points.")
        self._info_label.setWordWrap(True)
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setWidget(self._info_label)
        info_scroll.setMinimumHeight(150)
        info_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._info_scroll = info_scroll
        cl.addWidget(info_scroll)

    # -----------------------------------------------------------------------
    # Dynamic parameter sliders
    # -----------------------------------------------------------------------

    def _rebuild_param_sliders(self):
        layout = self._param_container.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._param_inputs: dict[str, QLineEdit] = {}

        for key, spec in self.params_spec.items():
            mn, mx = spec["min"], spec["max"]

            row_widget = QWidget()
            row_lay = QHBoxLayout(row_widget)
            row_lay.setContentsMargins(0, 1, 0, 1)
            row_lay.setSpacing(4)

            lbl = QLabel(f"{spec['label']}:")
            lbl.setMinimumWidth(70)
            row_lay.addWidget(lbl)

            field = QLineEdit(f"{self.params[key]:.6g}")
            field.setValidator(_make_validator(mn, mx, 8))
            field.setLocale(_C_LOCALE)
            field.setFixedWidth(80)
            field.setToolTip(f"Range: [{mn}, {mx}]")
            row_lay.addWidget(field)

            range_lbl = QLabel(f"[{mn}, {mx}]")
            range_lbl.setStyleSheet("font-size:10px; color:#888;")
            row_lay.addWidget(range_lbl)
            row_lay.addStretch()

            def make_slot(k):
                def slot():
                    try:
                        val = _parse_float(self._param_inputs[k].text())
                        self.params[k] = val
                        self._refresh_phase_field()
                    except ValueError:
                        pass

                return slot

            field.editingFinished.connect(make_slot(key))
            self._param_inputs[key] = field
            _install_dot_filter(field)
            layout.addWidget(row_widget)

    # -----------------------------------------------------------------------
    # Coordinate helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_coord_field(value: float) -> QLineEdit:
        f = QLineEdit(f"{value:.4g}")
        f.setValidator(_make_validator(-1e9, 1e9, 6))
        f.setLocale(_C_LOCALE)
        f.setFixedWidth(70)
        _install_dot_filter(f)
        return f

    def _sync_coord_fields(self):
        self._field_u0.setText(f"{self._pending_u0:.4g}")
        self._field_v0.setText(f"{self._pending_v0:.4g}")
        self.phase.set_init_marker(self._pending_u0, self._pending_v0)

    def _read_coord_fields(self) -> tuple[float, float] | None:
        try:
            u = _parse_float(self._field_u0.text())
            v = _parse_float(self._field_v0.text())
            return u, v
        except ValueError:
            return None

    # -----------------------------------------------------------------------
    # Trajectory operations
    # -----------------------------------------------------------------------

    def _add_trajectory(self):
        coords = self._read_coord_fields()
        if coords is None:
            return
        u0, v0 = coords

        # --- Domain validation ---
        cfg = MODELS[self._model_name]
        domain_check = cfg.get("domain_check")
        if domain_check is not None:
            ok, msg = domain_check(u0, v0)
            if not ok:
                QMessageBox.warning(self, "Invalid initial conditions", msg)
                return

        color = TRAJECTORY_PALETTE[self._palette_idx % len(TRAJECTORY_PALETTE)]
        self._palette_idx += 1

        tid = self._next_tid
        self._next_tid += 1

        n_traj = len(self._traj_list) + 1
        label = f"#{n_traj}  u0={u0:.3g}  v0={v0:.3g}"

        # Stash everything needed once the worker finishes
        self._integ_pending = {
            "u0": u0,
            "v0": v0,
            "color": color,
            "tid": tid,
            "label": label,
            "n_arrows": self._arrows_slider.value(),
        }

        # Disable button for the duration of the background integration
        self._btn_add.setEnabled(False)
        self._btn_add.setText("Integrating…")

        self._worker = IntegrationWorker(
            model_func=self.model_func,
            params=self.params,
            guesses_fn=cfg["guesses"],
            jac_func=self._jac_func,
            u0=u0,
            v0=v0,
            t_max=self._t_max,
            parent=self,
        )
        self._worker.finished.connect(self._on_integration_done)
        self._worker.start()

    # ------------------------------------------------------------------

    def _on_integration_done(self, sol, fixed_pts):
        """Called on the main thread when the background worker finishes."""
        self._btn_add.setEnabled(True)
        self._btn_add.setText("Add Trajectory")

        p = self._integ_pending
        self._integ_pending = None
        if p is None:
            return

        u0, v0 = p["u0"], p["v0"]
        color = p["color"]
        tid = p["tid"]
        label = p["label"]
        n_arrows = p["n_arrows"]

        traj = Trajectory(
            tid=tid,
            label=label,
            params=dict(self.params),
            u0=u0,
            v0=v0,
            color=color,
            fixed_points=fixed_pts,
        )
        if sol is not None and sol.success:
            traj.t_arr = sol.t
            traj.u_arr = sol.y[0]
            traj.v_arr = sol.y[1]
        else:
            traj.t_arr = np.array([0.0])
            traj.u_arr = np.array([u0])
            traj.v_arr = np.array([v0])

        self._traj_list.append(traj)

        # --- Phase plot ---
        self.phase.add_trajectory(
            tid,
            traj.u_arr,
            traj.v_arr,
            color,
            n_arrows=n_arrows,
            u0=u0,
            v0=v0,
            fixed_points=fixed_pts,
        )

        # --- Kinetic plot ---
        self.kinetic.add_trajectory(
            tid,
            traj.t_arr,
            traj.u_arr,
            traj.v_arr,
            color,
            params=traj.params,
            u0=traj.u0,
            v0=traj.v0,
        )

        # --- Row widget ---
        row = TrajectoryRow(
            traj,
            on_color_change=self._on_traj_color_changed,
            on_visibility_change=self._on_traj_visibility_changed,
            on_delete=self._remove_trajectory,
        )
        self._traj_rows[tid] = row
        self._traj_container.layout().addWidget(row)

        self._refresh_analysis_panel()

    def _remove_trajectory(self, tid: int):
        self._traj_list = [t for t in self._traj_list if t.tid != tid]
        self.phase.remove_trajectory(tid)
        self.kinetic.remove_trajectory(tid)
        if tid in self._traj_rows:
            row = self._traj_rows.pop(tid)
            row.setParent(None)
            row.deleteLater()
        self._refresh_analysis_panel()

    def _clear_all_trajectories(self):
        for t in list(self._traj_list):
            self._remove_trajectory(t.tid)

    def _on_traj_color_changed(self, tid: int, color: str):
        for t in self._traj_list:
            if t.tid == tid:
                t.color = color
                break
        self.phase.set_color(tid, color)
        self.kinetic.set_color(tid, color)
        self._refresh_analysis_panel()

    def _on_traj_visibility_changed(self, tid: int, visible: bool):
        for t in self._traj_list:
            if t.tid == tid:
                t.visible = visible
                break
        self.phase.set_visible(tid, visible)
        self.kinetic.set_visible(tid, visible)
        self._refresh_analysis_panel()

    # -----------------------------------------------------------------------
    # Phase plot interaction
    # -----------------------------------------------------------------------

    def _on_phase_click(self, u: float, v: float):
        self._pending_u0 = u
        self._pending_v0 = v
        self._sync_coord_fields()

    def _on_coords_typed(self):
        coords = self._read_coord_fields()
        if coords:
            self._pending_u0, self._pending_v0 = coords
            self.phase.set_init_marker(self._pending_u0, self._pending_v0)

    # -----------------------------------------------------------------------
    # Field, nullclines, analysis
    # -----------------------------------------------------------------------

    def _refresh_phase_field(self):
        density = self._density_slider.value()
        self.phase.update_vector_field(self.model_func, self.params, density=density)
        if self._show_nullclines:
            self.phase.update_nullclines(
                MODELS[self._model_name]["isoclines"], self.params
            )
        else:
            self.phase.clear_nullclines()

    # -----------------------------------------------------------------------
    # Stability Analysis Panel
    # -----------------------------------------------------------------------

    def _refresh_analysis_panel(self):
        """
        Rebuild the stability analysis HTML for ALL trajectories that
        currently have a visible plot item.  Trajectories with the same
        initial point may appear multiple times (one block each).
        """
        theme = THEMES[self._theme_name]

        # Collect visible trajectories
        visible_trajs = [t for t in self._traj_list if t.visible]

        if not visible_trajs:
            if self._traj_list:
                self._info_label.setText(
                    "<i>No visible trajectories.  Toggle visibility to see analysis.</i>"
                )
            else:
                self._info_label.setText("Add a trajectory to see fixed points.")
            return

        html_parts = []
        for traj in visible_trajs:
            block = self._analysis_block_html(traj, theme)
            html_parts.append(block)

        self._info_label.setText("<hr>".join(html_parts))

    def _analysis_block_html(self, traj: Trajectory, theme: dict) -> str:
        """Return the HTML analysis block for a single trajectory."""
        hc = theme["header_color"]
        itc = theme["info_text"]
        pts = traj.fixed_points

        # Header with trajectory colour swatch + label
        css_color = "rgba({},{},{},{})".format(
            *[
                to_qcolor(traj.color).red(),
                to_qcolor(traj.color).green(),
                to_qcolor(traj.color).blue(),
                to_qcolor(traj.color).alpha(),
            ]
        )
        swatch = (
            f"<span style='background:{css_color};"
            f"color:{css_color};padding:0 6px;border-radius:3px;'>"
            f"&#9632;</span>"
        )
        header = (
            f"{swatch}&nbsp;"
            f"<span style='color:{hc};font-size:13px;'>"
            f"<b>{traj.label}</b></span><br>"
            f"<span style='font-size:10px;color:{itc};opacity:0.7;'>"
            + "  ".join(f"{k}={v:.4g}" for k, v in traj.params.items())
            + "</span>"
        )

        if not pts:
            return header + "<br><i style='font-size:11px;'>No fixed points found.</i>"

        rows = []
        for i, pt in enumerate(pts):
            J = self._jac_func(pt[0], pt[1], traj.params)
            s, t_type, osc = classify_fixed_point(J)

            # Color-code stability
            if s == "Stable":
                sc = "#44dd88"
            elif s == "Unstable":
                sc = "#ff5555"
            else:
                sc = "#ffcc44"

            rows.append(
                f"<b>P{i+1}</b> "
                f"<span style='font-size:11px;'>({pt[0]:.3f},&nbsp;{pt[1]:.3f})</span><br>"
                f"<span style='color:{sc};'>{s}</span> "
                f"<b>{t_type}</b> - <i>{osc}</i>"
            )

        body = "<br><br>".join(rows)
        return header + "<br>" + body

    def _on_range_changed(self):
        # Sync the viewport text fields immediately so they always reflect
        # the current view (cheap — just four setText calls).
        xr, yr = self.phase.plot.viewRange()
        self._vp_xmin.setText(f"{xr[0]:.3g}")
        self._vp_xmax.setText(f"{xr[1]:.3g}")
        self._vp_ymin.setText(f"{yr[0]:.3g}")
        self._vp_ymax.setText(f"{yr[1]:.3g}")
        # Restart the debounce timer — the actual vector field / nullcline
        # recalculation fires 120 ms after the last range event, not on every
        # pixel of mouse movement.
        self._refresh_timer.start()

    # -----------------------------------------------------------------------
    # Slots – controls
    # -----------------------------------------------------------------------

    def _on_model_changed(self, name: str):
        self._clear_all_trajectories()
        self._load_model(name)
        self._refresh_phase_field()

    def _on_t_max_changed(self):
        self._t_max = float(self._t_max_slider.value())
        self._update_t_max_label()

    def _on_density_changed(self):
        self._update_density_label()
        self._refresh_phase_field()

    def _on_arrows_changed(self):
        self._update_arrows_label()
        w = self._width_spin.value()
        n = self._arrows_slider.value()
        for traj in self._traj_list:
            self.phase.remove_trajectory(traj.tid)
            self.phase.add_trajectory(
                traj.tid,
                traj.u_arr,
                traj.v_arr,
                traj.color,
                traj.visible,
                n_arrows=n,
                u0=traj.u0,
                v0=traj.v0,
                fixed_points=traj.fixed_points,
            )
        self.phase.set_line_width(w)

    def _on_arrow_head_changed(self):
        self._update_arrow_head_label()
        raw = self._arrow_head_slider.value()

        frac = 0.005 + (raw - 1) / 19.0 * 0.05
        self.phase.set_arrow_head_size(frac)
        self._on_arrows_changed()

    def _on_vf_length_changed(self):
        self._update_vf_length_label()
        mult = self._vf_length_slider.value() / 100.0
        self.phase.set_vf_length(mult)
        self._refresh_phase_field()

    def _on_nc_width_changed(self, w: int):
        self.phase.set_nullcline_width(w)
        if self._show_nullclines:
            self._refresh_phase_field()

    def _on_nc_tick_count_changed(self):
        self._update_nc_tick_count_label()
        self.phase.set_nullcline_ticks(
            self._nc_tick_count_slider.value(),
            self._nc_tick_len_slider.value(),
        )

    def _on_nc_tick_len_changed(self):
        self._update_nc_tick_len_label()
        self.phase.set_nullcline_ticks(
            self._nc_tick_count_slider.value(),
            self._nc_tick_len_slider.value(),
        )

    def _on_nullclines_toggled(self, state: int):
        self._show_nullclines = bool(state)
        self._nc_controls.setVisible(self._show_nullclines)
        self._refresh_phase_field()

    def _on_width_changed(self, w: int):
        self.phase.set_line_width(w)
        self.kinetic.set_line_width(w)

    def _on_theme_changed(self, name: str):
        self._apply_theme(name)

    def _apply_theme(self, name: str):
        self._theme_name = name
        theme = THEMES[name]
        QApplication.instance().setStyleSheet(theme["stylesheet"])
        self.kinetic.apply_theme(theme)
        self.phase.apply_theme(theme)
        self._info_label.setStyleSheet(
            f"color:{theme['info_text']};"
            f"background-color:{theme['info_bg']};"
            "font-size:13px; padding:8px;"
        )
        self._info_scroll.setStyleSheet(
            f"background-color:{theme['info_bg']}; border:none;"
        )
        self._refresh_analysis_panel()

    def _on_elem_color_changed(self, key: str, color: str):
        self._elem_colors[key] = color
        self.phase.apply_element_colors(self._elem_colors)

    def _reset_elem_colors(self):
        self._elem_colors = dict(DEFAULT_ELEMENT_COLORS)
        for key, sw in self._elem_swatch.items():
            sw.set_color(self._elem_colors[key])
        self.phase.apply_element_colors(self._elem_colors)

    def _apply_viewport(self):
        try:
            xmin = _parse_float(self._vp_xmin.text())
            xmax = _parse_float(self._vp_xmax.text())
            ymin = _parse_float(self._vp_ymin.text())
            ymax = _parse_float(self._vp_ymax.text())
        except ValueError:
            return
        self.phase.set_viewport(xmin, xmax, ymin, ymax)

    # -----------------------------------------------------------------------
    # Label helpers
    # -----------------------------------------------------------------------

    def _update_t_max_label(self):
        self._t_max_label.setText(f"Integration time T = {self._t_max:.0f}")

    def _update_density_label(self):
        self._density_label.setText(
            f"Vector field density: {self._density_slider.value()}"
        )

    def _update_arrows_label(self):
        self._arrows_label.setText(
            f"Arrows per trajectory: {self._arrows_slider.value()}"
        )

    def _update_arrow_head_label(self):
        self._arrow_head_label.setText(
            f"Arrow head size: {self._arrow_head_slider.value()}"
        )

    def _update_vf_length_label(self):
        self._vf_length_label.setText(
            f"Vector field arrow length: {self._vf_length_slider.value()}%"
        )

    def _update_nc_tick_count_label(self):
        self._nc_tick_count_label.setText(
            f"Nullcline tick marks: {self._nc_tick_count_slider.value()}"
        )

    def _update_nc_tick_len_label(self):
        self._nc_tick_len_label.setText(
            f"Nullcline tick length: {self._nc_tick_len_slider.value()}"
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(f"<b>{text}</b>")
        lbl.setContentsMargins(0, 10, 0, 2)
        return lbl

    @staticmethod
    def _make_slider(mn: int, mx: int, init: int, slot) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(mn, mx)
        s.setValue(init)
        s.valueChanged.connect(slot)
        return s
