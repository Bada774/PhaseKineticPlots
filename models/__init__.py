"""
models/__init__.py - Auto-discovery registry.

To add a new model, drop a .py file in this directory that exposes:

    META = {
        "name":          str,          # display name in the selector
        "initial_state": [u0, v0],     # starting y values for the integrator
    }

    PARAMS = {
        "param_name": {
            "default": float,
            "min":     float,
            "max":     float,
            "label":   str,            # text shown next to the slider
        },
        ...
    }

    def model(t, y, params: dict) -> [du, dv]:
        ...

The two functions below are OPTIONAL - if absent the framework generates
sensible defaults automatically:

    def guesses(params: dict) -> list[list[float, float]]:
        # Return candidate starting points for fsolve.
        ...

    def isoclines(u_arr, params: dict) -> ((x1, y1), (x2, y2)):
        # Return two (x, y) array pairs for the two nullclines.
        ...

The registry scans this directory at import time and registers every
module that has META, PARAMS, and model.
"""

import importlib
import pkgutil
import traceback
from pathlib import Path

import numpy as np

import logging

_log = logging.getLogger(__name__)

MODELS: dict[str, dict] = {}

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Fallback generators
# ---------------------------------------------------------------------------


def _default_guesses(meta: dict, params: dict):
    """Grid of starting points around the initial state."""
    u0, v0 = meta.get("initial_state", [1.0, 1.0])
    pts = []
    for su in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for sv in [0.1, 0.5, 1.0, 2.0, 5.0]:
            pts.append([max(u0 * su, 1e-6), max(v0 * sv, 1e-6)])
    pts.append([u0, v0])
    return pts


def _default_isoclines(
    model_func, u_arr: np.ndarray, params: dict, v_range: tuple | None = None
):
    """
    Numerically estimate nullclines by scanning v for sign changes in f and g.
    Samples ~80 u values for speed; interpolates crossing position linearly.

    v_range: (v_min, v_max) from the current viewport - covers negative values.
    Falls back to a positive heuristic if not provided.
    """
    if v_range is not None:
        v_lo, v_hi = v_range
        # Add 20% margin beyond viewport so nullclines don't clip at the edge
        margin = (v_hi - v_lo) * 0.2
        v_lo -= margin
        v_hi += margin
    else:
        v_lo = -max(float(np.max(np.abs(u_arr))) * 3.0, 10.0)
        v_hi = max(float(np.max(np.abs(u_arr))) * 3.0, 10.0)
    v_scan = np.linspace(v_lo, v_hi, 400)

    iso_u_x, iso_u_y = [], []
    iso_v_x, iso_v_y = [], []

    step = max(1, len(u_arr) // 80)
    for u in u_arr[::step]:
        f_vals = np.array([model_func(0, [u, v], params)[0] for v in v_scan])
        g_vals = np.array([model_func(0, [u, v], params)[1] for v in v_scan])

        for vals, xu, yu in ((f_vals, iso_u_x, iso_u_y), (g_vals, iso_v_x, iso_v_y)):
            signs = np.sign(vals)
            for ci in np.where(np.diff(signs) != 0)[0]:
                denom = vals[ci + 1] - vals[ci]
                frac = -vals[ci] / denom if abs(denom) > 1e-30 else 0.5
                v_cross = v_scan[ci] + (v_scan[ci + 1] - v_scan[ci]) * frac
                xu.append(float(u))
                yu.append(float(v_cross))

    if not iso_u_x:
        iso_u_x, iso_u_y = [u_arr[0], u_arr[-1]], [0.0, 0.0]
    if not iso_v_x:
        iso_v_x, iso_v_y = [u_arr[0], u_arr[-1]], [0.0, 0.0]

    return (
        (np.array(iso_u_x), np.array(iso_u_y)),
        (np.array(iso_v_x), np.array(iso_v_y)),
    )


# ---------------------------------------------------------------------------
# Registry scan
# ---------------------------------------------------------------------------

_REQUIRED = ("META", "PARAMS", "model")

for _finder, _mod_name, _is_pkg in pkgutil.iter_modules([str(_HERE)]):
    try:
        _mod = importlib.import_module(f".{_mod_name}", package=__name__)
    except Exception:
        _log.error(
            f"[models] Could not import '{_mod_name}':\n{traceback.format_exc()}"
        )
        continue

    if not all(hasattr(_mod, attr) for attr in _REQUIRED):
        _missing = [a for a in _REQUIRED if not hasattr(_mod, a)]
        _log.info(
            f"[models] Skipping '{_mod_name}': missing required attributes {_missing}"
        )
        continue

    _meta = _mod.META
    _display = _meta["name"]
    _model_func = _mod.model

    # guesses: use module-provided or fall back to grid
    if hasattr(_mod, "guesses"):
        _guesses_fn = _mod.guesses
    else:
        # capture _meta by value so each closure is independent
        def _guesses_fn(params, _m=_meta):
            return _default_guesses(_m, params)

        _log.info(f"[models] '{_display}': no guesses() - using auto grid")

    # isoclines: use module-provided or fall back to numerical scan
    if hasattr(_mod, "isoclines"):
        # Wrap so calling with v_range= keyword (from update_nullclines) is safe
        def _isoclines_fn(u_arr, params, _iso=_mod.isoclines, **kw):
            return _iso(u_arr, params, **kw)

    else:

        def _isoclines_fn(u_arr, params, _f=_model_func, **kw):
            return _default_isoclines(_f, u_arr, params, **kw)

        _log.info(f"[models] '{_display}': no isoclines() - using numerical scan")

    MODELS[_display] = {
        "module": _mod,
        "func": _model_func,
        "params_spec": _mod.PARAMS,
        "initial_state": _meta["initial_state"],
        "guesses": _guesses_fn,
        "isoclines": _isoclines_fn,
        "domain_check": getattr(_mod, "domain_check", None),
    }
    _log.info(f"[models] Registered '{_display}'")
