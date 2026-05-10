"""
models/lotka_volterra.py – Lotka-Volterra predator-prey model.

    du/dt = α*u - β*u*v      (prey: birth - predation)
    dv/dt = δ*u*v - γ*v      (predator: growth from prey - death)

Fixed points:
  (0, 0)          – trivial, unstable saddle
  (γ/δ, α/β)     – coexistence, neutrally stable center
"""

import numpy as np

META = {
    "name": "Lotka-Volterra",
    "initial_state": [2.0, 1.0],
}

PARAMS = {
    "alpha": {"default": 1.0, "min": 0.1, "max": 4.0, "label": "α  prey birth"},
    "beta": {"default": 0.5, "min": 0.1, "max": 4.0, "label": "β  predation"},
    "delta": {"default": 0.5, "min": 0.1, "max": 4.0, "label": "δ  pred. growth"},
    "gamma": {"default": 1.0, "min": 0.1, "max": 4.0, "label": "γ  pred. death"},
}


def model(t, y, params: dict):
    a = params["alpha"]
    b = params["beta"]
    d = params["delta"]
    g = params["gamma"]
    u, v = y
    du = a * u - b * u * v
    dv = d * u * v - g * v
    return [du, dv]


def guesses(params: dict):
    a = params["alpha"]
    b = params["beta"]
    d = params["delta"]
    g = params["gamma"]
    return [
        [0.01, 0.01],  # near origin
        [g / d, a / b],  # coexistence fixed point
        [g / d + 0.5, a / b + 0.5],
        [1.0, 1.0],
    ]


def isoclines(u_arr, params: dict, **kwargs):
    a = params["alpha"]
    b = params["beta"]
    d = params["delta"]
    g = params["gamma"]

    # du/dt = 0  →  v = α/β  (horizontal line, valid for all u > 0)
    v_u = np.full_like(u_arr, a / b)

    # dv/dt = 0  →  u = γ/δ  (vertical line)
    u_star = g / d
    # Use viewport v_range if provided, otherwise fall back to a sensible default
    v_range = kwargs.get("v_range")
    if v_range is not None:
        v_lo, v_hi = float(v_range[0]), float(v_range[1])
    else:
        v_lo, v_hi = 0.0, (a / b) * 3
    x_v = np.array([u_star, u_star])
    y_v = np.array([v_lo, v_hi])

    return (u_arr, v_u), (x_v, y_v)
