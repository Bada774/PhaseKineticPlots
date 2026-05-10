"""
models/brusselator.py – Brusselator reaction-diffusion model.

    du/dt = A - (B+1)*u + u^2*v
    dv/dt = B*u - u^2*v

Fixed point: (A, B/A).
"""

import numpy as np

META = {
    "name": "Brusselator",
    "initial_state": [0, 0.25],
}

PARAMS = {
    "A": {"default": 1.0, "min": 0.1, "max": 5.0, "label": "A (feed)"},
    "B": {"default": 3.0, "min": 0.1, "max": 10.0, "label": "B (decay)"},
}


def model(t, y, params: dict):
    A, B = params["A"], params["B"]
    u, v = y
    du = A - (B + 1) * u + u * u * v
    dv = B * u - u * u * v
    return [du, dv]
