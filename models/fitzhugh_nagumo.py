"""
models/fitzhugh_nagumo.py – FitzHugh-Nagumo neuron model.

    du/dt = u - u^3/3 - v + I
    dv/dt = (u + a - b*v) / tau

This file intentionally has NO guesses() and NO isoclines() functions.
Both are generated automatically by the framework, demonstrating the
minimum required model definition.
"""

META = {
    "name": "FitzHugh-Nagumo",
    "initial_state": [-1.0, -0.5],
    "viewport": (-2.5, 2.5, -1.0, 2.0),
}

PARAMS = {
    "I": {"default": 0.5, "min": -2.0, "max": 2.0, "label": "I  stimulus"},
    "a": {"default": 0.7, "min": 0.0, "max": 2.0, "label": "a"},
    "b": {"default": 0.8, "min": 0.1, "max": 2.0, "label": "b"},
    "tau": {"default": 12.5, "min": 1.0, "max": 30.0, "label": "tau  time scale"},
}


def model(t, y, params):
    u, v = y
    I = params["I"]
    a = params["a"]
    b = params["b"]
    tau = params["tau"]
    du = u - (u**3) / 3.0 - v + I
    dv = (u + a - b * v) / tau
    return [du, dv]
