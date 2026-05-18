"""
models/gray_scott.py – FitzHugh-Nagumo neuron model.

    du/dt = u - u^3/3 - v + I
    dv/dt = (u + a - b*v) / tau

"""

META = {"name": "Gray-Scott", "initial_state": [0.5, 0.5]}

PARAMS = {
    "F": {"default": 0.5, "min": -2.0, "max": 2.0, "label": "F feed rate"},
    "k": {"default": 0.7, "min": 0.0, "max": 2.0, "label": "k kill rate"},
}


def model(t, y, params):
    u, v = y
    F = params["F"]
    k = params["k"]
    du = -u * (v**2) + F * (1 - u)
    dv = u * (v**2) - (F + k) * v
    return [du, dv]
