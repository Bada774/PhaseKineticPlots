"""
analysis.py – Fixed-point finding, Jacobian, stability classification,
and automatic nullcline / guess generation for models that don't provide them.
"""

import numpy as np
from scipy.optimize import fsolve, brentq
from sympy import symbols, Matrix, lambdify


def build_jacobian_func(model_func, param_names: list[str]):
    """Return callable  jac(u, v, params_dict) -> 2×2 numpy array."""
    u_sym, v_sym = symbols("u v")
    param_syms = {k: symbols(k) for k in param_names}
    derivs = model_func(0, [u_sym, v_sym], param_syms)
    J_sym = Matrix([derivs[0], derivs[1]]).jacobian([u_sym, v_sym])
    sym_args = [u_sym, v_sym] + list(param_syms.values())
    _jac_fn = lambdify(sym_args, J_sym, "numpy")

    def jac(u_val, v_val, params: dict):
        return np.array(
            _jac_fn(u_val, v_val, *[params[k] for k in param_names]), dtype=float
        )

    return jac


def find_fixed_points(model_func, params: dict, guesses: list) -> list:
    """Find equilibria; return list of [u*, v*] (duplicates removed, atol=1e-3)."""

    def eqs(p):
        return model_func(0, p, params)

    found = []
    for g in guesses:
        try:
            sol, _, ier, _ = fsolve(eqs, g, full_output=True, maxfev=200)
            if ier != 1:
                continue
            if np.max(np.abs(eqs(sol))) > 1e-8:
                continue
            if not any(np.allclose(sol, p, atol=1e-3) for p in found):
                found.append(sol.tolist())
        except Exception:
            pass
    return found


def classify_fixed_point(J, eps: float = 1e-4) -> tuple[str, str, str]:
    """Return (stability, point_type, oscillation_status) from a 2×2 Jacobian."""
    eigvals = np.linalg.eigvals(J)
    re = np.real(eigvals)
    im = np.imag(eigvals)
    has_complex = any(abs(i) > eps for i in im)

    if all(r < -eps for r in re):
        stability = "Stable"
    elif any(r > eps for r in re):
        stability = "Unstable"
    else:
        stability = "Neutral"

    if has_complex:
        mean_re = float(np.mean(re))
        if abs(mean_re) < eps:
            pt_type, osc = "Center", "Boundary of stability (conservative)"
        elif mean_re > eps:
            pt_type, osc = "Focus", "Limit cycle (auto-oscillations)"
        else:
            pt_type, osc = "Focus", "Damped oscillations"
    else:
        if float(re[0]) * float(re[1]) < -eps:
            pt_type, stability = "Saddle", "Unstable"
        else:
            pt_type = "Node"
        osc = "No oscillations"

    return stability, pt_type, osc
