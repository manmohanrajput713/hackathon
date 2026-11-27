"""Tubing friction closures.

The Runge-Kutta plant uses the Haaland explicit formula. The PINC physics
loss replaces only the tubing friction factor with the cubic in Appendix B
of Kittelsen et al. (2024), after clipping the Reynolds number. Bottom-hole
friction stays Haaland in both modes because it does not depend on the state.
"""

from __future__ import annotations

import numpy as np

# Floor applied only when the cubic itself is negative. The tabulated
# polynomials are positive on their fit intervals; the counter records
# any training evaluation that still needed the floor.
CUBIC_FLOOR_HITS = 0
LAMBDA_FLOOR = 1e-6


def note_cubic_floor(n_negative: int) -> None:
    global CUBIC_FLOOR_HITS
    if n_negative > 0:
        CUBIC_FLOOR_HITS += int(n_negative)


def haaland_lambda(re, eps: float, diameter: float):
    """Return (lambda, ok). `ok` is false when the formula is undefined.

    1/sqrt(lambda) = -1.8 * log10( (eps/D/3.7)**1.11 + 6.9/Re )   (23b, 23e)
    """
    re_arr = np.asarray(re, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = (eps / diameter / 3.7) ** 1.11 + 6.9 / re_arr
        ok = np.isfinite(re_arr) & (re_arr > 0.0) & np.isfinite(arg) & (arg > 0.0)
        arg_safe = np.where(ok, arg, 0.1)
        inv = -1.8 * np.log10(arg_safe)
        ok = ok & np.isfinite(inv) & (inv > 0.0)
        lam = 1.0 / np.where(ok, inv, 1.0) ** 2
    if np.isscalar(re):
        return float(lam), bool(ok)
    return lam, ok


def cubic_lambda(re, coeff: dict):
    """lambda(Re) = g(clip(Re, Re_min, Re_max)), floored at 1e-6.

    g(Re) = a Re^3 + b Re^2 + c Re + d. Coefficients are the already-scaled
    values from Appendix B, Table 6.
    """
    global CUBIC_FLOOR_HITS
    re_arr = np.asarray(re, dtype=np.float64)
    re_c = np.clip(re_arr, coeff["Re_min"], coeff["Re_max"])
    g = (
        coeff["a"] * re_c**3
        + coeff["b"] * re_c**2
        + coeff["c"] * re_c
        + coeff["d"]
    )
    neg = np.isfinite(g) & (g < 0.0)
    note_cubic_floor(int(np.sum(neg)))
    lam = np.maximum(g, LAMBDA_FLOOR)
    ok = np.isfinite(lam) & np.isfinite(re_arr)
    if np.isscalar(re):
        return float(lam), bool(ok)
    return lam, ok


def haaland_lambda_torch(re, eps: float, diameter: float):
    """Differentiable Haaland factor. Caller must already know Re > 0."""
    import torch

    arg = (eps / diameter / 3.7) ** 1.11 + 6.9 / re
    inv = -1.8 * torch.log10(arg)
    return 1.0 / inv**2


def cubic_lambda_torch(re, coeff: dict):
    """Differentiable cubic with Reynolds clipping and a 1e-6 floor."""
    import torch

    re_c = torch.clamp(re, min=float(coeff["Re_min"]), max=float(coeff["Re_max"]))
    g = (
        float(coeff["a"]) * re_c**3
        + float(coeff["b"]) * re_c**2
        + float(coeff["c"]) * re_c
        + float(coeff["d"])
    )
    neg = torch.isfinite(g) & (g < 0)
    if bool(neg.any()):
        note_cubic_floor(int(neg.sum().item()))
    return torch.clamp(g, min=LAMBDA_FLOOR)
