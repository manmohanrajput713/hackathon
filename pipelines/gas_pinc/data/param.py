"""Three gas-lift wells from Kittelsen et al. (2024), Table 4.

Pressures are stored in pascal. Table values in bar are multiplied by 1e5
before they are used with the gas constant. P_bh is converted back to bar
only for logs, plots, targets of the algebraic network, and IAE.

D_bh is not in the table. It is set to sqrt(4 S_bh / pi) so that the
bottom-hole cross-section matches S_bh. F_bh is constant, so this choice
is a constant pressure offset.

P_0 in (18f) is not in the table. It is standard atmospheric pressure,
1.01325 bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from data.friction import haaland_lambda

BAR_TO_PA = 1.0e5
P0_BAR = 1.01325
P0_PA = P0_BAR * BAR_TO_PA
T_HORIZON = 60.0  # seconds, one PINC / MPC step

# Cubic tubing-friction coefficients, already scaled (Appendix B, Table 6).
FRICTION_CUBIC = {
    1: dict(a=-1.78e-17, b=4.56e-12, c=-4.18e-7, d=3.29e-2, Re_min=13_000.0, Re_max=115_000.0),
    2: dict(a=-1.78e-17, b=4.55e-12, c=-4.17e-7, d=3.29e-2, Re_min=13_000.0, Re_max=115_000.0),
    3: dict(a=-2.03e-18, b=8.87e-13, c=-1.48e-7, d=2.67e-2, Re_min=50_000.0, Re_max=160_000.0),
}

_WELLS = {
    # rho_L, T_an, V_an, L_an, L_bh, T_tb, GOR, P_res_bar, w_res_bar,
    # D_tb, L_tb, V_tb, PI, K_gs, K_inj, K_pr
    1: dict(
        rho_L=760.0, T_an=348.0, V_an=64.34, L_an=2048.0, L_bh=75.0, T_tb=369.4,
        GOR=0.0, P_res_bar=160.0, w_res_bar=18.0, D_tb=0.134, L_tb=2048.0,
        V_tb=25.03, PI=2.47e-6, K_gs=9.98e-5, K_inj=1.40e-4, K_pr=2.90e-3,
    ),
    2: dict(
        rho_L=760.0, T_an=335.0, V_an=84.82, L_an=2700.0, L_bh=75.0, T_tb=355.6,
        GOR=0.0, P_res_bar=165.0, w_res_bar=11.0, D_tb=0.130, L_tb=2700.0,
        V_tb=31.00, PI=2.12e-6, K_gs=10.43e-5, K_inj=1.20e-4, K_pr=2.43e-3,
    ),
    3: dict(
        rho_L=730.0, T_an=360.0, V_an=56.55, L_an=1800.0, L_bh=40.0, T_tb=381.2,
        GOR=0.2, P_res_bar=157.0, w_res_bar=30.0, D_tb=0.134, L_tb=1800.0,
        V_tb=22.08, PI=3.89e-6, K_gs=3.89e-5, K_inj=1.78e-4, K_pr=3.22e-3,
    ),
}


@dataclass(frozen=True)
class Well:
    well_id: int
    R: float
    g: float
    mu: float
    M_G: float
    P_gs: float
    S_bh: float
    eps: float
    rho_L: float
    T_an: float
    V_an: float
    L_an: float
    L_bh: float
    T_tb: float
    GOR: float
    P_res: float
    w_res_bar: float
    D_tb: float
    L_tb: float
    V_tb: float
    PI: float
    K_gs: float
    K_inj: float
    K_pr: float
    P0: float
    D_bh: float
    alpha_G_bh_m: float
    Re_bh: float
    lambda_bh: float
    F_bh: float
    cubic: dict

    @property
    def liquid_cap(self) -> float:
        """Liquid mass that fills tubing plus bottom-hole, kg."""
        return self.rho_L * (self.V_tb + self.S_bh * self.L_bh)

    @property
    def bottomhole_liquid(self) -> float:
        """rho_L * S_bh * L_bh, the liquid mass sitting in the bottom-hole."""
        return self.rho_L * self.S_bh * self.L_bh


def _build(well_id: int) -> Well:
    raw = _WELLS[well_id]
    R = 8.314
    g = 9.81
    mu = 3.64e-3
    M_G = 0.0167
    S_bh = 0.0314
    eps = 2.80e-5
    P_gs = 140.0 * BAR_TO_PA
    D_bh = math.sqrt(4.0 * S_bh / math.pi)
    gor = float(raw["GOR"])
    alpha = gor / (gor + 1.0)
    # (22d), (23d)-(23f). Constant for a given well.
    U_L_bh = raw["w_res_bar"] / (raw["rho_L"] * S_bh)
    Re_bh = raw["rho_L"] * U_L_bh * D_bh / mu
    lambda_bh, ok = haaland_lambda(Re_bh, eps, D_bh)
    if not ok:
        raise RuntimeError(f"well {well_id}: bottom-hole Haaland factor is undefined")
    F_bh = lambda_bh * raw["rho_L"] * U_L_bh**2 * raw["L_bh"] / (2.0 * D_bh)
    return Well(
        well_id=well_id,
        R=R,
        g=g,
        mu=mu,
        M_G=M_G,
        P_gs=P_gs,
        S_bh=S_bh,
        eps=eps,
        rho_L=float(raw["rho_L"]),
        T_an=float(raw["T_an"]),
        V_an=float(raw["V_an"]),
        L_an=float(raw["L_an"]),
        L_bh=float(raw["L_bh"]),
        T_tb=float(raw["T_tb"]),
        GOR=gor,
        P_res=float(raw["P_res_bar"]) * BAR_TO_PA,
        w_res_bar=float(raw["w_res_bar"]),
        D_tb=float(raw["D_tb"]),
        L_tb=float(raw["L_tb"]),
        V_tb=float(raw["V_tb"]),
        PI=float(raw["PI"]),
        K_gs=float(raw["K_gs"]),
        K_inj=float(raw["K_inj"]),
        K_pr=float(raw["K_pr"]),
        P0=P0_PA,
        D_bh=D_bh,
        alpha_G_bh_m=alpha,
        Re_bh=float(Re_bh),
        lambda_bh=float(lambda_bh),
        F_bh=float(F_bh),
        cubic=FRICTION_CUBIC[well_id],
    )


_CACHE = {i: _build(i) for i in (1, 2, 3)}


def get_well(well_id: int) -> Well:
    try:
        return _CACHE[int(well_id)]
    except KeyError as exc:
        raise KeyError(f"well id must be 1, 2, or 3, got {well_id}") from exc


# The prompt's well-1 guess. It makes the extrapolated tubing-top mixture
# density negative, so it is not a collocation point. The sweep replaces it.
PROMPT_GUESS_WELL1 = np.array([2000.0, 300.0, 8000.0], dtype=np.float64)

# Inside the feasible set, near the 95 bar operating region at u = (0.5, 0.5).
# P_bh is about 94.7 bar. Documented in readme.md.
FEASIBLE_START = {
    1: np.array([3500.0, 300.0, 8000.0], dtype=np.float64),
}


def nominal_state(well_id: int) -> np.ndarray:
    """Initial mass vector [m_G_an, m_G_tb, m_L_tb], kg.

    Well 1 returns the feasible replacement of the prompt guess, not the
    infeasible prompt guess itself. Wells 2 and 3 match well 1's annulus
    pressure and tubing liquid fill; the dataset sweep drops them if the
    ODE is undefined there.
    """
    if int(well_id) in FEASIBLE_START:
        return FEASIBLE_START[int(well_id)].copy()
    ref = get_well(1)
    well = get_well(well_id)
    p_an_ref = ref.R * ref.T_an * 3500.0 / (ref.M_G * ref.V_an)
    m_an = p_an_ref * well.M_G * well.V_an / (well.R * well.T_an)
    fill = 8000.0 / (ref.rho_L * ref.V_tb)
    m_L = fill * well.rho_L * well.V_tb
    m_G = 300.0 * (well.V_tb / ref.V_tb)
    return np.array([m_an, m_G, m_L], dtype=np.float64)


def nominal_input() -> np.ndarray:
    return np.array([0.5, 0.5], dtype=np.float64)
