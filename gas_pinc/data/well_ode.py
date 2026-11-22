"""Explicit Jahanshahi gas-lift ODE (Kittelsen et al. 2024, Appendix A).

State y = [m_G_an, m_G_tb, m_L_tb] in kg.
Input u = [u1, u2], production choke and gas-lift choke, each in [0, 1].

ode_numpy is the Runge-Kutta plant (Haaland friction, plain square roots).
ode_torch is the training graph (cubic tubing friction and sqrt_safe when
friction='cubic'). Both implement the same mass, pressure, and density
equations. A point with a non-positive denominator, a non-positive density,
a non-positive Haaland argument, or a non-finite intermediate raises
InfeasibleState. Nothing is replaced by zero.
"""

from __future__ import annotations

import numpy as np
import torch

from data.friction import cubic_lambda, cubic_lambda_torch, haaland_lambda, haaland_lambda_torch
from data.param import Well, get_well

SQRT_SAFE_FLOOR = 1e-3


class InfeasibleState(ValueError):
    """The ODE is undefined at this (state, input)."""


def _as_well(well) -> Well:
    if isinstance(well, Well):
        return well
    return get_well(well)


def _batch_yu(y, u):
    y_arr = np.asarray(y, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)
    scalar = y_arr.ndim == 1
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(1, -1)
    if y_arr.shape[1] != 3 or u_arr.shape[1] != 2:
        raise ValueError(f"expected y (..., 3) and u (..., 2), got {y_arr.shape}, {u_arr.shape}")
    if u_arr.shape[0] == 1 and y_arr.shape[0] != 1:
        u_arr = np.repeat(u_arr, y_arr.shape[0], axis=0)
    if y_arr.shape[0] == 1 and u_arr.shape[0] != 1:
        y_arr = np.repeat(y_arr, u_arr.shape[0], axis=0)
    if y_arr.shape[0] != u_arr.shape[0]:
        raise ValueError(f"batch mismatch y {y_arr.shape} u {u_arr.shape}")
    return y_arr, u_arr, scalar


def _mark(ok, cond):
    return ok & cond


def evaluate_numpy(y, u, well, friction: str = "haaland") -> dict:
    """Vectorized evaluation. `ok` is false on rows that must be rejected.

    Infeasible rows are not returned as a physical derivative: their `dy`
    entries are nan, and `ode_numpy` raises before handing them to a caller.
    """
    if friction not in ("haaland", "cubic"):
        raise ValueError(f"friction must be 'haaland' or 'cubic', got {friction}")
    p = _as_well(well)
    y_arr, u_arr, scalar = _batch_yu(y, u)
    n = y_arr.shape[0]
    m_an = y_arr[:, 0]
    m_gt = y_arr[:, 1]
    m_lt = y_arr[:, 2]
    u1 = u_arr[:, 0]
    u2 = u_arr[:, 1]
    ok = np.isfinite(y_arr).all(axis=1) & np.isfinite(u_arr).all(axis=1)

    # (20c), (20d), (21a) from the state alone.
    denom = p.V_tb + p.S_bh * p.L_bh - m_lt / p.rho_L
    ok = _mark(ok, np.isfinite(denom) & (denom > 0.0) & (m_gt > 0.0) & (m_an > 0.0) & (m_lt > 0.0))
    denom_s = np.where(ok, denom, 1.0)
    rho_G_tb_t = m_gt / denom_s
    ok = _mark(ok, np.isfinite(rho_G_tb_t) & (rho_G_tb_t > 0.0))
    rho_mix = (m_gt + m_lt - p.rho_L * p.S_bh * p.L_bh) / p.V_tb
    ok = _mark(ok, np.isfinite(rho_mix) & (rho_mix > 0.0))
    alpha_L_bar = (m_lt - p.rho_L * p.S_bh * p.L_bh) / (p.V_tb * p.rho_L)

    # (19a), (19b), (20a), (20b).
    P_an_t = p.R * p.T_an * m_an / (p.M_G * p.V_an)
    P_an_b = P_an_t + m_an * p.g * p.L_an / p.V_an
    rho_G_an_b = P_an_b * p.M_G / (p.R * p.T_an)
    rho_G_in = p.P_gs * p.M_G / (p.R * p.T_an)
    ok = _mark(ok, np.isfinite(P_an_t) & np.isfinite(P_an_b) & (rho_G_an_b > 0.0) & (rho_G_in > 0.0))

    # (18a). Plain sqrt on the plant. The training graph clamps separately.
    rad_in = rho_G_in * np.maximum(p.P_gs - P_an_t, 0.0)
    ok = _mark(ok, np.isfinite(rad_in) & (rad_in >= 0.0))
    w_G_in = p.K_gs * u2 * np.sqrt(np.maximum(rad_in, 0.0))

    # (22a)-(22c). (22b) uses w_G_in, and the constant w_res_bar, on purpose.
    area_tb = np.pi * p.D_tb**2
    U_L = 4.0 * (1.0 - p.alpha_G_bh_m) * p.w_res_bar / (p.rho_L * area_tb)
    U_G = 4.0 * (w_G_in + p.alpha_G_bh_m * p.w_res_bar) / (rho_G_tb_t * area_tb)
    U_mix = U_L + U_G
    ok = _mark(ok, np.isfinite(U_mix) & (U_mix > 0.0))

    Re_tb = rho_mix * U_mix * p.D_tb / p.mu
    if friction == "haaland":
        lam_tb, lam_ok = haaland_lambda(Re_tb, p.eps, p.D_tb)
        ok = _mark(ok, lam_ok & np.isfinite(Re_tb) & (Re_tb > 0.0))
    else:
        lam_tb, lam_ok = cubic_lambda(Re_tb, p.cubic)
        ok = _mark(ok, lam_ok)
    # (23c)
    F_tb = alpha_L_bar * lam_tb * rho_mix * U_mix**2 * p.L_tb / (2.0 * p.D_tb)

    # (19c)-(19e)
    P_tb_t = rho_G_tb_t * p.R * p.T_tb / p.M_G
    P_tb_b = P_tb_t + rho_mix * p.g * p.L_tb + F_tb
    P_bh = P_tb_b + p.F_bh + p.rho_L * p.g * p.L_bh
    ok = _mark(ok, np.isfinite(P_bh) & np.isfinite(P_tb_t) & np.isfinite(P_tb_b) & (P_tb_b > 0.0))

    # (18b)-(18e)
    rad_inj = rho_G_an_b * np.maximum(P_an_b - P_tb_b, 0.0)
    ok = _mark(ok, np.isfinite(rad_inj) & (rad_inj >= 0.0))
    w_G_inj = p.K_inj * np.sqrt(np.maximum(rad_inj, 0.0))
    w_res = p.PI * np.maximum(p.P_res - P_bh, 0.0)
    w_L_res = (1.0 - p.alpha_G_bh_m) * w_res
    w_G_res = p.alpha_G_bh_m * w_res

    # (20e), (21c)-(21e), (20f)
    rho_G_tb_b = P_tb_b * p.M_G / (p.R * p.T_tb)
    ok = _mark(ok, np.isfinite(rho_G_tb_b) & (rho_G_tb_b > 0.0))
    denom_alpha = w_L_res * rho_G_tb_b + (w_G_inj + w_G_res) * p.rho_L
    ok = _mark(ok, np.isfinite(denom_alpha) & (denom_alpha > 0.0))
    denom_alpha_s = np.where(ok, denom_alpha, 1.0)
    alpha_L_b = (w_L_res * rho_G_tb_b) / denom_alpha_s
    alpha_L_t = 2.0 * alpha_L_bar - alpha_L_b
    rho_mix_t = alpha_L_t * p.rho_L + (1.0 - alpha_L_t) * rho_G_tb_t
    ok = _mark(ok, np.isfinite(rho_mix_t) & (rho_mix_t > 0.0) & np.isfinite(alpha_L_t))
    rho_mix_t_s = np.where(ok, rho_mix_t, 1.0)
    alpha_G_m = ((1.0 - alpha_L_t) * rho_G_tb_t) / rho_mix_t_s
    ok = _mark(ok, np.isfinite(alpha_G_m))

    # (18f)-(18h)
    rad_out = rho_mix_t * np.maximum(P_tb_t - p.P0, 0.0)
    ok = _mark(ok, np.isfinite(rad_out) & (rad_out >= 0.0))
    w_out = p.K_pr * u1 * np.sqrt(np.maximum(rad_out, 0.0))
    w_L_out = (1.0 - alpha_G_m) * w_out
    w_G_out = alpha_G_m * w_out

    dy = np.column_stack(
        [
            w_G_in - w_G_inj,
            w_G_inj + w_G_res - w_G_out,
            w_L_res - w_L_out,
        ]
    )
    ok = _mark(ok, np.isfinite(dy).all(axis=1))
    dy = np.where(ok[:, None], dy, np.nan)

    def _maybe(arr):
        return arr[0] if scalar else arr

    return {
        "dy": _maybe(dy),
        "ok": bool(ok[0]) if scalar else ok,
        "P_bh": _maybe(P_bh),
        "P_bh_bar": _maybe(P_bh / 1.0e5),
        "w_G_in": _maybe(w_G_in),
        "w_G_res": _maybe(w_G_res),
        "w_L_res": _maybe(w_L_res),
        "w_G_inj": _maybe(w_G_inj),
        "w_out": _maybe(w_out),
        "Re_tb": _maybe(Re_tb),
        "lambda_tb": _maybe(lam_tb),
        "F_tb": _maybe(F_tb),
        "scalar": scalar,
    }


def ode_numpy(y, u, well, friction: str = "haaland") -> np.ndarray:
    """State derivative. Raises InfeasibleState instead of returning zeros or nan."""
    out = evaluate_numpy(y, u, well, friction=friction)
    ok = out["ok"]
    if out["scalar"]:
        bad = not ok
    else:
        bad = not np.all(ok)
    if bad:
        raise InfeasibleState(
            f"ODE undefined ({friction}); refused a non-finite or non-positive intermediate"
        )
    return out["dy"]


def algebraics_numpy(y, u, well, friction: str = "haaland") -> np.ndarray:
    """[P_bh (bar), w_G_in, w_G_res, w_L_res] from the Haaland (or cubic) map."""
    out = evaluate_numpy(y, u, well, friction=friction)
    ok = out["ok"]
    bad = (not ok) if out["scalar"] else (not np.all(ok))
    if bad:
        raise InfeasibleState("algebraic map undefined at this state")
    if out["scalar"]:
        return np.array(
            [out["P_bh_bar"], out["w_G_in"], out["w_G_res"], out["w_L_res"]],
            dtype=np.float64,
        )
    return np.column_stack(
        [out["P_bh_bar"], out["w_G_in"], out["w_G_res"], out["w_L_res"]]
    )


def is_feasible(y, u, well, friction: str = "haaland") -> bool:
    out = evaluate_numpy(y, u, well, friction=friction)
    if out["scalar"]:
        return bool(out["ok"])
    return bool(np.all(out["ok"]))


def feasible_mask(y, u, well, friction: str = "haaland") -> np.ndarray:
    out = evaluate_numpy(y, u, well, friction=friction)
    if out["scalar"]:
        return np.array([bool(out["ok"])])
    return np.asarray(out["ok"], dtype=bool)


def _sqrt_train(rad, train_sqrt: bool):
    """Plant: sqrt(max(rad, 0)). Training: sqrt(max(rad, 1e-3))."""
    if train_sqrt:
        return torch.sqrt(torch.clamp(rad, min=SQRT_SAFE_FLOOR))
    return torch.sqrt(torch.clamp(rad, min=0.0))


def _torch_batch(y, u):
    # Numpy inputs stay float64 so the torch twin matches ode_numpy.
    # A tensor that already requires grad keeps its dtype and device.
    if not torch.is_tensor(y):
        y = torch.tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
    if not torch.is_tensor(u):
        u = torch.tensor(np.asarray(u, dtype=np.float64), dtype=y.dtype, device=y.device)
    else:
        u = u.to(dtype=y.dtype, device=y.device)
    scalar = y.ndim == 1
    if y.ndim == 1:
        y = y.reshape(1, 3)
    if u.ndim == 1:
        u = u.reshape(1, 2)
    if u.shape[0] == 1 and y.shape[0] != 1:
        u = u.expand(y.shape[0], -1)
    if y.shape[0] == 1 and u.shape[0] != 1:
        y = y.expand(u.shape[0], -1)
    return y, u, scalar


def _reject(cond: torch.Tensor, message: str) -> None:
    # Data-dependent check. It is not part of the returned graph.
    if bool((~cond).any()):
        raise InfeasibleState(message)


def ode_torch(y, u, well, friction: str = "cubic") -> torch.Tensor:
    """Differentiable derivative. friction='cubic' is the physics-loss graph.

    Square-root clamping (1e-3) is applied only for friction='cubic'.
    Haaland mode matches ode_numpy and is what SLMPC linearizes.
    """
    if friction not in ("haaland", "cubic"):
        raise ValueError(f"friction must be 'haaland' or 'cubic', got {friction}")
    p = _as_well(well)
    y_b, u_b, scalar = _torch_batch(y, u)
    train_sqrt = friction == "cubic"
    m_an = y_b[:, 0]
    m_gt = y_b[:, 1]
    m_lt = y_b[:, 2]
    u1 = u_b[:, 0]
    u2 = u_b[:, 1]

    finite_in = torch.isfinite(y_b).all(dim=1) & torch.isfinite(u_b).all(dim=1)
    _reject(finite_in, "non-finite state or input")

    denom = p.V_tb + p.S_bh * p.L_bh - m_lt / p.rho_L
    _reject(torch.isfinite(denom) & (denom > 0) & (m_gt > 0) & (m_an > 0) & (m_lt > 0), "(20c) denominator or mass")
    rho_G_tb_t = m_gt / denom
    _reject(torch.isfinite(rho_G_tb_t) & (rho_G_tb_t > 0), "rho_G_tb_t")
    rho_mix = (m_gt + m_lt - p.rho_L * p.S_bh * p.L_bh) / p.V_tb
    _reject(torch.isfinite(rho_mix) & (rho_mix > 0), "rho_mix_bar")
    alpha_L_bar = (m_lt - p.rho_L * p.S_bh * p.L_bh) / (p.V_tb * p.rho_L)

    P_an_t = p.R * p.T_an * m_an / (p.M_G * p.V_an)
    P_an_b = P_an_t + m_an * p.g * p.L_an / p.V_an
    rho_G_an_b = P_an_b * p.M_G / (p.R * p.T_an)
    rho_G_in = p.P_gs * p.M_G / (p.R * p.T_an)
    _reject(torch.isfinite(rho_G_an_b) & (rho_G_an_b > 0) & (rho_G_in > 0), "annulus density")

    rad_in = rho_G_in * torch.clamp(p.P_gs - P_an_t, min=0.0)
    w_G_in = p.K_gs * u2 * _sqrt_train(rad_in, train_sqrt)

    area_tb = math_pi(y_b) * p.D_tb**2
    U_L = 4.0 * (1.0 - p.alpha_G_bh_m) * p.w_res_bar / (p.rho_L * area_tb)
    U_G = 4.0 * (w_G_in + p.alpha_G_bh_m * p.w_res_bar) / (rho_G_tb_t * area_tb)
    U_mix = U_L + U_G
    _reject(torch.isfinite(U_mix) & (U_mix > 0), "mixture velocity")

    Re_tb = rho_mix * U_mix * p.D_tb / p.mu
    if friction == "haaland":
        _reject(torch.isfinite(Re_tb) & (Re_tb > 0), "Re_tb")
        lam_tb = haaland_lambda_torch(Re_tb, p.eps, p.D_tb)
        arg = (p.eps / p.D_tb / 3.7) ** 1.11 + 6.9 / Re_tb
        inv = -1.8 * torch.log10(arg)
        _reject(torch.isfinite(lam_tb) & (arg > 0) & (inv > 0), "Haaland tubing friction")
    else:
        _reject(torch.isfinite(Re_tb), "Re_tb")
        lam_tb = cubic_lambda_torch(Re_tb, p.cubic)
        _reject(torch.isfinite(lam_tb) & (lam_tb > 0), "cubic tubing friction")

    F_tb = alpha_L_bar * lam_tb * rho_mix * U_mix**2 * p.L_tb / (2.0 * p.D_tb)
    P_tb_t = rho_G_tb_t * p.R * p.T_tb / p.M_G
    P_tb_b = P_tb_t + rho_mix * p.g * p.L_tb + F_tb
    P_bh = P_tb_b + p.F_bh + p.rho_L * p.g * p.L_bh
    _reject(torch.isfinite(P_bh) & (P_tb_b > 0), "tubing pressure")

    rad_inj = rho_G_an_b * torch.clamp(P_an_b - P_tb_b, min=0.0)
    w_G_inj = p.K_inj * _sqrt_train(rad_inj, train_sqrt)
    w_res = p.PI * torch.clamp(p.P_res - P_bh, min=0.0)
    w_L_res = (1.0 - p.alpha_G_bh_m) * w_res
    w_G_res = p.alpha_G_bh_m * w_res

    rho_G_tb_b = P_tb_b * p.M_G / (p.R * p.T_tb)
    _reject(torch.isfinite(rho_G_tb_b) & (rho_G_tb_b > 0), "rho_G_tb_b")
    denom_alpha = w_L_res * rho_G_tb_b + (w_G_inj + w_G_res) * p.rho_L
    _reject(torch.isfinite(denom_alpha) & (denom_alpha > 0), "(21c) denominator")
    alpha_L_b = (w_L_res * rho_G_tb_b) / denom_alpha
    alpha_L_t = 2.0 * alpha_L_bar - alpha_L_b
    rho_mix_t = alpha_L_t * p.rho_L + (1.0 - alpha_L_t) * rho_G_tb_t
    _reject(torch.isfinite(rho_mix_t) & (rho_mix_t > 0), "rho_mix_tb_t")
    alpha_G_m = ((1.0 - alpha_L_t) * rho_G_tb_t) / rho_mix_t
    _reject(torch.isfinite(alpha_G_m), "alpha_G_tb_t_m")

    rad_out = rho_mix_t * torch.clamp(P_tb_t - p.P0, min=0.0)
    w_out = p.K_pr * u1 * _sqrt_train(rad_out, train_sqrt)
    w_L_out = (1.0 - alpha_G_m) * w_out
    w_G_out = alpha_G_m * w_out

    dy = torch.stack(
        [
            w_G_in - w_G_inj,
            w_G_inj + w_G_res - w_G_out,
            w_L_res - w_L_out,
        ],
        dim=1,
    )
    _reject(torch.isfinite(dy).all(dim=1), "non-finite derivative")
    if scalar:
        return dy.reshape(3)
    return dy


def algebraics_torch(y, u, well, friction: str = "haaland") -> torch.Tensor:
    """Differentiable [P_bh bar, w_G_in, w_G_res, w_L_res].

    P_bh in bar is a log/plot/IAE unit. The pascal pressure is converted
    only at this output. Jacobians for SLMPC use friction='haaland'.
    """
    # Recompute through the same graph as ode_torch, keeping P_bh.
    # A second full pass is simpler than threading every intermediate out of
    # ode_torch, and both are the same equations. SLMPC calls this once per
    # control step, not inside the PINC training loop.
    p = _as_well(well)
    y_b, u_b, scalar = _torch_batch(y, u)
    # Delegate the feasibility checks to a joint evaluation.
    pack = _algebraic_pack(y_b, u_b, p, friction)
    if scalar:
        return pack.reshape(4)
    return pack


def _algebraic_pack(y_b, u_b, p: Well, friction: str) -> torch.Tensor:
    """Same sequence as ode_torch, returning the four algebraic outputs."""
    train_sqrt = friction == "cubic"
    m_an = y_b[:, 0]
    m_gt = y_b[:, 1]
    m_lt = y_b[:, 2]
    u2 = u_b[:, 1]
    denom = p.V_tb + p.S_bh * p.L_bh - m_lt / p.rho_L
    _reject(denom > 0, "(20c) denominator")
    rho_G_tb_t = m_gt / denom
    _reject(rho_G_tb_t > 0, "rho_G_tb_t")
    rho_mix = (m_gt + m_lt - p.rho_L * p.S_bh * p.L_bh) / p.V_tb
    _reject(rho_mix > 0, "rho_mix_bar")
    alpha_L_bar = (m_lt - p.rho_L * p.S_bh * p.L_bh) / (p.V_tb * p.rho_L)
    P_an_t = p.R * p.T_an * m_an / (p.M_G * p.V_an)
    P_an_b = P_an_t + m_an * p.g * p.L_an / p.V_an
    rho_G_an_b = P_an_b * p.M_G / (p.R * p.T_an)
    rho_G_in = p.P_gs * p.M_G / (p.R * p.T_an)
    rad_in = rho_G_in * torch.clamp(p.P_gs - P_an_t, min=0.0)
    w_G_in = p.K_gs * u2 * _sqrt_train(rad_in, train_sqrt)
    area_tb = y_b.new_tensor(np.pi) * p.D_tb**2
    U_L = 4.0 * (1.0 - p.alpha_G_bh_m) * p.w_res_bar / (p.rho_L * area_tb)
    U_G = 4.0 * (w_G_in + p.alpha_G_bh_m * p.w_res_bar) / (rho_G_tb_t * area_tb)
    U_mix = U_L + U_G
    Re_tb = rho_mix * U_mix * p.D_tb / p.mu
    if friction == "haaland":
        _reject((Re_tb > 0), "Re_tb")
        lam_tb = haaland_lambda_torch(Re_tb, p.eps, p.D_tb)
    else:
        lam_tb = cubic_lambda_torch(Re_tb, p.cubic)
    F_tb = alpha_L_bar * lam_tb * rho_mix * U_mix**2 * p.L_tb / (2.0 * p.D_tb)
    P_tb_t = rho_G_tb_t * p.R * p.T_tb / p.M_G
    P_tb_b = P_tb_t + rho_mix * p.g * p.L_tb + F_tb
    P_bh = P_tb_b + p.F_bh + p.rho_L * p.g * p.L_bh
    _reject(torch.isfinite(P_bh), "P_bh")
    w_res = p.PI * torch.clamp(p.P_res - P_bh, min=0.0)
    w_L_res = (1.0 - p.alpha_G_bh_m) * w_res
    w_G_res = p.alpha_G_bh_m * w_res
    return torch.stack([P_bh / 1.0e5, w_G_in, w_G_res, w_L_res], dim=1)


def math_pi(like: torch.Tensor) -> torch.Tensor:
    return like.new_tensor(np.pi)
