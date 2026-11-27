"""Self-loop PINC rollout against the Haaland Runge-Kutta plant.

Each control step queries the network at t = T and re-injects the predicted
state. The reference integrates the same valve sequence with RK45. IAE uses
the C post-step knots. The shared initial state is stored in the trajectory
and is not part of the average.
"""

from __future__ import annotations

import numpy as np
import torch

from data.algebraic_nn import algebraic_predict
from data.param import T_HORIZON
from data.sample import integrate_haaland
from data.utils import scale_state, unscale_state
from data.well_ode import InfeasibleState, algebraics_numpy, is_feasible


def pinc_step(net, y, u, y_min, y_max) -> np.ndarray:
    """One control step: network at t = T, physical masses in, physical masses out."""
    y = np.asarray(y, dtype=np.float64).reshape(3)
    u = np.asarray(u, dtype=np.float64).reshape(2)
    y_s = scale_state(y, y_min, y_max)
    features = np.concatenate([np.array([1.0]), y_s, u])
    dtype = next(net.parameters()).dtype
    device = next(net.parameters()).device
    xt = torch.as_tensor(features, dtype=dtype, device=device).reshape(1, -1)
    was = net.training
    net.eval()
    with torch.no_grad():
        pred = net(xt).detach().cpu().numpy().reshape(3)
    if was:
        net.train()
    return unscale_state(pred, y_min, y_max)


def self_loop(net, y0, u_seq, y_min, y_max) -> np.ndarray:
    """Physical trajectory of shape (steps + 1, 3), including the initial state."""
    u_seq = np.asarray(u_seq, dtype=np.float64).reshape(-1, 2)
    y = np.asarray(y0, dtype=np.float64).reshape(3).copy()
    rows = [y.copy()]
    for u in u_seq:
        y = pinc_step(net, y, u, y_min, y_max)
        rows.append(y.copy())
    return np.vstack(rows)


def rk_rollout(y0, u_seq, well, T: float = T_HORIZON) -> np.ndarray:
    u_seq = np.asarray(u_seq, dtype=np.float64).reshape(-1, 2)
    y = np.asarray(y0, dtype=np.float64).reshape(3).copy()
    rows = [y.copy()]
    for u in u_seq:
        out = integrate_haaland(y, u, well, T, steady_tol=None)
        if not out["reached"]:
            raise InfeasibleState(
                f"Haaland plant left the explicit model at t={out['t_end']:.2f}s "
                f"of a {T:.0f}s step"
            )
        y = out["y_end"].copy()
        if not is_feasible(y, u, well, friction="haaland"):
            raise InfeasibleState("RK step ended outside the explicit model")
        rows.append(y.copy())
    return np.vstack(rows)


def iae_states(y_pinc, y_rk, y_min, y_max) -> float:
    """(1/C) sum_k (1/3) ||y_s_pinc - y_s_rk||_2 on the post-step knots."""
    sp = scale_state(y_pinc, y_min, y_max)
    sr = scale_state(y_rk, y_min, y_max)
    err = np.linalg.norm(sp - sr, axis=1) / 3.0
    return float(np.mean(err))


def iae_pbh(pbh_pinc, pbh_rk) -> float:
    return float(np.mean(np.abs(np.asarray(pbh_pinc) - np.asarray(pbh_rk))))


def rollout_vs_rk(net, algebraic, alg_ckpt, y0, u_seq, y_min, y_max, well):
    y_pinc = self_loop(net, y0, u_seq, y_min, y_max)
    y_rk = rk_rollout(y0, u_seq, well, T_HORIZON)
    u_seq = np.asarray(u_seq, dtype=np.float64).reshape(-1, 2)
    # Knots after each step, paired with the valve that was held on that step.
    pbh_pinc = []
    pbh_rk = []
    for k, u in enumerate(u_seq):
        pbh_rk.append(float(algebraics_numpy(y_rk[k + 1], u, well, friction="haaland")[0]))
        if algebraic is None:
            pbh_pinc.append(float("nan"))
        else:
            pred = algebraic_predict(
                algebraic,
                y_pinc[k + 1],
                u,
                y_min,
                y_max,
                alg_ckpt["z_mean"],
                alg_ckpt["z_std"],
            )
            pbh_pinc.append(float(pred[0]))
    pbh_pinc = np.asarray(pbh_pinc, dtype=np.float64)
    pbh_rk = np.asarray(pbh_rk, dtype=np.float64)
    # Initial pressure, for the plot only. The IAE starts at the first step.
    pbh_pinc_plot = np.concatenate(
        [
            [
                float(algebraics_numpy(y_rk[0], u_seq[0], well, friction="haaland")[0])
                if algebraic is None
                else float(
                    algebraic_predict(
                        algebraic, y_pinc[0], u_seq[0], y_min, y_max, alg_ckpt["z_mean"], alg_ckpt["z_std"]
                    )[0]
                )
            ],
            pbh_pinc,
        ]
    )
    pbh_rk_plot = np.concatenate(
        [[float(algebraics_numpy(y_rk[0], u_seq[0], well, friction="haaland")[0])], pbh_rk]
    )
    metrics = {
        "IAE_states": iae_states(y_pinc[1:], y_rk[1:], y_min, y_max),
        "IAE_Pbh": iae_pbh(pbh_pinc, pbh_rk) if algebraic is not None else float("nan"),
        "C": int(u_seq.shape[0]),
    }
    return {
        "y_pinc": y_pinc,
        "y_rk": y_rk,
        "pbh_pinc": pbh_pinc_plot,
        "pbh_rk": pbh_rk_plot,
        "u": u_seq,
        "metrics": metrics,
    }
