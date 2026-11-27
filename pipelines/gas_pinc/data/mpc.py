"""Nonlinear MPC and successive-linearization MPC in CasADi.

The PINC and the algebraic net are rebuilt from the saved weights so IPOPT
differentiates the tanh skip network exactly. P_bh is an MX expression
assigned from the algebraic net and substituted into the cost. It is not an
equality constraint: CasADi Opti cannot assign into an opti.variable slice,
and Appendix C warns that a constraint on P_bh only slows the NLP.

On the first solve the leading 2x2 block of R is zero, so the first move is
not tied to an unknown previous valve position. Later solves warm-start by
shifting the previous trajectory one step.
"""

from __future__ import annotations

import numpy as np
import torch

from data.forecast import pinc_step
from data.param import T_HORIZON
from data.sample import integrate_haaland
from data.well_ode import InfeasibleState, algebraics_numpy, algebraics_torch, ode_torch

try:
    import casadi as ca
except ImportError as exc:  # pragma: no cover
    ca = None
    _CASADI_ERROR = exc
else:
    _CASADI_ERROR = None


def _need_casadi():
    if ca is None:
        raise ImportError("casadi is required for MPC") from _CASADI_ERROR


def _as_dm_linear(layer, x):
    weight = layer.weight.detach().cpu().numpy().astype(np.float64)
    bias = layer.bias.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
    return ca.mtimes(ca.DM(weight), x) + ca.DM(bias)


def skip_mx(net, features):
    """Wang skip block, or a plain tanh MLP when net.skip is false."""
    if not net.skip:
        hidden = features
        for layer in net.hidden:
            hidden = ca.tanh(_as_dm_linear(layer, hidden))
        return _as_dm_linear(net.out, hidden)
    encoded_u = ca.tanh(_as_dm_linear(net.enc_u, features))
    encoded_v = ca.tanh(_as_dm_linear(net.enc_v, features))
    activated = features
    for k, layer in enumerate(net.hidden):
        gate = ca.tanh(_as_dm_linear(layer, features if k == 0 else activated))
        activated = (1.0 - gate) * encoded_u + gate * encoded_v
    return _as_dm_linear(net.out, activated)


def mlp_mx(net, features):
    hidden = features
    for layer in net.hidden:
        hidden = ca.tanh(_as_dm_linear(layer, hidden))
    return _as_dm_linear(net.out, hidden)


def casadi_skip_function(net):
    """Raw network, (6, 1) -> (3, 1), for the torch comparison test."""
    _need_casadi()
    features = ca.MX.sym("X", net.in_dim)
    output = skip_mx(net, features)
    return ca.Function("skip_pinc", [features], [output])


def pinc_transition(net, y_min, y_max):
    """Physical state and valve to the physical state T seconds later."""
    _need_casadi()
    state = ca.MX.sym("x", 3)
    control = ca.MX.sym("u", 2)
    lo = np.asarray(y_min, dtype=np.float64).reshape(3, 1)
    span = (np.asarray(y_max, dtype=np.float64) - np.asarray(y_min, dtype=np.float64)).reshape(3, 1)
    scaled = (state - ca.DM(lo)) / ca.DM(span)
    features = ca.vertcat(ca.DM(1.0), scaled, control)
    nxt = skip_mx(net, features) * ca.DM(span) + ca.DM(lo)
    return ca.Function("F", [state, control], [nxt])


def algebraic_transition(net, y_min, y_max, z_mean, z_std):
    """Physical state and valve to [P_bh bar, w_G_in, w_G_res, w_L_res]."""
    _need_casadi()
    state = ca.MX.sym("x", 3)
    control = ca.MX.sym("u", 2)
    lo = np.asarray(y_min, dtype=np.float64).reshape(3, 1)
    span = (np.asarray(y_max, dtype=np.float64) - np.asarray(y_min, dtype=np.float64)).reshape(3, 1)
    scaled = (state - ca.DM(lo)) / ca.DM(span)
    features = ca.vertcat(scaled, control)
    standardized = mlp_mx(net, features)
    physical = standardized * ca.DM(np.asarray(z_std, dtype=np.float64).reshape(-1, 1))
    physical = physical + ca.DM(np.asarray(z_mean, dtype=np.float64).reshape(-1, 1))
    return ca.Function("Fy", [state, control], [physical])


def q_weights(horizon: int, control_horizon: int) -> np.ndarray:
    weights = np.ones(int(horizon), dtype=np.float64)
    weights[int(control_horizon) :] = 100.0
    return weights


def reference_at(t_seconds: float) -> float:
    """30 min at 95 bar, 30 min at 97 bar, then 99 bar."""
    if t_seconds < 30.0 * 60.0:
        return 95.0
    if t_seconds < 60.0 * 60.0:
        return 97.0
    return 99.0


def horizon_reference(t0: float, horizon: int, step: float = T_HORIZON) -> np.ndarray:
    return np.array([reference_at(t0 + (i + 1) * step) for i in range(horizon)], dtype=np.float64)


class _Controller:
    def __init__(self, horizon, control_horizon):
        _need_casadi()
        self.N = int(horizon)
        self.Nu = int(control_horizon)
        if not (1 <= self.Nu <= self.N):
            raise ValueError("Nu must satisfy 1 <= Nu <= N")
        self.opti = ca.Opti()
        self.x = self.opti.variable(3, self.N + 1)
        self.u = self.opti.variable(2, self.N)
        self.du = self.opti.variable(2, self.Nu)
        self.x0 = self.opti.parameter(3, 1)
        self.u_last = self.opti.parameter(2, 1)
        self.ref = self.opti.parameter(1, self.N)
        # Per-step multiplier on R. The first entry is set to 0 on the first solve.
        self.r_scale = self.opti.parameter(self.Nu)
        self.q = q_weights(self.N, self.Nu)
        self.r_weight = 1.0e3
        self.guess = None
        self.opti.subject_to(self.x[:, 0] == self.x0)
        self.opti.subject_to(self.u[:, 0] == self.u_last + self.du[:, 0])
        for i in range(self.Nu - 1):
            self.opti.subject_to(self.u[:, i + 1] == self.u[:, i] + self.du[:, i + 1])
        for i in range(self.Nu, self.N):
            self.opti.subject_to(self.u[:, i] == self.u[:, self.Nu - 1])
        self.opti.subject_to(self.opti.bounded(0.0, self.u, 1.0))

    def _set_solver(self):
        self.opti.solver(
            "ipopt",
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 500,
                "ipopt.tol": 1e-6,
            },
        )

    def _apply_guess(self, x0, u_prev, first: bool):
        if self.guess is None or first:
            if first:
                u_guess = np.full((2, self.N), 0.5, dtype=np.float64)
            else:
                u_guess = np.repeat(np.clip(np.asarray(u_prev, dtype=np.float64).reshape(2, 1), 0.0, 1.0), self.N, axis=1)
            du_guess = np.zeros((2, self.Nu), dtype=np.float64)
            du_guess[:, 0] = u_guess[:, 0] - np.asarray(u_prev, dtype=np.float64).reshape(2)
            x_guess = self.initial_state(x0, u_guess)
        else:
            x_prev, u_prev_sol, du_prev = self.guess
            x_guess = np.array(x_prev, copy=True)
            u_guess = np.array(u_prev_sol, copy=True)
            du_guess = np.array(du_prev, copy=True)
            x_guess[:, :-1] = x_prev[:, 1:]
            x_guess[:, -1] = x_prev[:, -1]
            u_guess[:, :-1] = u_prev_sol[:, 1:]
            u_guess[:, -1] = u_prev_sol[:, -1]
            if self.Nu > 1:
                du_guess[:, :-1] = du_prev[:, 1:]
            du_guess[:, -1] = 0.0
            x_guess[:, 0] = np.asarray(x0, dtype=np.float64).reshape(3)
        self.opti.set_initial(self.x, x_guess)
        self.opti.set_initial(self.u, np.clip(u_guess, 0.0, 1.0))
        self.opti.set_initial(self.du, du_guess)

    def initial_state(self, x0, u_guess):
        return np.repeat(np.asarray(x0, dtype=np.float64).reshape(3, 1), self.N + 1, axis=1)

    def solve(self, x0, u_prev, ref, first: bool = False):
        x0 = np.asarray(x0, dtype=np.float64).reshape(3, 1)
        u_prev = np.asarray(u_prev, dtype=np.float64).reshape(2, 1)
        ref = np.asarray(ref, dtype=np.float64).reshape(1, self.N)
        scale = np.ones(self.Nu, dtype=np.float64)
        if first:
            scale[0] = 0.0
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.u_last, u_prev)
        self.opti.set_value(self.ref, ref)
        self.opti.set_value(self.r_scale, scale.reshape(self.Nu, 1))
        self._apply_guess(x0, u_prev, first)
        try:
            solution = self.opti.solve()
            value = solution.value
        except RuntimeError as exc:
            print(f"IPOPT returned without a certified solution: {exc}")
            value = self.opti.debug.value
        x_val = np.array(value(self.x), dtype=np.float64).reshape(3, self.N + 1)
        u_val = np.array(value(self.u), dtype=np.float64).reshape(2, self.N)
        du_val = np.array(value(self.du), dtype=np.float64).reshape(2, self.Nu)
        self.guess = (x_val, u_val, du_val)
        return u_val[:, 0].copy()


class PINCMPC(_Controller):
    def __init__(self, pinc, algebraic, y_min, y_max, z_mean, z_std, horizon, control_horizon):
        super().__init__(horizon, control_horizon)
        self.y_min = np.asarray(y_min, dtype=np.float64)
        self.y_max = np.asarray(y_max, dtype=np.float64)
        self.pinc = pinc
        transition = pinc_transition(pinc, self.y_min, self.y_max)
        algebraic_map = algebraic_transition(algebraic, self.y_min, self.y_max, z_mean, z_std)
        cost = 0
        for i in range(self.N):
            self.opti.subject_to(self.x[:, i + 1] == transition(self.x[:, i], self.u[:, i]))
            # Assignment, not a constraint. The expression enters the cost directly.
            pressure = algebraic_map(self.x[:, i + 1], self.u[:, i])[0]
            cost += float(self.q[i]) * (pressure - self.ref[0, i]) ** 2
        for j in range(self.Nu):
            cost += self.r_scale[j] * self.r_weight * ca.sumsqr(self.du[:, j])
        self.opti.minimize(cost)
        self._set_solver()

    def initial_state(self, x0, u_guess):
        rows = [np.asarray(x0, dtype=np.float64).reshape(3)]
        y = rows[0].copy()
        for i in range(self.N):
            nxt = pinc_step(self.pinc, y, u_guess[:, i], self.y_min, self.y_max)
            if not np.isfinite(nxt).all():
                return np.repeat(rows[0].reshape(3, 1), self.N + 1, axis=1)
            rows.append(nxt)
            y = nxt
        return np.column_stack(rows)


class SLMPC(_Controller):
    def __init__(self, horizon, control_horizon):
        super().__init__(horizon, control_horizon)
        self.A = self.opti.parameter(3, 3)
        self.B = self.opti.parameter(3, 2)
        self.delta = self.opti.parameter(3, 1)
        self.C = self.opti.parameter(1, 3)
        self.D = self.opti.parameter(1, 2)
        self.pbh0 = self.opti.parameter(1, 1)
        self.x_lin = self.opti.parameter(3, 1)
        self.u_lin = self.opti.parameter(2, 1)
        cost = 0
        for i in range(self.N):
            dx = self.x[:, i] - self.x_lin
            du = self.u[:, i] - self.u_lin
            nxt = self.x[:, i] + ca.mtimes(self.A, dx) + ca.mtimes(self.B, du) + self.delta
            self.opti.subject_to(self.x[:, i + 1] == nxt)
            dx_out = self.x[:, i + 1] - self.x_lin
            pressure = self.pbh0 + ca.mtimes(self.C, dx_out) + ca.mtimes(self.D, self.u[:, i] - self.u_lin)
            cost += float(self.q[i]) * (pressure - self.ref[0, i]) ** 2
        for j in range(self.Nu):
            cost += self.r_scale[j] * self.r_weight * ca.sumsqr(self.du[:, j])
        self.opti.minimize(cost)
        self._set_solver()

    def set_linear_model(self, matrices: dict):
        self.opti.set_value(self.A, np.asarray(matrices["A"], dtype=np.float64))
        self.opti.set_value(self.B, np.asarray(matrices["B"], dtype=np.float64))
        self.opti.set_value(self.delta, np.asarray(matrices["delta"], dtype=np.float64).reshape(3, 1))
        self.opti.set_value(self.C, np.asarray(matrices["C"], dtype=np.float64).reshape(1, 3))
        self.opti.set_value(self.D, np.asarray(matrices["D"], dtype=np.float64).reshape(1, 2))
        self.opti.set_value(self.pbh0, float(matrices["pbh"]))
        self.opti.set_value(self.x_lin, np.asarray(matrices["x"], dtype=np.float64).reshape(3, 1))
        self.opti.set_value(self.u_lin, np.asarray(matrices["u"], dtype=np.float64).reshape(2, 1))


def linearize(y, u, well, T: float = T_HORIZON) -> dict:
    """Euler linearization of the Haaland ODE and of P_bh, at one operating point.

    A = T df/dx, B = T df/du, delta = T f. C and D are the pressure Jacobians.
    """
    y_t = torch.tensor(np.asarray(y, dtype=np.float64).reshape(3), dtype=torch.float64, requires_grad=True)
    u_t = torch.tensor(np.asarray(u, dtype=np.float64).reshape(2), dtype=torch.float64, requires_grad=True)
    f = ode_torch(y_t, u_t, well, friction="haaland")
    pressure = algebraics_torch(y_t, u_t, well, friction="haaland")[0]
    rows_x = []
    rows_u = []
    for i in range(3):
        grad_x, grad_u = torch.autograd.grad(f[i], (y_t, u_t), retain_graph=True)
        rows_x.append(grad_x)
        rows_u.append(grad_u)
    df_dx = torch.stack(rows_x, dim=0)
    df_du = torch.stack(rows_u, dim=0)
    c_vec, d_vec = torch.autograd.grad(pressure, (y_t, u_t), retain_graph=False)
    return {
        "A": (T * df_dx).detach().cpu().numpy(),
        "B": (T * df_du).detach().cpu().numpy(),
        "delta": (T * f).detach().cpu().numpy(),
        "C": c_vec.detach().cpu().numpy(),
        "D": d_vec.detach().cpu().numpy(),
        "pbh": float(pressure.detach()),
        "x": np.asarray(y, dtype=np.float64).reshape(3),
        "u": np.asarray(u, dtype=np.float64).reshape(2),
    }


def closed_loop(
    controller,
    *,
    kind: str,
    well,
    x0,
    u_prev,
    minutes: int,
    noise: float,
    seed: int,
    step: float = T_HORIZON,
):
    """Plant is RK45 for `step` seconds. `kind` is 'pinc' or 'slmpc'."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64).reshape(3).copy()
    u_hold = np.asarray(u_prev, dtype=np.float64).reshape(2).copy()
    n_steps = int(minutes)
    t = [0.0]
    y = [x.copy()]
    pbh = [float(algebraics_numpy(x, u_hold, well, friction="haaland")[0])]
    ref = [reference_at(0.0)]
    valves = [u_hold.copy()]
    for k in range(n_steps):
        x_meas = x.copy()
        if noise > 0.0:
            x_meas = x + rng.normal(0.0, noise * np.abs(x))
        ref_h = horizon_reference(k * step, controller.N, step)
        if kind == "slmpc":
            try:
                matrices = linearize(x_meas, u_hold, well, step)
            except InfeasibleState:
                matrices = linearize(x, u_hold, well, step)
            controller.set_linear_model(matrices)
        u_apply = controller.solve(x_meas, u_hold, ref_h, first=(k == 0))
        u_apply = np.clip(u_apply, 0.0, 1.0)
        try:
            out = integrate_haaland(x, u_apply, well, step, steady_tol=None)
        except InfeasibleState as exc:
            print(f"{kind} plant rejected the step: {exc}")
            break
        if not out["reached"]:
            print(
                f"{kind} plant left the explicit model at t={out['t_end']:.1f}s "
                f"under u={u_apply.tolist()}"
            )
            break
        x = out["y_end"].copy()
        u_hold = u_apply.copy()
        t.append((k + 1) * step)
        y.append(x.copy())
        pbh.append(float(algebraics_numpy(x, u_hold, well, friction="haaland")[0]))
        ref.append(reference_at((k + 1) * step))
        valves.append(u_hold.copy())
    pbh_arr = np.asarray(pbh, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    # Tracking error on the samples after each applied move.
    iae = float(np.mean(np.abs(pbh_arr[1:] - ref_arr[1:]))) if n_steps else float("nan")
    return {
        "t": np.asarray(t, dtype=np.float64),
        "y": np.vstack(y),
        "u": np.vstack(valves),
        "pbh": pbh_arr,
        "ref": ref_arr,
        "iae": iae,
    }
