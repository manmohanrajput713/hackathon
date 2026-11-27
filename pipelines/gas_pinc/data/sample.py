"""Feasible-domain discovery and collocation sets for one gas-lift well.

The sampling cube is the axis-aligned box of states visited by Haaland
RK45 trajectories, expanded by 5% on each side. Points drawn in that cube
are kept only when the explicit ODE is finite. The Runge-Kutta plant stops
at the boundary where the explicit model becomes undefined instead of
returning a zero derivative.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from data.param import (
    PROMPT_GUESS_WELL1,
    T_HORIZON,
    get_well,
    nominal_input,
    nominal_state,
)
from data.well_ode import InfeasibleState, algebraics_numpy, feasible_mask, is_feasible, ode_numpy

U_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)
PBH_GATE = (80.0, 120.0)
STEADY_TOL = 1e-3


def domain_path(root, well_id: int) -> Path:
    return Path(root) / "dataset" / f"well{int(well_id)}_domain.npz"


def load_domain(path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _attempt(rhs, y, t0, t1, events, max_step):
    try:
        return solve_ivp(
            rhs,
            (t0, t1),
            np.asarray(y, dtype=np.float64),
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            max_step=max_step,
            dense_output=True,
            events=events,
        )
    except InfeasibleState:
        return None


def integrate_haaland(
    y0,
    u,
    well,
    t_final: float,
    *,
    steady_tol: float | None = None,
    max_step: float = 5.0,
):
    """RK45 of the Haaland plant.

    Returns a dict with the sampled trajectory, the last feasible state, and
    whether the requested final time was reached. A steady-state event, when
    requested, stops the run once ||dy|| drops through `steady_tol`.
    """
    well = well if hasattr(well, "well_id") else get_well(well)
    y0 = np.asarray(y0, dtype=np.float64).reshape(3).copy()
    u = np.asarray(u, dtype=np.float64).reshape(2).copy()
    if not is_feasible(y0, u, well, friction="haaland"):
        raise InfeasibleState("integrate_haaland: initial state is outside the model")

    def rhs(_t, y):
        return ode_numpy(y, u, well, friction="haaland")

    events = None
    if steady_tol is not None:
        def steady(_t, y):
            return np.linalg.norm(ode_numpy(y, u, well, friction="haaland")) - float(steady_tol)

        steady.terminal = True
        steady.direction = -1
        events = steady

    t_final = float(t_final)
    samples_t = [0.0]
    samples_y = [y0.copy()]
    t = 0.0
    y = y0.copy()
    settled = False
    if steady_tol is not None and np.linalg.norm(rhs(0.0, y)) <= steady_tol:
        settled = True
        return _pack(samples_t, samples_y, t, y, settled, t_final)

    chunk = min(600.0, t_final)
    while t < t_final - 1e-8 and not settled:
        t1 = min(t_final, t + chunk)
        sol = _attempt(rhs, y, t, t1, events, max_step)
        if sol is None or sol.t[-1] <= t + 1e-10:
            if chunk <= 0.25:
                break
            chunk *= 0.5
            continue
        n_keep = max(2, int(np.ceil((sol.t[-1] - sol.t[0]) / 30.0)) + 1)
        grid = np.linspace(sol.t[0], sol.t[-1], n_keep)
        values = sol.sol(grid).T
        for ti, yi in zip(grid, values):
            if ti <= samples_t[-1] + 1e-9:
                continue
            samples_t.append(float(ti))
            samples_y.append(np.asarray(yi, dtype=np.float64))
        t = float(sol.t[-1])
        y = np.asarray(sol.y[:, -1], dtype=np.float64).copy()
        fired = (
            steady_tol is not None
            and sol.t_events is not None
            and len(sol.t_events) > 0
            and len(sol.t_events[0]) > 0
        )
        if fired:
            settled = True
            break
        if not sol.success and t < t1 - 1e-6:
            break
        chunk = min(600.0, t_final - t) if t < t_final else chunk
    return _pack(samples_t, samples_y, t, y, settled, t_final)


def _pack(samples_t, samples_y, t_end, y_end, settled, t_final):
    return {
        "t": np.asarray(samples_t, dtype=np.float64),
        "y": np.vstack(samples_y),
        "t_end": float(t_end),
        "y_end": np.asarray(y_end, dtype=np.float64),
        "settled": bool(settled),
        "reached": float(t_end) >= float(t_final) - 1e-2,
    }


def _initial_conditions(well) -> list[np.ndarray]:
    base = nominal_state(well.well_id)
    raw = [base]
    for index in range(3):
        for factor in (0.92, 1.08):
            trial = base.copy()
            trial[index] *= factor
            raw.append(trial)
    if well.well_id == 1:
        raw.append(PROMPT_GUESS_WELL1.copy())
    kept = []
    seen = set()
    u = nominal_input()
    for y in raw:
        y = np.asarray(y, dtype=np.float64)
        key = tuple(np.round(y, 5))
        if key in seen:
            continue
        if not is_feasible(y, u, well, friction="haaland"):
            continue
        # Liquid mass has to stay below the geometric cap. The ODE already
        # rejects a non-positive gas volume; this is the explicit bound.
        if y[2] >= well.liquid_cap:
            continue
        seen.add(key)
        kept.append(y)
    if not kept:
        raise RuntimeError(f"well {well.well_id}: no feasible initial state")
    return kept


def _bounds_from_cloud(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = cloud.min(axis=0)
    hi = cloud.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    pad = 0.05 * span
    return lo - pad, hi + pad


def _fill_pool(y_min, y_max, well, rng, pool_size: int, seed_y, seed_u):
    y_rows = [np.asarray(seed_y, dtype=np.float64)] if len(seed_y) else []
    u_rows = [np.asarray(seed_u, dtype=np.float64)] if len(seed_u) else []
    if y_rows:
        y_acc = [np.vstack(y_rows)]
        u_acc = [np.vstack(u_rows)]
        have = y_acc[0].shape[0]
    else:
        y_acc = []
        u_acc = []
        have = 0
    span = y_max - y_min
    chunk = 8192
    guard = 0
    while have < pool_size and guard < 40:
        guard += 1
        draw = rng.random((chunk, 3)) * span + y_min
        u = rng.random((chunk, 2))
        # (0, 1], not including a fully shut valve.
        u = np.clip(u, 1e-3, 1.0)
        ok = feasible_mask(draw, u, well, friction="haaland")
        if not np.any(ok):
            continue
        y_acc.append(draw[ok])
        u_acc.append(u[ok])
        have += int(np.sum(ok))
    if have == 0:
        raise RuntimeError("feasible-pool sampling kept no points")
    y_pool = np.vstack(y_acc)[:pool_size]
    u_pool = np.vstack(u_acc)[:pool_size]
    return y_pool, u_pool


def discover_domain(well_id: int, pool_size: int = 20000, seed: int = 0, hours: float = 2.0) -> dict:
    well = get_well(well_id)
    rng = np.random.default_rng(seed)
    # Primary state on the full valve grid, plus up to three nearby states.
    ics = _initial_conditions(well)[:4]
    primary = ics[0]
    print(
        f"well {well_id}: {len(ics)} feasible initial states, "
        f"primary={primary.tolist()}"
    )
    if well_id == 1 and not is_feasible(PROMPT_GUESS_WELL1, nominal_input(), well):
        print(
            "prompt guess [2000, 300, 8000] is outside the explicit model "
            "(tubing-top mixture density is not positive). The sweep starts "
            f"from {primary.tolist()}."
        )
    cloud_y = []
    cloud_u = []
    settled_y = []
    settled_u = []
    settled_pbh = []
    t_final = float(hours) * 3600.0
    jobs = [(primary, u1, u2, True) for u1 in U_GRID for u2 in U_GRID]
    for extra in ics[1:]:
        for u1 in U_GRID:
            for u2 in U_GRID:
                jobs.append((extra, u1, u2, False))
    print(f"integrating {len(jobs)} Haaland trajectories, horizon {t_final:.0f} s")
    for index, (y0, u1, u2, report) in enumerate(jobs, start=1):
        u = np.array([u1, u2], dtype=np.float64)
        try:
            out = integrate_haaland(y0, u, well, t_final, steady_tol=STEADY_TOL)
        except InfeasibleState:
            if report:
                print(f"  u=({u1:.1f},{u2:.1f}) initial state rejected")
            continue
        cloud_y.append(out["y"])
        cloud_u.append(np.repeat(u.reshape(1, 2), out["y"].shape[0], axis=0))
        pbh = None
        if is_feasible(out["y_end"], u, well):
            pbh = float(algebraics_numpy(out["y_end"], u, well, friction="haaland")[0])
        status = "settled" if out["settled"] else ("full" if out["reached"] else "stopped")
        if report:
            pbh_txt = "nan" if pbh is None else f"{pbh:.2f}"
            print(
                f"  [{index:03d}] u=({u1:.1f},{u2:.1f}) {status} "
                f"t={out['t_end']:.0f}s P_bh={pbh_txt} bar"
            )
        if pbh is not None and (out["settled"] or out["reached"]):
            settled_y.append(out["y_end"])
            settled_u.append(u)
            settled_pbh.append(pbh)
    if not cloud_y:
        raise RuntimeError("every trajectory left the model immediately")
    cloud = np.vstack(cloud_y)
    cloud_inputs = np.vstack(cloud_u)
    y_min, y_max = _bounds_from_cloud(cloud)
    # Trajectory samples are already feasible at the input that produced them.
    take = min(len(cloud), max(pool_size // 5, 1))
    pick = rng.choice(len(cloud), size=take, replace=False) if len(cloud) > take else np.arange(len(cloud))
    y_pool, u_pool = _fill_pool(
        y_min, y_max, well, rng, pool_size, cloud[pick], cloud_inputs[pick]
    )
    settled_pbh_arr = np.asarray(settled_pbh, dtype=np.float64) if settled_pbh else np.zeros(0)
    print(
        f"visited {len(cloud)} states, pool {len(y_pool)}, "
        f"y_min={y_min.tolist()} y_max={y_max.tolist()}"
    )
    if settled_pbh_arr.size:
        print(
            "settled/full-horizon P_bh (bar): "
            + ", ".join(f"{value:.2f}" for value in settled_pbh_arr)
        )
    else:
        print("no trajectory reached steady state or the full horizon")
    return {
        "y_min": y_min,
        "y_max": y_max,
        "y_pool": y_pool,
        "u_pool": u_pool,
        "settled_y": np.vstack(settled_y) if settled_y else np.zeros((0, 3)),
        "settled_u": np.vstack(settled_u) if settled_u else np.zeros((0, 2)),
        "settled_pbh_bar": settled_pbh_arr,
        "well": np.array([well_id]),
        "T": np.array([T_HORIZON]),
        "nominal_y": primary,
    }


def gate_ok(settled_pbh_bar) -> bool:
    values = np.asarray(settled_pbh_bar, dtype=np.float64)
    if values.size == 0:
        return False
    lo, hi = PBH_GATE
    return bool(np.any((values >= lo) & (values <= hi)))


def save_domain(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"wrote {path}")


def draw_pairs(domain: dict, n: int, rng: np.random.Generator):
    pool_y = np.asarray(domain["y_pool"], dtype=np.float64)
    pool_u = np.asarray(domain["u_pool"], dtype=np.float64)
    index = rng.integers(0, pool_y.shape[0], size=int(n))
    return pool_y[index], pool_u[index]


def make_training_batches(domain: dict, nt: int, nf: int, n_val: int, well, seed: int) -> dict:
    """Fixed initial-condition rows, collocation rows, and RK validation targets."""
    rng = np.random.default_rng(seed)
    y_ic, u_ic = draw_pairs(domain, nt, rng)
    y_f, u_f = draw_pairs(domain, nf, rng)
    t_f = rng.random(int(nf))
    y_val = []
    u_val = []
    y_rk = []
    tries = 0
    limit = max(20, int(n_val) * 30)
    while len(y_val) < int(n_val) and tries < limit:
        tries += 1
        y0, u0 = draw_pairs(domain, 1, rng)
        y0 = y0[0]
        u0 = u0[0]
        try:
            out = integrate_haaland(y0, u0, well, T_HORIZON, steady_tol=None)
        except InfeasibleState:
            continue
        if not out["reached"]:
            continue
        if not is_feasible(out["y_end"], u0, well, friction="haaland"):
            continue
        y_val.append(y0)
        u_val.append(u0)
        y_rk.append(out["y_end"])
    if len(y_val) < int(n_val):
        print(f"validation set has {len(y_val)} of the requested {n_val} trajectories")
    if not y_val:
        raise RuntimeError("could not integrate any validation trajectory for 60 s")
    return {
        "y_ic": np.asarray(y_ic, dtype=np.float64),
        "u_ic": np.asarray(u_ic, dtype=np.float64),
        "y_f": np.asarray(y_f, dtype=np.float64),
        "u_f": np.asarray(u_f, dtype=np.float64),
        "t_f": np.asarray(t_f, dtype=np.float64),
        "y_val": np.vstack(y_val),
        "u_val": np.vstack(u_val),
        "y_rk": np.vstack(y_rk),
    }


def save_batches(path, batches: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **batches)


def load_batches(path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
