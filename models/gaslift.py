"""Gas-lift models built on gas_pinc (Jahanshahi plant, Kittelsen et al. 2024).

vfm():   virtual flow meter. Measured pressures fix the plant state; the gas_pinc algebraic network (or the same
         gas_pinc equations for wells that differ from its training well) turns that state into liquid rate, lift-gas
         rate and bottom-hole pressure. Samples with missing pressures are carried forward one 60 s control step at a
         time by the gas_pinc PINC, or by the gas_pinc plant while the PINC checkpoint fails its accuracy check.
nodal(): nodal analysis. Inflow (IPR) and outflow (VLP, one curve per lift-gas rate) at the bottom hole, solved from
         the gas_pinc steady-state equations. Operating points are the stable IPR/VLP intersections.
"""
import dataclasses
import functools
import json
import math
import sys
import numpy as np
from common import ROOT, RUN

GAS_PINC = ROOT / "gas_pinc"
_GAS_PINC_MODULES = {}
_ACTIVE = [False]


def _with_gas_pinc(fn):
    """gas_pinc and pinn both name their package `data`. Swap gas_pinc's modules in only while a gas-lift function runs,
    so the ESP models (pinn) keep importing their own `data` package in the same process."""
    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        if _ACTIVE[0]:  # already inside a gas-lift call: gas_pinc's modules are in place
            return fn(*args, **kwargs)
        _ACTIVE[0] = True
        is_data = lambda n: n == "data" or n.startswith("data.")
        saved = {n: m for n, m in sys.modules.items() if is_data(n)}
        for n in saved:
            del sys.modules[n]
        sys.modules.update(_GAS_PINC_MODULES)
        sys.path.insert(0, str(GAS_PINC))
        try:
            return fn(*args, **kwargs)
        finally:
            sys.path.remove(str(GAS_PINC))
            for n in [n for n in sys.modules if is_data(n)]:
                _GAS_PINC_MODULES[n] = sys.modules.pop(n)
            sys.modules.update(saved)
            _ACTIVE[0] = False
    return wrap

R_GAS, T_STD, P_STD = 8.314, 288.15, 101325.0
PINC_PATHS = [GAS_PINC / "runs/well1/lbfgs/model.pt", GAS_PINC / "runs/well1/adam/model.pt"]
ALGEBRAIC_PATH = GAS_PINC / "runs/well1/algebraic/model.pt"
# A PINC step is accepted when its mean scaled one-step error against the plant is below this (paper well 1: ~0.01).
PINC_TOLERANCE = 0.02
# Without a downhole gauge a pressure-derived state is accepted only if its annulus + tubing gas imbalance is below this.
QUASI_STEADY_GAS_KG_S = 1.5
TRAINING_SETTINGS = dict(reservoir_bar=160, pi_m3d_bar=28.1, density=760, glr_mass=0, tubing_depth_m=2048,
                         tubing_id_m=0.134, gas_supply_bar=140)


def _gp():
    from data import param
    return param


@_with_gas_pinc
def build_well(settings):
    """A gas_pinc Well for this well's settings, starting from the published well nearest in depth."""
    gp = _gp()
    depth = settings["tubing_depth_m"]
    template = min(gp._WELLS, key=lambda k: abs(gp._WELLS[k]["L_tb"] - depth))
    base = gp._WELLS[template]
    rho = settings["density"]
    diameter = settings["tubing_id_m"]
    raw = dict(base, rho_L=rho, GOR=settings["glr_mass"], P_res_bar=settings["reservoir_bar"],
               PI=settings["pi_m3d_bar"] * rho / 86400 / 1e5, L_tb=depth, L_an=depth, D_tb=diameter,
               V_tb=math.pi * diameter ** 2 / 4 * depth, V_an=base["V_an"] * depth / base["L_an"])
    gp._WELLS[99], gp.FRICTION_CUBIC[99] = raw, gp.FRICTION_CUBIC[template]
    try:
        well = dataclasses.replace(gp._build(99), P_gs=settings["gas_supply_bar"] * 1e5)
    finally:
        gp._WELLS.pop(99, None)
        gp.FRICTION_CUBIC.pop(99, None)
    return well, template


def gas_std_density(well):
    return P_STD * well.M_G / (R_GAS * T_STD)


def _evaluate(y, u, well):
    from data.well_ode import evaluate_numpy
    return evaluate_numpy(np.atleast_2d(y), np.atleast_2d(u), well)


def _differs_from_training(settings):
    diffs = []
    for key, ref in TRAINING_SETTINGS.items():
        value = settings.get(key)
        if value is None:
            continue
        if (ref == 0 and abs(value) > 1e-9) or (ref != 0 and abs(value - ref) > 0.02 * abs(ref)):
            diffs.append(key)
    return diffs


# ---------- checkpoints ----------

def _load_algebraic():
    from data.utils import load_algebraic
    return load_algebraic(ALGEBRAIC_PATH) if ALGEBRAIC_PATH.exists() else (None, None)


def _pinc_check():
    """Load the gas_pinc PINC and score one 60 s step against the plant. Cached until the checkpoint changes."""
    from data.utils import load_pinc
    path = next((p for p in PINC_PATHS if p.exists()), None)
    if path is None:
        return None, None, dict(usable=False, reason="No gas_pinc PINC checkpoint was found.")
    net, ckpt = load_pinc(path)
    stamp = f"{path}:{path.stat().st_mtime_ns}"
    cache = RUN / "models" / "gas_pinc_check.json"
    try:
        saved = json.loads(cache.read_text())
        if saved.get("stamp") == stamp:
            return net, ckpt, saved["check"]
    except (OSError, ValueError, KeyError):
        pass
    from data.param import get_well, nominal_state
    from data.forecast import pinc_step
    from data.sample import integrate_haaland
    from data.utils import scale_state
    well = get_well(int(ckpt.get("well", 1)))
    errors = []
    rng = np.random.default_rng(0)
    y = nominal_state(well.well_id)
    for _ in range(12):
        u = rng.uniform(0.3, 0.9, 2)
        try:
            ref = integrate_haaland(y, u, well, 60.0, max_step=1.0)
        except Exception:
            continue
        if not ref["reached"]:
            continue
        pred = pinc_step(net, y, u, ckpt["y_min"], ckpt["y_max"])
        errors.append(float(np.mean(np.abs(scale_state(pred, ckpt["y_min"], ckpt["y_max"]) -
                                           scale_state(ref["y_end"], ckpt["y_min"], ckpt["y_max"])))))
        y = ref["y_end"]
    err = float(np.mean(errors)) if errors else math.inf
    check = dict(usable=bool(err <= PINC_TOLERANCE), error=err if math.isfinite(err) else None, steps=len(errors),
                 checkpoint=str(path.relative_to(ROOT)), iterations=ckpt.get("lbfgs_iterations_ran"),
                 optimizer=ckpt.get("optimizer"))
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(dict(stamp=stamp, check=check)))
    except OSError:
        pass
    return net, ckpt, check


# ---------- virtual flow meter ----------

def _state_from_pressures(well, u, casing_pa, thp_pa, bhp_pa=None, previous=None):
    """Plant masses [m_G_an, m_G_tb, m_L_tb] that reproduce the measured pressures.

    Casing pressure fixes the annulus gas mass (ideal gas). Tubing-head pressure fixes tubing gas mass for a given
    liquid mass. Liquid mass then matches the measured bottom-hole pressure, or, without a downhole gauge, the
    quasi-steady liquid balance (reservoir inflow equals choke outflow).
    """
    m_an = casing_pa * well.M_G * well.V_an / (R_GAS * well.T_an)
    lo = well.rho_L * well.S_bh * well.L_bh * 1.0005
    hi = well.rho_L * (well.V_tb + well.S_bh * well.L_bh) * 0.9995
    m_lt = np.linspace(lo, hi, 600)
    gas_volume = well.V_tb + well.S_bh * well.L_bh - m_lt / well.rho_L
    m_gt = thp_pa * well.M_G / (R_GAS * well.T_tb) * gas_volume
    y = np.column_stack([np.full_like(m_lt, m_an), m_gt, m_lt])
    e = _evaluate(y, u, well)
    ok = np.asarray(e["ok"], bool)
    target = (np.asarray(e["P_bh"]) - bhp_pa) if bhp_pa is not None else np.asarray(e["dy"])[:, 2]
    target = np.where(ok, target, np.nan)
    roots = []
    for i in range(len(m_lt) - 1):
        a, b = target[i], target[i + 1]
        if np.isfinite(a) and np.isfinite(b) and (a == 0 or a * b < 0):
            roots.append(m_lt[i] + (m_lt[i + 1] - m_lt[i]) * a / (a - b) if a != b else m_lt[i])
    if not roots:
        return None
    gas_mass = lambda m: thp_pa * well.M_G / (R_GAS * well.T_tb) * (well.V_tb + well.S_bh * well.L_bh - m / well.rho_L)
    candidates = [np.array([m_an, gas_mass(r), r]) for r in roots]
    if bhp_pa is None:
        # Right after a large valve move the liquid balance can have only an unphysical root. Its gas balances are far from
        # closed (several kg/s), so it is rejected and the caller carries the previous state forward instead.
        dy = np.asarray(_evaluate(np.vstack(candidates), u, well)["dy"])
        keep = np.abs(dy[:, 0]) + np.abs(dy[:, 1]) <= QUASI_STEADY_GAS_KG_S
        candidates = [c for c, k in zip(candidates, keep) if k]
        if not candidates:
            return None
    if bhp_pa is None and len(candidates) > 1:
        # The liquid balance alone has several roots. The quasi-steady state also closes the annulus and tubing gas balances.
        # Masses cannot jump within one sample, so a large change from the previous estimate is penalised too
        # (1000 kg of tubing liquid weighs like 1 kg/s of gas imbalance). This keeps transients after valve moves on track.
        dy = np.asarray(_evaluate(np.vstack(candidates), u, well)["dy"])
        score = np.abs(dy[:, 0]) + np.abs(dy[:, 1])
        if previous is not None:
            score = score + 1e-3 * np.abs(np.array([c[2] for c in candidates]) - previous[2])
        return candidates[int(np.nanargmin(score))]
    if previous is not None:
        return min(candidates, key=lambda c: abs(c[2] - previous[2]))
    return candidates[-1]


def _outputs(y, u, well, algebraic, use_network):
    """Liquid inflow (kg/s), lift gas through the gas-lift valve (kg/s) and bottom-hole pressure (bar)."""
    if use_network and algebraic is not None:
        from data.algebraic_nn import algebraic_predict
        net, ck = algebraic
        inside = np.all(y >= ck["y_min"]) and np.all(y <= ck["y_max"])
        if inside:
            z = algebraic_predict(net, y, u, ck["y_min"], ck["y_max"], ck["z_mean"], ck["z_std"])
            return float(z[3]), float(z[1]), float(z[0]), "network"
    e = _evaluate(y, u, well)
    if not bool(np.asarray(e["ok"])[0]):
        return None
    return float(np.asarray(e["w_L_res"])[0]), float(np.asarray(e["w_G_in"])[0]), float(np.asarray(e["P_bh_bar"])[0]), "equations"


@_with_gas_pinc
def vfm(rows, settings, step_s=60.0):
    if not rows:
        return {"skipped": "Add gas-lift high-frequency telemetry to estimate flow."}
    import pandas as pd
    from data.forecast import pinc_step
    from data.sample import integrate_haaland
    from data.param import get_well
    differs = _differs_from_training(settings)
    use_network = not differs
    # The networks were trained on gas_pinc well 1 exactly; other wells use the same equations with their own inputs.
    well, template = (get_well(1), 1) if use_network else build_well(settings)
    algebraic = _load_algebraic()
    pinc, pinc_ckpt, check = _pinc_check()
    use_pinc = use_network and check.get("usable")
    rho_std = gas_std_density(well)
    t = pd.DataFrame(rows)
    for col in ("casing_bar", "tubing_head_bar", "bottomhole_bar", "test_m3h"):
        if col not in t:
            t[col] = np.nan
    for col in ("production_choke_percent", "gas_lift_choke_percent", "casing_bar", "tubing_head_bar", "bottomhole_bar", "test_m3h"):
        t[col] = pd.to_numeric(t[col], errors="coerce").astype(float)
    t["ts"] = pd.to_datetime(t.timestamp, utc=True)
    t = t.sort_values("ts").reset_index(drop=True)
    u_all = np.column_stack([t.production_choke_percent.to_numpy(float), t.gas_lift_choke_percent.to_numpy(float)]) / 100
    gaps = t.ts.diff().dt.total_seconds().to_numpy()
    out, counts = [], dict(measured=0, propagated=0, none=0)
    sources = dict(network=0, equations=0)
    def propagate(y0, u0):
        """One control step from the previous state with the previous valve positions held."""
        if use_pinc:
            return pinc_step(pinc, y0, u0, pinc_ckpt["y_min"], pinc_ckpt["y_max"])
        try:
            step = integrate_haaland(y0, u0, well, step_s, max_step=1.0)
            return step["y_end"] if step["reached"] else None
        except Exception:
            return None

    y_prev, last_u = None, None
    for i in range(len(t)):
        u = np.clip(u_all[i], 1e-3, 1.0)
        r = t.iloc[i]
        y, mode = None, None
        has_surface = np.isfinite(r.casing_bar) and np.isfinite(r.tubing_head_bar)
        on_cadence = y_prev is not None and last_u is not None and abs(gaps[i] - step_s) < 0.01 * step_s
        if has_surface:
            bhp = r.bottomhole_bar * 1e5 if np.isfinite(r.bottomhole_bar) else None
            y = _state_from_pressures(well, u, r.casing_bar * 1e5, r.tubing_head_bar * 1e5, bhp, y_prev)
            mode = "measured" if y is not None else None
        if y is None and on_cadence:
            y = propagate(y_prev, last_u)
            mode = "propagated" if y is not None else None
        result = _outputs(y, u, well, algebraic, use_network) if y is not None else None
        if result is None:
            counts["none"] += 1
            y_prev = None
            out.append(dict(timestamp=r.timestamp, vfm_m3h=None, test_m3h=_num(r.test_m3h), lift_gas_sm3d=None,
                            bhp_bar=None, bhp_measured_bar=_num(r.bottomhole_bar), mode="none"))
            last_u = u
            continue
        w_l, w_g, bhp_bar, source = result
        counts[mode] += 1
        sources[source] += 1
        y_prev, last_u = y, u
        out.append(dict(timestamp=r.timestamp, vfm_m3h=round(w_l / well.rho_L * 3600, 4), test_m3h=_num(r.test_m3h),
                        lift_gas_sm3d=round(w_g * 86400 / rho_std, 1), bhp_bar=round(bhp_bar, 3),
                        bhp_measured_bar=_num(r.bottomhole_bar), mode=mode))
    estimated = [r for r in out if r["vfm_m3h"] is not None]
    channels = {c: round(float(t[c].notna().mean()), 4) for c in ["production_choke_percent", "gas_lift_choke_percent", "casing_bar",
                                                                   "tubing_head_bar", "bottomhole_bar", "test_m3h"]}
    spacing = pd.Series(gaps[1:])
    segments, start = [], 0
    valid = [r["vfm_m3h"] is not None for r in out]
    for i in range(1, len(out) + 1):
        if i == len(out) or valid[i] != valid[start]:
            segments.append(dict(start=out[start]["timestamp"], end=out[i - 1]["timestamp"], valid=valid[start], samples=i - start))
            start = i
    coverage = dict(samples=len(t), valid_windows=len(estimated), window=1, step_s=step_s, segments=segments[-500:], channels=channels,
                    measured=counts["measured"], propagated=counts["propagated"],
                    spacing_s=dict(median=float(spacing.median()) if len(spacing) else None,
                                   off_cadence=int((abs(spacing - step_s) >= 0.01 * step_s).sum())))
    model = dict(rates="gas_pinc algebraic network" if use_network and sources["equations"] == 0 else
                 "gas_pinc plant equations" if not use_network else "gas_pinc algebraic network + plant equations",
                 propagation=("gas_pinc PINC" if use_pinc else "gas_pinc plant (RK45)"), pinc_check=check,
                 template_well=template, network_rows=sources["network"], equation_rows=sources["equations"])
    if not estimated:
        return {"skipped": "No sample could be estimated. Each estimate needs casing and tubing-head pressure, or a "
                           f"previous estimate exactly {step_s:g} s earlier to carry forward.", "coverage": coverage}
    warnings = []
    if differs:
        warnings.append("This well's inputs (" + ", ".join(k.replace("_", " ") for k in differs) + ") differ from gas_pinc well 1, "
                        "which the gas_pinc networks were trained on. Rates come from the gas_pinc plant equations with this well's inputs.")
    elif sources["equations"]:
        warnings.append(f"{sources['equations']} samples fell outside the algebraic network's training box and used the plant equations.")
    if counts["propagated"]:
        if use_pinc:
            warnings.append(f"{counts['propagated']} samples without pressures were carried forward by the gas_pinc PINC.")
        else:
            why = (f"mean one-step error {check['error']:.3f} vs tolerance {PINC_TOLERANCE}" if check.get("error") is not None
                   else check.get("reason", "it failed its check"))
            warnings.append(f"{counts['propagated']} samples without pressures were carried forward by the gas_pinc plant (RK45). "
                            f"The PINC checkpoint is not used: {why}. Train it with gas_pinc's main_train_adam.py and "
                            "main_train_lbfgs.py to enable it.")
    if channels["bottomhole_bar"] < 0.5:
        warnings.append("No downhole gauge for most samples: the tubing liquid load is solved from a quasi-steady liquid balance, "
                        "which smooths casing-heading cycles.")
    if counts["none"] > 0.5 * len(t):
        warnings.append(f"Only {len(estimated)} of {len(t)} samples could be estimated.")
    pairs = [(r["vfm_m3h"], r["test_m3h"]) for r in estimated if r["test_m3h"] is not None]
    bhp_pairs = [(r["bhp_bar"], r["bhp_measured_bar"]) for r in estimated if r["bhp_measured_bar"] is not None and r["mode"] == "propagated"]
    metrics = {}
    if pairs:
        err = np.array([a - b for a, b in pairs])
        metrics.update(tests=len(pairs), mean_error_m3h=float(err.mean()), mean_abs_error_m3h=float(np.abs(err).mean()))
    if bhp_pairs:
        metrics["propagated_bhp_mae_bar"] = float(np.mean([abs(a - b) for a, b in bhp_pairs]))
    latest = estimated[-1]
    return {"rows": out[-20000:], "warnings": warnings, "metrics": metrics, "coverage": coverage, "model": model, "lift": "gas_lift",
            "summary": dict(label="Estimated liquid rate", value=latest["vfm_m3h"], unit="m³/h", at=latest["timestamp"],
                            quality="good" if not warnings else "check")}


def _num(v):
    return None if v is None or not np.isfinite(v) else float(v)


def surface_pressures(y, well):
    """Casing (annulus top) and tubing-head pressure in Pa for a plant state."""
    m_an, m_gt, m_lt = y
    casing = R_GAS * well.T_an * m_an / (well.M_G * well.V_an)
    thp = m_gt / (well.V_tb + well.S_bh * well.L_bh - m_lt / well.rho_L) * R_GAS * well.T_tb / well.M_G
    return casing, thp


@_with_gas_pinc
def simulate_telemetry(settings, start, samples=360, step_s=60.0, seed=0, test_every=60, noise_bar=0.05):
    """Synthetic 60 s telemetry from the gas_pinc plant: valve steps every 30 min, measured pressures with gauge noise,
    and a well test (true liquid inflow) every `test_every` samples. Used for the demo field and for tests."""
    import pandas as pd
    from data.param import get_well
    from data.sample import integrate_haaland
    from data.param import nominal_state
    from data.well_ode import is_feasible
    if _differs_from_training(settings):
        well, template = build_well(settings)
    else:
        well, template = get_well(1), 1
    rng = np.random.default_rng(seed)
    u = np.array([settings.get("choke_fraction", 0.5), 0.8])
    scale = well.L_tb / _gp()._WELLS[template]["L_tb"]
    starts = [np.array(nominal_state(template)) * [1, scale, scale * f] for f in (1.0, 0.8, 1.2, 0.6, 0.4)]
    y0 = next((s for s in starts if is_feasible(s, u, well)), None)
    if y0 is None:
        raise ValueError("No feasible starting state for this well's inputs.")
    y = integrate_haaland(y0, u, well, 7200, steady_tol=1e-3, max_step=15)["y_end"]
    stamps = pd.date_range(start, periods=samples, freq=f"{int(step_s)}s", tz="UTC")
    rows = []
    for i, ts in enumerate(stamps):
        if i and i % 30 == 0:
            u = np.array([np.clip(u[0] + rng.uniform(-0.1, 0.1), 0.3, 0.9), rng.uniform(0.6, 1.0)])
        e = _evaluate(y, u, well)
        casing, thp = surface_pressures(y, well)
        noise = rng.normal(0, noise_bar, 3)
        rows.append(dict(timestamp=ts.isoformat(), production_choke_percent=round(float(u[0]) * 100, 3),
                         gas_lift_choke_percent=round(float(u[1]) * 100, 3), casing_bar=round(casing / 1e5 + noise[0], 4),
                         tubing_head_bar=round(thp / 1e5 + noise[1], 4),
                         bottomhole_bar=round(float(np.asarray(e["P_bh_bar"])[0]) + noise[2], 4),
                         test_m3h=round(float(np.asarray(e["w_L_res"])[0]) / well.rho_L * 3600, 3) if i % test_every == 20 else None,
                         true_m3h=float(np.asarray(e["w_L_res"])[0]) / well.rho_L * 3600))
        step = integrate_haaland(y, u, well, step_s, max_step=5.0)
        if not step["reached"]:
            break
        y = step["y_end"]
    return rows


# ---------- nodal analysis ----------

def _fixed_rate_well(well, liquid_kg_s):
    """The same well with the reservoir replaced by a fixed-rate source, so steady states can be solved at a chosen rate."""
    p_res = 1e13
    total = liquid_kg_s / max(1.0 - well.alpha_G_bh_m, 1e-9)
    return dataclasses.replace(well, P_res=p_res, PI=total / p_res)


def _tubing_state(well, m_gt, m_lt, gas):
    """Evaluate the plant with the annulus placed so lift-gas supply and injection both equal `gas` (kg/s)."""
    rho_in = well.P_gs * well.M_G / (R_GAS * well.T_an)
    k_t = R_GAS * well.T_an / (well.M_G * well.V_an)       # P_an_t = k_t m_an
    k_b = k_t + well.g * well.L_an / well.V_an              # P_an_b = k_b m_an
    n = np.size(m_gt)
    # Pass 1: any annulus pressure, gas-lift valve opened so that supply equals `gas`. Tubing pressures only see the supply.
    p_an_t = np.full(n, 0.5 * well.P_gs)
    m_an = p_an_t / k_t
    u2 = gas / (well.K_gs * np.sqrt(rho_in * (well.P_gs - p_an_t)))
    y = np.column_stack([m_an, np.broadcast_to(m_gt, n), np.broadcast_to(m_lt, n)])
    u = np.column_stack([np.ones(n), u2])
    first = _evaluate(y, u, well)
    p_tb_b = np.asarray(first["P_bh"]) - well.F_bh - well.rho_L * well.g * well.L_bh
    # Pass 2: annulus mass where injection equals `gas`: gas² / K_inj² = rho_an_b (P_an_b - P_tb_b), rho_an_b = c P_an_b.
    c = well.M_G / (R_GAS * well.T_an)
    rhs = (gas / well.K_inj) ** 2 / c
    p_an_b = 0.5 * (p_tb_b + np.sqrt(np.maximum(p_tb_b ** 2 + 4 * rhs, 0)))
    m_an = p_an_b / k_b
    p_an_t = k_t * m_an
    headroom = well.P_gs - p_an_t
    u2 = np.where(headroom > 0, gas / (well.K_gs * np.sqrt(rho_in * np.maximum(headroom, 1e-9))), np.inf)
    y = np.column_stack([m_an, np.broadcast_to(m_gt, n), np.broadcast_to(m_lt, n)])
    return y, u2


GAS_GRID = np.geomspace(0.5, 4000.0, 120)


def _vlp_point(well, liquid_kg_s, gas, choke):
    """Steady bottom-hole pressure needed to produce `liquid_kg_s` with `gas` kg/s of lift gas, or None.

    Unknowns are the tubing gas and liquid masses (the annulus follows from the lift-gas rate). For each candidate
    liquid mass the tubing gas mass that balances gas in and out is found by vectorised bisection; the liquid mass
    that balances liquid in and out is then bracketed on a coarse grid and refined on a fine one.
    """
    src = _fixed_rate_well(well, liquid_kg_s)

    def evaluate(m_gt, m_lt):
        y, u2 = _tubing_state(src, m_gt, m_lt, gas)
        u = np.column_stack([np.full(len(y), choke), np.where(np.isfinite(u2), u2, 1.0)])
        e = _evaluate(y, u, src)
        ok = np.asarray(e["ok"], bool) & np.isfinite(u2)
        return e, ok, u2

    def gas_roots(m_lt):
        """Largest tubing gas mass with zero gas balance, per liquid mass (nan where none)."""
        n, k = len(m_lt), len(GAS_GRID)
        e, ok, _ = evaluate(np.tile(GAS_GRID, n), np.repeat(m_lt, k))
        r = np.where(ok, np.asarray(e["dy"])[:, 1], np.nan).reshape(n, k)
        change = np.isfinite(r[:, :-1]) & np.isfinite(r[:, 1:]) & (r[:, :-1] * r[:, 1:] <= 0)
        has = change.any(axis=1)
        last = k - 2 - np.argmax(change[:, ::-1], axis=1)
        lo, hi = GAS_GRID[last], GAS_GRID[last + 1]
        f_lo = r[np.arange(n), last]
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            em, okm, _ = evaluate(mid, m_lt)
            fm = np.where(okm, np.asarray(em["dy"])[:, 1], np.nan)
            left = np.isfinite(fm) & (np.sign(fm) == np.sign(f_lo))
            lo, f_lo = np.where(left, mid, lo), np.where(left, fm, f_lo)
            hi = np.where(left, hi, mid)
        return np.where(has, 0.5 * (lo + hi), np.nan)

    def liquid_balance(m_lt):
        m_gt = gas_roots(m_lt)
        good = np.isfinite(m_gt)
        e, ok, u2 = evaluate(np.where(good, m_gt, 1.0), m_lt)
        ok &= good
        return np.where(ok, np.asarray(e["dy"])[:, 2], np.nan), m_gt, np.asarray(e["P_bh_bar"]), u2

    lo_l = well.rho_L * well.S_bh * well.L_bh * 1.0005
    hi_l = well.rho_L * (well.V_tb + well.S_bh * well.L_bh) * 0.9995
    grid = np.linspace(lo_l, hi_l, 36)
    for _level in range(3):
        r, m_gt, bhp, u2 = liquid_balance(grid)
        idx = [i for i in range(len(grid) - 1) if np.isfinite(r[i]) and np.isfinite(r[i + 1]) and r[i] * r[i + 1] <= 0]
        if not idx:
            return None
        i = idx[-1]
        if _level < 2:
            grid = np.linspace(grid[i], grid[i + 1], 12)
    w = r[i] / (r[i] - r[i + 1]) if r[i] != r[i + 1] else 0.0
    m_lt = grid[i] + w * (grid[i + 1] - grid[i])
    gas_mass = m_gt[i] + w * (m_gt[i + 1] - m_gt[i])
    y, valve = _tubing_state(src, np.array([gas_mass]), np.array([m_lt]), gas)
    return dict(bhp_bar=float(bhp[i] + w * (bhp[i + 1] - bhp[i])), valve=float(valve[0]), state=y[0])


def _stable(well, y, u):
    """Linear stability of the full plant (real reservoir) at a steady state: unstable means casing heading."""
    from data.well_ode import ode_numpy
    jac = np.zeros((3, 3))
    for j in range(3):
        h = max(1e-4 * abs(y[j]), 1e-3)
        a, b = y.copy(), y.copy()
        a[j] += h
        b[j] -= h
        try:
            jac[:, j] = (ode_numpy(a, u, well) - ode_numpy(b, u, well)) / (2 * h)
        except Exception:
            return None
    return bool(np.max(np.linalg.eigvals(jac).real) < 0)


@_with_gas_pinc
def nodal(settings, rates_sm3d=None):
    from scipy.optimize import brentq
    well, template = build_well(settings)
    rho_std = gas_std_density(well)
    wc = settings["water_cut"] / 100
    choke = settings["choke_fraction"]
    p_res_bar = well.P_res / 1e5
    pi_liquid = well.PI * (1 - well.alpha_G_bh_m)                  # kg/s/Pa of liquid
    q_max = pi_liquid * well.P_res / well.rho_L * 86400              # m³/d at zero bottom-hole pressure
    rho_in = well.P_gs * well.M_G / (R_GAS * well.T_an)
    g_cap = well.K_gs * math.sqrt(rho_in * well.P_gs)                # supply limit with an empty annulus, kg/s
    if rates_sm3d is None:
        # The deliverable lift gas is well below the empty-annulus limit (about 55-60 % of it for the gas_pinc wells).
        rates = list(np.linspace(0.05, 0.62, 12) * g_cap * 86400 / rho_std)
        for key in ("current_gas_sm3d", "env_min_gas", "env_max_gas"):
            v = settings.get(key)
            if v:
                rates.append(float(v))
        rates_sm3d = sorted({round(r, -2) for r in rates if r > 0})
    q_grid = np.linspace(0.02, 0.98, 22) * q_max
    inflow = lambda q: p_res_bar - q / 86400 * well.rho_L / pi_liquid / 1e5
    ipr = [dict(q=float(q), p=float(inflow(q))) for q in q_grid]

    def solve(rate, note=None):
        """Outflow curve at this lift-gas rate and its operating point."""
        gas = rate * rho_std / 86400
        vlp = []
        for q in q_grid:
            pt = _vlp_point(well, q / 86400 * well.rho_L, gas, choke)
            vlp.append(dict(q=float(q), p=pt["bhp_bar"] if pt else None, valve=pt["valve"] if pt else None))
        curve = dict(setpoint=float(rate), points=vlp, feasible=False)
        # Intersections: VLP pressure minus IPR pressure changes sign. The right-most crossing is the stable one.
        diff = [(v["p"] - i["p"]) if v["p"] is not None else None for v, i in zip(vlp, ipr)]
        crossing = [k for k in range(len(diff) - 1) if diff[k] is not None and diff[k + 1] is not None and diff[k] * diff[k + 1] <= 0]
        if not crossing:
            reason = ("Lift-gas supply pressure cannot deliver this rate" if all(v["valve"] is None or v["valve"] > 1 for v in vlp if v["p"] is not None)
                      else "Outflow curve does not meet the inflow curve: the well does not flow at this lift-gas rate")
            return curve, dict(setpoint=float(rate), feasible=False, note=reason)
        k = crossing[-1]

        def gap(q):
            pt = _vlp_point(well, q / 86400 * well.rho_L, gas, choke)
            return (pt["bhp_bar"] if pt else math.nan) - inflow(q)

        try:
            q_op = brentq(gap, q_grid[k], q_grid[k + 1], xtol=0.05)
        except ValueError:
            q_op = float(q_grid[k])
        op = _vlp_point(well, q_op / 86400 * well.rho_L, gas, choke)
        if op is None:
            return curve, dict(setpoint=float(rate), feasible=False, note="No steady operating state")
        if op["valve"] > 1.002:
            return curve, dict(setpoint=float(rate), feasible=False, valve=op["valve"],
                               note=f"Needs gas-lift valve {op['valve']:.0%} open: supply pressure too low")
        curve["feasible"] = True
        stable = _stable(well, op["state"], np.array([choke, min(op["valve"], 1.0)]))
        oil = q_op * (1 - wc)
        formation_gas = q_op * well.rho_L / 86400 * (well.alpha_G_bh_m / max(1 - well.alpha_G_bh_m, 1e-9)) * 86400 / rho_std
        return curve, dict(setpoint=float(rate), feasible=True, stable=stable is not False, valve=min(op["valve"], 1.0), liquid_m3d=q_op,
                           oil_m3d=oil, water_m3d=q_op * wc, gas_sm3d=formation_gas, bhp_bar=op["bhp_bar"],
                           note=note if stable is not False else "Unstable operating point (casing heading)")

    curves, points = [], []
    for rate in rates_sm3d:
        curve, point = solve(rate)
        curves.append(curve)
        points.append(point)
    # Maximum deliverable lift gas: interpolate where the gas-lift valve reaches fully open, between the last rate that
    # works and the first one the supply pressure cannot deliver.
    ok = [p for p in points if p["feasible"]]
    short = [p for p in points if not p["feasible"] and p.get("valve")]
    if ok and short and short[0]["setpoint"] > ok[-1]["setpoint"]:
        a, b = ok[-1], short[0]
        top = a["setpoint"] + (1 - a["valve"]) * (b["setpoint"] - a["setpoint"]) / (b["valve"] - a["valve"])
        top = math.floor(top / 100) * 100
        if top > a["setpoint"] + 100:
            curve, point = solve(top, note="Maximum lift gas: gas-lift valve fully open")
            if point["feasible"]:
                curves.append(curve)
                points.append(point)
    curves.sort(key=lambda c: c["setpoint"])
    feasible = sorted([p for p in points if p["feasible"]], key=lambda p: p["setpoint"])
    return dict(setpoint="lift_gas", setpoint_label="Lift-gas injection rate", setpoint_unit="sm³/d",
                points=feasible + [p for p in points if not p["feasible"]],
                nodal=dict(node="Bottom hole", rate_unit="m³/d", pressure_unit="bar a", ipr=ipr, curves=curves,
                           curve_label="Lift gas", curve_unit="sm³/d", reservoir_bar=p_res_bar, aof_m3d=q_max),
                model="Nodal analysis · gas_pinc plant equations (Jahanshahi)",
                notes=[f"Inflow: linear PI from reservoir pressure. Outflow: steady gas_pinc tubing, annulus and choke equations at "
                       f"each liquid rate with the lift-gas rate held. Valve, annulus and friction coefficients come from gas_pinc "
                       f"well {template} (nearest depth), scaled to this well.",
                       "Each operating point is the right-most IPR/VLP intersection. Stability is the sign of the plant's "
                       "eigenvalues there; unstable points are casing heading.",
                       f"Lift gas is converted at standard conditions ({rho_std:.3f} kg/sm³)."])
