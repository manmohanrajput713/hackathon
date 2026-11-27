"""Nodal analysis for the Well Models tab, from the project's physics models. The node is the bottom hole.

ESP: inflow q = PI (pr - pbh). Outflow is the pinn BCS momentum balance (Krishnamoorthy et al.) solved for the bottom-hole
     pressure the well needs at each rate: pbh = pwh + rho g hw + F1 + F2 - rho g H(q, f), with pwh from the choke
     equation q = zc Cc sqrt(pwh - pm) and the pump head H scaled by the affinity laws. One outflow curve per frequency.
Gas lift: gaslift.nodal(), the gas_pinc (Jahanshahi) plant at steady state, one outflow curve per lift-gas rate.
Operating points (IPR/VLP intersections) keep the same fields as before, so the optimizer reads them unchanged.
"""
import importlib.util
import numpy as np
from common import ROOT


def _pinn_constants():
    spec = importlib.util.spec_from_file_location("pinn_param", ROOT / "pinn/data/param.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def esp_curve(settings, frequencies=None):
    from scipy.optimize import brentq
    c = _pinn_constants()
    pr = settings["reservoir_bar"] * 1e5
    pi = settings["pi_m3d_bar"] / 86400 / 1e5           # m³/s/Pa
    rho = settings["density"]
    pm = settings["manifold_bar"] * 1e5
    zc = settings["choke_percent"] / 100
    wc = settings["water_cut"] / 100
    gor = settings["gor"]
    eff = settings["pump_efficiency"] / 100
    frequencies = frequencies or list(np.arange(35.0, 65.01, 2.5))

    def head(q, f):
        q0 = q / c.Cq * (c.f0 / f)
        return c.CH * (-1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2) * (f / c.f0) ** 2

    def friction(q, length):
        return 0.158 * (rho * length * q ** 2 / (c.D1 * c.A1 ** 2)) * (c.mu / (rho * c.D1 * q)) ** 0.25

    def outflow(q, f):
        """Bottom-hole pressure (Pa) the tubing, pump and choke need to deliver q (m³/s) at frequency f."""
        pwh = pm + (q / (zc * c.Cc)) ** 2
        return pwh + rho * c.g * c.hw + friction(q, c.L1) + friction(q, c.L2) - rho * c.g * head(q, f)

    inflow = lambda q: pr - q / pi
    residual = lambda q, f: inflow(q) - outflow(q, f)
    qmax = pi * pr
    q_grid = np.linspace(0.02, 0.98, 30) * qmax
    ipr = [dict(q=float(q * 86400), p=float(inflow(q) / 1e5)) for q in q_grid]
    curves, points = [], []
    fine = np.linspace(1e-6, qmax * 0.999, 400)
    for f in frequencies:
        # Below zero absolute pressure the pump would cavitate long before; those outflow points are left out.
        curves.append(dict(setpoint=float(f), points=[dict(q=float(q * 86400), p=float(outflow(q, f) / 1e5) if outflow(q, f) > 0 else None)
                                                      for q in q_grid]))
        values = np.array([residual(q, f) for q in fine])
        crossing = np.flatnonzero((values[:-1] > 0) & (values[1:] <= 0))
        if not len(crossing):
            points.append(dict(setpoint=float(f), feasible=False, note="Outflow curve stays above inflow: the pump cannot lift fluid to surface at this frequency"))
            continue
        i = crossing[-1]
        q = brentq(residual, fine[i], fine[i + 1], args=(f,))
        liquid = q * 86400
        h = head(q, f)
        hydraulic_kw = rho * c.g * h * q / 1000
        q_ref = q * c.f0 / f  # flow at reference speed, for the recommended operating range
        in_range = c.q0_dt <= q_ref <= c.q0_ut
        oil = liquid * (1 - wc)
        points.append(dict(setpoint=float(f), feasible=True, liquid_m3d=liquid, oil_m3d=oil, water_m3d=liquid * wc,
                           gas_sm3d=oil * gor, bhp_bar=inflow(q) / 1e5, whp_bar=(pm + (q / (zc * c.Cc)) ** 2) / 1e5,
                           head_m=h, hydraulic_kw=hydraulic_kw, power_kw=hydraulic_kw / eff, in_pump_range=bool(in_range),
                           note=None if in_range else ("Below pump range (downthrust)" if q_ref < c.q0_dt else "Above pump range (upthrust)")))
    return dict(setpoint="frequency", setpoint_label="Pump frequency", setpoint_unit="Hz", points=points,
                nodal=dict(node="Bottom hole", rate_unit="m³/d", pressure_unit="bar a", ipr=ipr, curves=curves,
                           curve_label="Pump", curve_unit="Hz", reservoir_bar=pr / 1e5, aof_m3d=qmax * 86400),
                model="Nodal analysis · pinn BCS equations",
                notes=["Inflow: linear PI from reservoir pressure. Outflow: wellhead choke, tubing friction and hydrostatic head minus "
                       "the pump head at each frequency. Pump curve, tubing and choke geometry are the pinn reference ESP well.",
                       "Each operating point is the right-most IPR/VLP intersection. Formation gas is oil × GOR "
                       "(the ESP equations are single-phase liquid). Power is hydraulic power divided by the pump and motor efficiency."])


def gas_lift_curve(settings):
    import gaslift
    return gaslift.nodal(settings)


def curve(well_type, settings):
    if well_type == "esp":
        return esp_curve(settings)
    if well_type == "gas_lift":
        return gas_lift_curve(settings)
    raise ValueError("Well models exist for ESP and gas-lift producers only.")


def interpolate(points, setpoint, key):
    pts = sorted((p for p in points if p.get("feasible") and p.get(key) is not None), key=lambda p: p["setpoint"])
    if not pts or setpoint is None:
        return None
    xs = [p["setpoint"] for p in pts]
    if setpoint <= xs[0]:
        return pts[0][key]
    if setpoint >= xs[-1]:
        return pts[-1][key]
    return float(np.interp(setpoint, xs, [p[key] for p in pts]))
