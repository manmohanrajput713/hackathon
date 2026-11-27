"""Run registry: every model execution is a saved, reproducible run.

A run pins its field, scenario, wells, input dataset versions, model versions and parent runs.
create() validates and queues; execute() runs it (the Rust API calls it in the background).
"""
import json
import math
import sys
import traceback
import uuid
import numpy as np
import store
from schema import POLICIES, WELL_TYPES
from store import UserError, now
import analysis
import crmdata

KINDS = {
    "crm_match": "Deep-CRM history match",
    "crm_forecast": "Deep-CRM forecast",
    "vfm": "Virtual flow meter",
    "esp_risk": "ESP failure risk",
    "valve_risk": "Gas-lift valve risk",
    "well_curve": "Well model curve",
    "optimize": "Optimizer",
}
VFM_DATASET = {"esp": "esp_fast", "gas_lift": "gl_fast"}
DATASET_LABEL = {"esp_fast": "ESP high-frequency telemetry", "gl_fast": "gas-lift high-frequency telemetry"}
WELL_KINDS = {"vfm": (None, []), "esp_risk": ("esp", ["esp_daily", "events"]),
              "valve_risk": ("gas_lift", ["gl_pressure", "events"]), "well_curve": (None, [])}


def _row(r, full=True):
    out = dict(r)
    for k in ("params", "inputs", "summary"):
        out[k] = json.loads(out[k]) if out[k] else {}
    out["outputs"] = json.loads(out["outputs"]) if full and out["outputs"] else None
    if not full:
        out.pop("outputs", None)
    out["kind_label"] = KINDS.get(out["kind"], out["kind"])
    return out


def _pins(db, well_ids, datasets):
    return {w: {d: store.current_version(db, w, d) for d in datasets} for w in well_ids}


def _model_settings(well):
    settings = json.loads(well["settings"])
    from schema import SETTINGS
    return {s["key"]: settings.get(s["key"], s.get("default")) for s in SETTINGS.get(well["type"], []) if s["group"] == "Well model"}


def stale_reasons(db, run):
    """Why a finished run may no longer reflect current data or models. Its stored inputs are never changed."""
    reasons = []
    for well_id, sets in run["inputs"].get("datasets", {}).items():
        well = db.execute("SELECT name FROM wells WHERE id=?", (well_id,)).fetchone()
        if well is None:
            reasons.append("A well used by this run was deleted")
            continue
        for dataset, version in sets.items():
            current = store.current_version(db, well_id, dataset)
            if current != version:
                reasons.append(f"{well['name']} {dataset.replace('_', ' ')} changed (v{version} → v{current})")
    for name, version in run["inputs"].get("models", {}).items():
        try:
            if analysis.model_version(name) != version:
                reasons.append(f"Model files for {name.replace('_', ' ')} changed")
        except KeyError:
            pass
    pinned = run["inputs"].get("crm")
    if pinned:
        latest_version = crmdata.current_version(db, pinned["field"])
        if latest_version != pinned["version"]:
            reasons.append(f"CRM dataset changed (v{pinned['version']} → v{latest_version})")
        if run["inputs"].get("locations"):
            names = [r["name"] for r in run["inputs"]["locations"]]
            kinds = {r["name"]: r["kind"] for r in run["inputs"]["locations"]}
            now_loc = crmdata.registry_locations(db, [n for n in names if kinds[n] == "Producer"], [n for n in names if kinds[n] == "Injector"])
            if now_loc is None or {r["name"]: (r["easting"], r["northing"]) for r in now_loc} != {r["name"]: (r["easting"], r["northing"]) for r in run["inputs"]["locations"]}:
                reasons.append("Well positions on the canvas changed")
    if run["kind"] in ("well_curve", "vfm") and run["well_id"] and run["inputs"].get("settings") is not None:
        well = db.execute("SELECT * FROM wells WHERE id=?", (run["well_id"],)).fetchone()
        if well and _model_settings(well) != run["inputs"].get("settings"):
            reasons.append("Well model inputs were edited")
    for parent in run["inputs"].get("parents", []):
        p = db.execute("SELECT * FROM runs WHERE id=?", (parent,)).fetchone()
        if p is None:
            reasons.append("A run this depends on was deleted")
        elif p["status"] == "done" and stale_reasons(db, _row(p, full=False)):
            reasons.append(f"Upstream run '{p['name']}' needs review")
    return sorted(set(reasons))


def get(db, run_id, full=True):
    r = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if r is None:
        raise UserError("Run not found.", 404)
    run = _row(r, full)
    if run["status"] == "done":
        run["stale"] = stale_reasons(db, run)
    return run


def list_runs(db, body):
    sql, args = "SELECT * FROM runs WHERE 1=1", []
    for key in ("kind", "field", "scenario_id", "well_id", "status", "parent_id"):
        if body.get(key) not in (None, ""):
            sql += f" AND {key}=?"
            args.append(body[key])
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(min(int(body.get("limit", 200)), 500))
    out = []
    for r in db.execute(sql, args):
        run = _row(r, full=False)
        if run["status"] == "done":
            run["stale"] = stale_reasons(db, run)
        out.append(run)
    return out


def rename(db, run_id, body):
    get(db, run_id, full=False)
    name = str(body.get("name", "")).strip()
    if not name or len(name) > 100:
        raise UserError("Enter a run name of 1 to 100 characters.")
    with db:
        db.execute("UPDATE runs SET name=? WHERE id=?", (name, run_id))
    return get(db, run_id, full=False)


def delete(db, run_id):
    run = get(db, run_id, full=False)
    if run["status"] in ("queued", "running"):
        raise UserError("Wait for the run to finish before deleting it.")
    with db:
        db.execute("DELETE FROM runs WHERE id=?", (run_id,))
    return {"deleted": run_id}


# ---------- create ----------

def create(db, body):
    kind = body.get("kind")
    if kind not in KINDS:
        raise UserError("Unknown run kind.")
    params = dict(body.get("params") or {})
    field = str(body.get("field") or "")
    scenario_id = body.get("scenario_id") or None
    well_id, parent_id = params.get("well_id"), params.get("parent_id")
    inputs = {"models": {}, "datasets": {}, "parents": []}
    if kind in WELL_KINDS:
        well = store.get_well(db, well_id)
        want, datasets = WELL_KINDS[kind]
        if want and well["type"] != want:
            raise UserError(f"{KINDS[kind]} runs on {WELL_TYPES[want]['label']} wells.")
        field = well["field"]
        inputs["datasets"] = _pins(db, [well_id], datasets)
        if kind == "vfm":
            if well["type"] not in VFM_DATASET:
                raise UserError("The virtual flow meter runs on ESP and gas-lift producers.")
            dataset = VFM_DATASET[well["type"]]
            inputs["datasets"] = _pins(db, [well_id], [dataset])
            inputs["models"]["vfm" if well["type"] == "esp" else "gas_lift_vfm"] = analysis.model_version(
                "vfm" if well["type"] == "esp" else "gas_lift_vfm")
            if well["type"] == "gas_lift":
                inputs["settings"] = _model_settings(well)
            if not store.current_version(db, well_id, dataset):
                raise UserError(f"{well['name']} has no {DATASET_LABEL[dataset]} yet.")
        elif kind == "well_curve":
            if well["type"] not in ("esp", "gas_lift"):
                raise UserError("Well models exist for ESP and gas-lift producers.")
            inputs["settings"] = _model_settings(well)
            inputs["models"]["esp_curve" if well["type"] == "esp" else "gas_lift_curve"] = analysis.model_version(
                "esp_curve" if well["type"] == "esp" else "gas_lift_curve")
        else:
            inputs["models"][kind] = analysis.model_version(kind)
            if not any(store.current_version(db, well_id, d) for d in datasets[:1]):
                raise UserError(f"{well['name']} has no {datasets[0].replace('_', ' ')} data yet.")
        default_name = f"{KINDS[kind]} · {well['name']}"
    elif kind == "crm_match":
        data = crmdata.current(db, field)
        if not data:
            raise UserError("Save production.csv, injection.csv and time.csv on the Data tab first.")
        params = _match_params(params, data)
        inputs["crm"] = dict(field=field, version=data["version"])
        if params["initial_guess"] == "geometry":
            inputs["locations"] = data["locations"]  # positions from the canvas at the time of the run
        inputs["models"]["crm"] = analysis.model_version("crm")
        variant = "CRM-IP" if params["tau_selection"] == "per-pair" else "CRM-P"
        default_name = (f"{'CrmCompensated' if params['model'] == 'compensated' else 'CRM'} {variant} · {params['constraints']}"
                        f" · {params['epochs']} epochs · seed {params['seed']}")
    elif kind == "crm_forecast":
        parent = get(db, parent_id)
        if parent["kind"] != "crm_match" or parent["status"] != "done":
            raise UserError("Choose a completed history match.")
        field = parent["field"]
        params = _forecast_params(params, parent, field)
        inputs["parents"] = [parent_id]
        inputs["models"]["crm"] = analysis.model_version("crm")
        default_name = (f"Plan · hold last injection {params['steps']} steps" if params["mode"] == "hold"
                        else f"Plan · uploaded injection ({len(params['injection'])} steps)") + f" · {parent['name']}"
    elif kind == "optimize":
        scenario = store.get_scenario(db, scenario_id)
        field = scenario["field"]
        inputs["scenario"] = scenario
        default_name = f"Optimizer · {scenario['name']}"
    run_id = uuid.uuid4().hex[:12]
    name = str(body.get("name") or default_name).strip()[:100]
    with db:
        db.execute("INSERT INTO runs(id,kind,name,field,scenario_id,well_id,parent_id,status,created_at,params,inputs) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?,?)", (run_id, kind, name, field, scenario_id, well_id, parent_id, "queued", now(),
                                                     json.dumps(params), json.dumps(inputs)))
    return get(db, run_id)


# ---------- execute ----------

def execute(db, run_id):
    run = get(db, run_id)
    if run["status"] not in ("queued",):
        return run
    with db:
        db.execute("UPDATE runs SET status='running', started_at=? WHERE id=?", (now(), run_id))
    try:
        outputs = EXECUTORS[run["kind"]](db, run)
        if "rows" not in outputs and "points" not in outputs and "series" not in outputs and "wells" not in outputs \
                and "producers" not in outputs:
            raise UserError(outputs.get("skipped") or outputs.get("error") or "The model produced no output.")
        summary = outputs.get("summary", {})
        status, error = "done", None
        if outputs.get("parents"):  # runs discovered during execution (curves, risk) become pinned inputs
            inputs = dict(run["inputs"], parents=sorted(set(run["inputs"].get("parents", []) + outputs["parents"])))
            with db:
                db.execute("UPDATE runs SET inputs=? WHERE id=?", (json.dumps(inputs), run_id))
        if run["well_id"] and run["kind"] in WELL_KINDS:
            store.save_result(db, run["well_id"], run["kind"], dict(outputs, run_id=run_id))
    except UserError as e:
        outputs, summary, status, error = None, {}, "failed", str(e)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        outputs, summary, status, error = None, {}, "failed", f"Internal error: {e}"
    with db:
        db.execute("UPDATE runs SET status=?, finished_at=?, outputs=?, summary=?, error=? WHERE id=?",
                   (status, now(), json.dumps(outputs, allow_nan=False, default=float) if outputs is not None else None,
                    json.dumps(summary, default=float), error, run_id))
    return get(db, run_id, full=False)


def fail(db, run_id, message):
    with db:
        db.execute("UPDATE runs SET status='failed', finished_at=?, error=? WHERE id=? AND status IN ('queued','running')",
                   (now(), message[:500], run_id))
    return get(db, run_id, full=False)


def recover(db):
    """Runs left queued or running when the API stopped can never finish; mark them failed."""
    with db:
        n = db.execute("UPDATE runs SET status='failed', finished_at=?, error='Interrupted when the console stopped. Run it again.' "
                       "WHERE status IN ('queued','running')", (now(),)).rowcount
    return {"recovered": n}


def _well_rows(db, run, dataset):
    return store.read_rows(db, run["well_id"], dataset)


def _exec_vfm(db, run):
    well = store.get_well(db, run["well_id"])
    if well["type"] == "gas_lift":
        return analysis.gas_lift_vfm(_well_rows(db, run, "gl_fast"), run["inputs"].get("settings") or _model_settings(well))
    return analysis.esp_vfm(_well_rows(db, run, "esp_fast"), json.loads(well["settings"]))


def _exec_esp_risk(db, run):
    out = analysis.esp_risk(_well_rows(db, run, "esp_daily"))
    out["events"] = _well_rows(db, run, "events")
    return out


def _exec_valve_risk(db, run):
    well = store.get_well(db, run["well_id"])
    return analysis.valve_risk(well, _well_rows(db, run, "gl_pressure"), _well_rows(db, run, "events"))


def _exec_well_curve(db, run):
    import wellmodels
    well = store.get_well(db, run["well_id"])
    out = wellmodels.curve(well["type"], run["inputs"]["settings"])
    feasible = [p for p in out["points"] if p["feasible"]]
    if not feasible:
        raise UserError("No feasible operating point. Check reservoir pressure, PI and surface pressure.")
    best = max(feasible, key=lambda p: p["oil_m3d"])
    out["summary"] = dict(label="Well curve", value=len(feasible), unit="points",
                          note=f"Peak oil {best['oil_m3d']:.0f} m³/d at {best['setpoint']:,.0f} {out['setpoint_unit']}")
    return out


# ---------- Deep-CRM (pywaterflood user guide) ----------

TRAINING_DEFAULTS = dict(epochs=500, lr=1e-3, optimizer="rmsprop", hidden_layers=4, hidden_units=20, data_weight=1.0,
                         physics_weight=1.0, pressure_weight=1.0, constraint_weight=1.0, collocation_multiplier=1,
                         seed=0, num_cores=1, device="cpu")


def _match_params(p, data):
    """Validate the fit options described in 'CRM in practice'."""
    model = p.get("model", "crm")
    if model not in ("crm", "compensated"):
        raise UserError("Model must be CRM or CrmCompensated.")
    if model == "compensated" and not data["pressure"]:
        raise UserError("CrmCompensated needs pressure.csv (producer BHP) in the dataset.")
    tau = p.get("tau_selection", "per-pair")
    if tau not in ("per-pair", "per-producer"):
        raise UserError("Time constants must be per-pair (CRM-IP) or per-producer (CRM-P).")
    constraints = p.get("constraints", "up-to one")
    if constraints not in ("up-to one", "sum-to-one", "positive"):
        raise UserError("Constraints must be up-to one, sum-to-one or positive.")
    initial = p.get("initial_guess", "uniform")
    if initial not in ("uniform", "random", "geometry"):
        raise UserError("Initial guess must be uniform, random or geometry.")
    out = dict(model=model, primary=bool(p.get("primary", True)), tau_selection=tau, constraints=constraints,
               initial_guess=initial, random=bool(p.get("random", False)))
    try:
        for key, default in TRAINING_DEFAULTS.items():
            value = p.get(key, default)
            out[key] = value if key in ("optimizer", "device") else (int(value) if isinstance(default, int) else float(value))
        out["holdout_steps"] = int(p.get("holdout_steps", 0))
    except (TypeError, ValueError):
        raise UserError("Training options must be numbers.")
    if out["optimizer"] not in ("rmsprop", "adam"):
        raise UserError("Optimizer must be rmsprop or adam.")
    if out["device"] not in ("cpu", "cuda"):
        raise UserError("Device must be cpu or cuda.")
    if not 1 <= out["epochs"] <= 20000:
        raise UserError("Epochs must be between 1 and 20000.")
    if out["lr"] <= 0 or out["hidden_layers"] < 1 or out["hidden_units"] < 1 or out["collocation_multiplier"] < 1 or out["num_cores"] < 1:
        raise UserError("Learning rate, layers, units, collocation multiplier and cores must be positive.")
    if min(out["data_weight"], out["physics_weight"], out["pressure_weight"], out["constraint_weight"]) < 0:
        raise UserError("Loss weights cannot be negative.")
    n = len(data["time"])
    if not 0 <= out["holdout_steps"] <= n - 2:
        raise UserError(f"Hold out between 0 and {n - 2} time steps (the fit needs at least two).")
    if initial == "geometry":
        if not data["locations"]:
            raise UserError("The geometry seed needs a position for every well in the dataset: place them on the canvas or upload locations.csv.")
        try:
            out["x_e"], out["y_e"] = float(p["x_e"]), float(p["y_e"])
        except (KeyError, TypeError, ValueError):
            raise UserError("The geometry seed needs reservoir extents x_e and y_e.")
        if not (out["x_e"] > 0 and out["y_e"] > 0):
            raise UserError("x_e and y_e must be positive.")
    return out


def _rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=0)) if len(a) else None


def _exec_crm_match(db, run):
    from pywaterflood import CRM, CrmCompensated
    p = run["params"]
    data = crmdata.current(db, run["inputs"]["crm"]["field"], run["inputs"]["crm"]["version"])
    if data is None:
        raise UserError("The CRM dataset version this run used no longer exists.")
    q, inj, t = np.array(data["production"], float), np.array(data["injection"], float), np.array(data["time"], float)
    bhp = np.array(data["pressure"], float) if data["pressure"] else None
    n = len(t) - p["holdout_steps"]
    fit_kw = {k: p[k] for k in ("epochs", "lr", "optimizer", "hidden_layers", "hidden_units", "data_weight", "physics_weight",
                                "constraint_weight", "collocation_multiplier", "seed", "device")}
    fit_kw["num_cores"] = p["num_cores"]
    if p["initial_guess"] == "geometry":
        fit_kw.update(initial_guess="geometry", locations=crmdata._locations_frame(run["inputs"]["locations"]), x_e=p["x_e"], y_e=p["y_e"],
                      producer_ids=data["producers"], injector_ids=data["injectors"])
    elif p["initial_guess"] == "random" or p["random"]:
        fit_kw["random"] = True
    if p["model"] == "compensated":
        fit_kw["pressure_weight"] = p["pressure_weight"]
        model = CrmCompensated(primary=p["primary"], tau_selection=p["tau_selection"], constraints=p["constraints"])
        model.fit(q[:n], bhp[:n], inj[:n], t[:n], **fit_kw)
        q_hat = model.predict(inj, t, production=q, pressure=bhp)
    else:
        model = CRM(primary=p["primary"], tau_selection=p["tau_selection"], constraints=p["constraints"])
        model.fit(q[:n], inj[:n], t[:n], **fit_kw)
        q_hat = model.predict(inj, t, production=q)
    gains = np.asarray(model.gains, float)
    tau = np.asarray(model.tau, float)
    if p["tau_selection"] == "per-producer":
        tau = tau.reshape(-1)  # one time constant per producer
    resid = q - q_hat
    rmse_fit = _rmse(q[:n], q_hat[:n])
    rmse_hold = _rmse(q[n:], q_hat[n:]) if n < len(t) else None
    history = getattr(model, "loss_history_", []) or []
    step = max(1, len(history) // 500)
    out = dict(producers=data["producers"], injectors=data["injectors"], time=t.tolist(), train_rows=n,
               observed=q.tolist(), predicted=q_hat.round(6).tolist(), residual=resid.round(6).tolist(),
               rmse_fit=rmse_fit.tolist(), rmse_holdout=rmse_hold.tolist() if rmse_hold is not None else None,
               rmse_fit_overall=float(np.sqrt(np.mean((q[:n] - q_hat[:n]) ** 2))),
               rmse_holdout_overall=float(np.sqrt(np.mean((q[n:] - q_hat[n:]) ** 2))) if n < len(t) else None,
               gains=gains.round(6).tolist(), tau=tau.round(6).tolist(), tau_selection=p["tau_selection"],
               gains_producer=np.asarray(model.gains_producer, float).reshape(-1).tolist(),
               tau_producer=np.asarray(model.tau_producer, float).reshape(-1).tolist(),
               injector_sums=gains.sum(axis=0).round(6).tolist(), producer_sums=gains.sum(axis=1).round(6).tolist(),
               loss_history=[{k: float(v) for k, v in h.items() if v is not None} for h in history[::step]],
               injection=inj.tolist(), model=p["model"], primary=p["primary"], constraints=p["constraints"])
    if getattr(model, "productivity_index", None) is not None:
        out["productivity_index"] = np.asarray(model.productivity_index, float).reshape(-1).tolist()
    if p["model"] == "compensated":
        out["gain_pressure"] = np.asarray(model.gain_pressure, float).tolist()
        out["pressure"] = bhp.tolist()
    holdout = out["rmse_holdout_overall"]
    out["summary"] = dict(label="History match", value=holdout if holdout is not None else out["rmse_fit_overall"],
                          unit="held-out RMSE" if holdout is not None else "RMSE")
    return out


def _forecast_params(p, parent, field):
    o = parent["outputs"]
    mode = p.get("mode", "hold")
    t = np.array(o["time"])
    dt = float(p.get("dt") or (t[1] - t[0]))
    if dt <= 0:
        raise UserError("The time step must be positive.")
    if mode == "hold":
        steps = int(p.get("steps", 12))
        if not 1 <= steps <= 5000:
            raise UserError("Forecast between 1 and 5000 steps.")
        return dict(parent_id=parent["id"], mode="hold", steps=steps, dt=dt)
    if mode != "table":
        raise UserError("Forecast mode must be hold or table.")
    future = crmdata.staged_array(field, "future_injection")
    if future is None:
        raise UserError("Upload the future injection CSV first.")
    future = np.atleast_2d(future)
    if future.shape[1] != len(o["injectors"]):
        raise UserError(f"The future injection CSV needs {len(o['injectors'])} columns (one per injector, in order); it has {future.shape[1]}.")
    times = crmdata.staged_array(field, "future_time")
    if times is not None:
        if len(times) != len(future):
            raise UserError("The future time CSV must have one row per future injection row.")
        if times[0] <= t[-1]:
            raise UserError(f"Future times must start after the last history time ({t[-1]:g}).")
        times = times.tolist()
    else:
        times = (t[-1] + dt * np.arange(1, len(future) + 1)).tolist()
    return dict(parent_id=parent["id"], mode="table", injection=future.tolist(), time=times, steps=len(future), dt=dt)


def _exec_crm_forecast(db, run):
    from pywaterflood import CRM, CrmCompensated
    p = run["params"]
    o = get(db, p["parent_id"])["outputs"]
    t, inj, q = np.array(o["time"]), np.array(o["injection"]), np.array(o["observed"])
    if p["mode"] == "hold":
        t_future = t[-1] + p["dt"] * np.arange(1, p["steps"] + 1)
        i_future = np.repeat(inj[[-1], :], p["steps"], axis=0)
    else:
        t_future, i_future = np.array(p["time"]), np.array(p["injection"])
    t_all, i_all = np.concatenate([t, t_future]), np.vstack([inj, i_future])
    connections = dict(gains=np.array(o["gains"]), tau=np.array(o["tau"]), gains_producer=np.array(o["gains_producer"]),
                       tau_producer=np.array(o["tau_producer"]))
    notes = []
    if o["model"] == "compensated":
        bhp = np.array(o["pressure"])
        p_all = np.vstack([bhp, np.repeat(bhp[[-1], :], len(t_future), axis=0)])
        connections["gains_pressure"] = np.array(o["gain_pressure"])
        model = CrmCompensated(primary=o["primary"], tau_selection=o["tau_selection"], constraints=o["constraints"])
        q_all = model.predict(i_all, t_all, connections=connections, production=q, pressure=p_all)
        notes.append("Producer BHP is held at its last value over the plan.")
    else:
        model = CRM(primary=o["primary"], tau_selection=o["tau_selection"], constraints=o["constraints"])
        q_all = model.predict(i_all, t_all, connections=connections, production=q)
    q_plan = q_all[len(t):]
    return dict(producers=o["producers"], injectors=o["injectors"], history_time=t.tolist(), time=t_future.tolist(),
                observed=q.tolist(), history_predicted=q_all[:len(t)].round(6).tolist(), plan=q_plan.round(6).tolist(),
                injection=i_future.tolist(), forecast_start=float(t_future[0]), notes=notes,
                last_step=q_plan[-1].round(6).tolist(),
                summary=dict(label="Forecast", value=float(q_plan[-1].sum()), unit="field rate at last step", steps=len(t_future)))


# ---------- Optimizer ----------

def _latest_done(db, kind, well_id):
    r = db.execute("SELECT * FROM runs WHERE kind=? AND well_id=? AND status='done' ORDER BY finished_at DESC LIMIT 1",
                   (kind, well_id)).fetchone()
    return _row(r) if r else None


def _exec_optimize(db, run):
    from scipy.optimize import Bounds, LinearConstraint, milp
    import wellmodels
    scenario = run["inputs"]["scenario"]
    fac = scenario["facility"]
    policy = scenario["policy"]
    allow_unstable = bool(run["params"].get("allow_unstable"))
    allow_shut_in = bool(run["params"].get("allow_shut_in"))
    kwh_per_ksm3 = fac.get("compressor_kwh_per_1000sm3", 100)
    co2 = fac.get("emissions_kg_per_kwh", 0.4)
    we, wc = fac.get("energy_weight", 0), fac.get("emissions_weight", 0)
    wells, excluded, parents = [], [], []
    for w in db.execute("SELECT * FROM wells WHERE field=? AND type IN ('esp','gas_lift') ORDER BY name", (scenario["field"],)):
        settings = json.loads(w["settings"])
        lift = w["type"]
        if w["status"] != "active":
            excluded.append(dict(id=w["id"], well=w["name"], reason=f"Status is {w['status'].replace('_', ' ')}"))
            continue
        curve_run = _latest_done(db, "well_curve", w["id"])
        points, source = None, None
        if curve_run:
            points = [dict(setpoint=p["setpoint"], oil=p["oil_m3d"], water=p["water_m3d"], liquid=p["liquid_m3d"],
                           gas=p.get("gas_sm3d", 0) or 0, power=p.get("power_kw", 0) or 0, stable=p.get("stable", True),
                           note=p.get("note")) for p in curve_run["outputs"]["points"] if p["feasible"]]
            source = dict(kind="well_curve", run_id=curve_run["id"], label="Well model")
            parents.append(curve_run["id"])
        elif lift == "gas_lift" and store.read_rows(db, w["id"], "gl_curve"):
            points = [dict(setpoint=r["gas_sm3d"], oil=r["oil_m3d"], water=None, liquid=None, gas=0, power=0, stable=True, note=None)
                      for r in store.read_rows(db, w["id"], "gl_curve")]
            source = dict(kind="tested", label="Tested curve", version=store.current_version(db, w["id"], "gl_curve"))
        if not points or len(points) < 2:
            excluded.append(dict(id=w["id"], well=w["name"], reason="No response curve: run Well Models" +
                                 (" or enter a tested curve" if lift == "gas_lift" else "")))
            continue
        risk_kind = "esp_risk" if lift == "esp" else "valve_risk"
        risk = _latest_done(db, risk_kind, w["id"])
        if risk:
            parents.append(risk["id"])
        status = risk["summary"].get("status") if risk else None
        current = settings.get("current_hz" if lift == "esp" else "current_gas_sm3d")
        lo = settings.get("env_min_hz" if lift == "esp" else "env_min_gas", -math.inf)
        hi = settings.get("env_max_hz" if lift == "esp" else "env_max_gas", math.inf)
        applied = "Approved envelope"
        if status == "alert" and policy != "ignore":
            if current is None:
                excluded.append(dict(id=w["id"], well=w["name"], reason=f"In alert and policy '{policy}' needs a current setpoint; "
                                     "enter it on the well"))
                continue
            if policy == "hold":
                nearest = min(points, key=lambda pt: abs(pt["setpoint"] - current))["setpoint"]
                lo, hi, applied = nearest, nearest, "Held at current (alert)"
            else:
                hi, applied = min(hi, current), "Reduce only (alert)"
        allowed = [pt for pt in points if lo - 1e-9 <= pt["setpoint"] <= hi + 1e-9 and (allow_unstable or pt["stable"])]
        if allow_shut_in:  # shutting in is always available as a decision, outside the operating envelope
            allowed.append(dict(setpoint=0.0, oil=0.0, water=0.0, liquid=0.0, gas=0.0, power=0.0, stable=True, note="Shut in", shut_in=True))
        if not allowed:
            excluded.append(dict(id=w["id"], well=w["name"], reason=f"No stable curve point inside the envelope "
                                 f"{lo if math.isfinite(lo) else '−'}–{hi if math.isfinite(hi) else '∞'}"))
            continue
        for pt in allowed:
            pt["lift_gas"] = pt["setpoint"] if lift == "gas_lift" else 0.0
            pt["energy"] = pt["lift_gas"] / 1000 * kwh_per_ksm3 + pt["power"] * 24
            pt["emissions"] = pt["energy"] * co2
        cur = {k: wellmodels.interpolate([dict(p, feasible=True) for p in points], current, k)
               for k in ("oil", "liquid", "gas", "power")} if current is not None else None
        wells.append(dict(id=w["id"], well=w["name"], lift=lift, unit="Hz" if lift == "esp" else "sm³/d", points=allowed,
                          current=current, current_outcome=cur, risk=status, risk_run=risk["id"] if risk else None,
                          envelope=[lo if math.isfinite(lo) else None, hi if math.isfinite(hi) else None], policy_applied=applied,
                          source=source))
    if not wells:
        return dict(feasible=False, wells=[], excluded=excluded, constraints=[], scenario=scenario,
                    message="No well can be optimised. See the reasons per well.",
                    summary=dict(label="Optimizer", status="no_wells"))
    rows = [(i, pt) for i, w in enumerate(wells) for pt in w["points"]]
    resources = {
        "Lift gas": ("lift_gas", min([v for v in (fac.get("lift_gas_available_sm3d"), fac.get("compressor_capacity_sm3d")) if v is not None], default=None), "sm³/d"),
        "Separator liquid": ("liquid", fac.get("separator_liquid_m3d"), "m³/d"),
        "Separator gas": ("total_gas", fac.get("separator_gas_sm3d"), "sm³/d"),
        "ESP power": ("power", fac.get("esp_power_kw"), "kW"),
    }
    for _, pt in rows:
        pt["total_gas"] = (pt["gas"] or 0) + pt["lift_gas"]
        pt["liquid"] = pt["liquid"] if pt["liquid"] is not None else pt["oil"]
    matrix = [[float(i == j) for i, _ in rows] for j in range(len(wells))]
    lower, upper = [1.0] * len(wells), [1.0] * len(wells)
    active = []
    for label, (key, cap, unit) in resources.items():
        if cap is not None:
            matrix.append([pt[key] or 0 for _, pt in rows])
            lower.append(-np.inf)
            upper.append(cap)
            active.append(label)
    objective = [we * pt["energy"] + wc * pt["emissions"] - pt["oil"] for _, pt in rows]
    solved = milp(objective, integrality=np.ones(len(rows)), bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, lower, upper))
    constraints = []
    if solved.status == 2 or not solved.success:
        reasons = []
        for label in active:
            key, cap, unit = resources[label]
            need = sum(min(pt[key] or 0 for pt in w["points"]) for w in wells)
            if need > cap + 1e-9:
                reasons.append(f"{label}: the wells need at least {need:,.0f} {unit} at their lowest allowed setpoints, but only {cap:,.0f} is available")
            constraints.append(dict(name=label, unit=unit, capacity=cap, minimum_needed=need))
        message = "; ".join(reasons) or "The facility limits cannot all be met together at any allowed setpoints."
        hints = [h for h, on in (("allow shut-in", allow_shut_in), ("allow unstable points", allow_unstable)) if not on]
        if hints:
            message += ". Options: " + " or ".join(hints) + ", or raise the limit."
        return dict(feasible=False, wells=[dict(w, points=None) for w in wells], excluded=excluded, constraints=constraints,
                    message=message, scenario=scenario, summary=dict(label="Optimizer", status="infeasible"))
    chosen = {}
    for x, (i, pt) in zip(solved.x, rows):
        if x > 0.5:
            chosen[i] = pt
    results, totals = [], dict(oil=0, liquid=0, gas=0, lift_gas=0, power=0, energy=0, emissions=0, current_oil=0)
    for i, w in enumerate(wells):
        pt = chosen[i]
        cur = w["current_outcome"]
        delta = pt["oil"] - cur["oil"] if cur and cur["oil"] is not None else None
        results.append(dict(id=w["id"], well=w["well"], lift=w["lift"], unit=w["unit"], current=w["current"],
                            current_oil=cur["oil"] if cur else None, recommended=pt["setpoint"], oil=pt["oil"], liquid=pt["liquid"],
                            gas=pt["gas"], lift_gas=pt["lift_gas"], power=pt["power"], energy=pt["energy"], emissions=pt["emissions"],
                            delta_oil=delta, risk=w["risk"], risk_run=w["risk_run"], envelope=w["envelope"],
                            policy_applied=w["policy_applied"], source=w["source"], note=pt.get("note"), shut_in=bool(pt.get("shut_in")),
                            curve=[dict(setpoint=p["setpoint"], oil=p["oil"]) for p in w["points"]]))
        for k in ("oil", "liquid", "gas", "lift_gas", "power", "energy", "emissions"):
            totals[k] += pt[k] or 0
        totals["current_oil"] += cur["oil"] if cur and cur["oil"] is not None else 0
    totals["total_gas"] = totals["gas"] + totals["lift_gas"]
    for label in active:
        key, cap, unit = resources[label]
        used = sum(chosen[i][key] or 0 for i in chosen)
        # Binding: little slack, or some well could raise oil by moving to a point that would exceed the limit.
        better_blocked = any(pt["oil"] > chosen[i]["oil"] and used - (chosen[i][key] or 0) + (pt[key] or 0) > cap + 1e-9
                             for i, w in enumerate(wells) for pt in w["points"])
        constraints.append(dict(name=label, unit=unit, capacity=cap, used=used, slack=cap - used,
                                binding=bool(cap - used <= 0.01 * max(cap, 1e-9) or better_blocked)))
    return dict(feasible=True, wells=results, excluded=excluded, constraints=constraints, totals=totals, scenario=scenario,
                policy=POLICIES[policy], allow_unstable=allow_unstable, allow_shut_in=allow_shut_in,
                summary=dict(label="Optimizer", status="ok", value=totals["oil"], unit="m³/d oil",
                             delta=totals["oil"] - totals["current_oil"] if totals["current_oil"] else None,
                             binding=[c["name"] for c in constraints if c.get("binding")]),
                parents=parents)


EXECUTORS = {"vfm": _exec_vfm, "esp_risk": _exec_esp_risk, "valve_risk": _exec_valve_risk, "well_curve": _exec_well_curve,
             "crm_match": _exec_crm_match, "crm_forecast": _exec_crm_forecast, "optimize": _exec_optimize}
