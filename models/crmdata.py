"""Deep-CRM input data in the pywaterflood layout, plus the companion models from the user guide.

Files (headerless, comma separated, one row per time step, columns in well order):
  production.csv  (n_time, n_producers)     injection.csv (n_time, n_injectors)
  time.csv        (n_time,) elapsed time    pressure.csv  (n_time, n_producers) producer BHP, optional
  locations.csv   header: name,easting,northing,kind (Producer/Injector), optional
"""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from common import RUN
import store
from store import UserError, now

STAGING = RUN / "crm_staging"
MATRIX_KINDS = {"production", "injection", "time", "pressure", "aquifer_dp", "future_injection", "future_time"}
KINDS = MATRIX_KINDS | {"locations"}
LABELS = {"production": "production.csv", "injection": "injection.csv", "time": "time.csv", "pressure": "pressure.csv",
          "locations": "locations.csv", "aquifer_dp": "pressure-drop CSV", "future_injection": "future injection CSV",
          "future_time": "future time CSV"}


def _table(db):
    db.execute("""CREATE TABLE IF NOT EXISTS crm_data(
        field TEXT NOT NULL, version INTEGER NOT NULL, created_at TEXT NOT NULL, source TEXT NOT NULL,
        producers TEXT NOT NULL, injectors TEXT NOT NULL, time TEXT NOT NULL, production TEXT NOT NULL,
        injection TEXT NOT NULL, pressure TEXT, locations TEXT, PRIMARY KEY(field, version))""")


def _folder(field):
    return STAGING / hashlib.sha1(field.encode()).hexdigest()[:16]


def read_matrix(path, label):
    """Headerless numeric CSV, as numpy.genfromtxt reads pywaterflood's files."""
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise UserError(f"{label} is empty.")
    try:
        data = np.genfromtxt(path, delimiter=",", dtype=float, encoding="utf-8-sig")
    except Exception as error:
        raise UserError(f"{label} could not be read as numbers: {error}")
    data = np.atleast_1d(data)
    if np.isnan(data).any():
        first = np.argwhere(np.isnan(np.atleast_2d(data)))[0]
        raise UserError(f"{label} has a missing or non-numeric value at row {first[0] + 1}, column {first[1] + 1}. "
                        "Files must be headerless numbers (pywaterflood layout); fill shut-in periods with 0.")
    return data


def _check_rates(data, label):
    # A one-well file is a single column, which genfromtxt returns as a flat array.
    data = data.reshape(-1, 1) if data.ndim == 1 else data
    if not np.isfinite(data).all():
        raise UserError(f"{label} must contain finite values.")
    if (data < 0).any():
        r, c = np.argwhere(data < 0)[0]
        raise UserError(f"{label} has a negative value at row {r + 1}, column {c + 1}. Values must be non-negative.")
    return data


def _summary(kind, data):
    if kind == "locations":
        return dict(rows=len(data), producers=sum(r["kind"] == "Producer" for r in data), injectors=sum(r["kind"] == "Injector" for r in data))
    arr = np.atleast_2d(data) if kind not in ("time", "future_time", "aquifer_dp") else np.asarray(data).reshape(-1)
    out = dict(rows=int(arr.shape[0]), columns=int(arr.shape[1]) if arr.ndim == 2 else 1,
               min=float(np.min(arr)), max=float(np.max(arr)))
    return out


def parse(kind, path):
    label = LABELS[kind]
    if kind == "locations":
        import pandas as pd
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as error:
            raise UserError(f"{label} could not be read: {error}")
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        missing = {"name", "easting", "northing", "kind"} - set(frame.columns)
        if missing:
            raise UserError(f"{label} needs the columns name, easting, northing, kind. Missing: {', '.join(sorted(missing))}.")
        rows = []
        for i, r in enumerate(frame.to_dict("records"), start=2):
            kind_value = str(r["kind"]).strip().capitalize()
            if kind_value not in ("Producer", "Injector"):
                raise UserError(f"{label} row {i}: kind must be Producer or Injector.")
            try:
                x, y = float(r["easting"]), float(r["northing"])
            except (TypeError, ValueError):
                raise UserError(f"{label} row {i}: easting and northing must be numbers.")
            if not (np.isfinite(x) and np.isfinite(y)):
                raise UserError(f"{label} row {i}: easting and northing must be finite.")
            rows.append(dict(name=str(r["name"]).strip(), easting=x, northing=y, kind=kind_value))
        names = [r["name"] for r in rows]
        if len(set(names)) != len(names):
            raise UserError(f"{label} has repeated well names.")
        return rows
    data = read_matrix(path, label)
    if kind in ("time", "future_time", "aquifer_dp"):
        if data.ndim == 2 and min(data.shape) != 1:
            raise UserError(f"{label} must have a single column.")
        data = data.reshape(-1)
        if kind != "aquifer_dp" and len(data) > 1 and (np.diff(data) <= 0).any():
            k = int(np.argwhere(np.diff(data) <= 0)[0][0])
            raise UserError(f"{label} must increase: row {k + 2} is not after row {k + 1}.")
        return data
    return _check_rates(data, label)


def stage(db, field, kind, path):
    if kind not in KINDS:
        raise UserError("Unknown file type.", 404)
    data = parse(kind, path)
    folder = _folder(field)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, folder / f"{kind}.csv")
    return dict(kind=kind, file=LABELS[kind], **_summary(kind, data))


def staged(field):
    folder = _folder(field)
    out = {}
    if folder.exists():
        for f in folder.glob("*.csv"):
            kind = f.stem
            if kind in KINDS:
                try:
                    out[kind] = dict(kind=kind, file=LABELS[kind], **_summary(kind, parse(kind, f)))
                except UserError as error:
                    out[kind] = dict(kind=kind, file=LABELS[kind], error=str(error))
    return out


def staged_array(field, kind):
    path = _folder(field) / f"{kind}.csv"
    if not path.exists():
        return None
    return parse(kind, path)


def unstage(field):
    shutil.rmtree(_folder(field), ignore_errors=True)
    return {"cleared": True}


def _ids(text, n, prefix):
    ids = [s.strip() for s in (text if isinstance(text, list) else str(text or "").split(",")) if s.strip()]
    if not ids:
        ids = [f"{prefix}{k + 1}" for k in range(n)]
    if len(ids) != n:
        raise UserError(f"Enter {n} {'producer' if prefix == 'P' else 'injector'} IDs in column order (got {len(ids)}).")
    if len(set(ids)) != len(ids):
        raise UserError("Well IDs must be unique.")
    return ids


def current(db, field, version=None):
    _table(db)
    sql = "SELECT * FROM crm_data WHERE field=?" + (" AND version=?" if version else "") + " ORDER BY version DESC LIMIT 1"
    row = db.execute(sql, (field, version) if version else (field,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    for k in ("producers", "injectors", "time", "production", "injection", "pressure"):
        out[k] = json.loads(out[k]) if out[k] else None
    out["locations"] = registry_locations(db, out["producers"], out["injectors"])
    return out


def registry_locations(db, producers, injectors):
    """Well positions from the field canvas, in the locations.csv layout. None unless every well has a position."""
    rows = []
    for name, kind in [(w, "Producer") for w in producers] + [(w, "Injector") for w in injectors]:
        well = store.find_well(db, name)
        if well is None or well["easting"] is None:
            return None
        rows.append(dict(name=name, easting=well["easting"], northing=well["northing"], kind=kind))
    return rows


def _link_wells(db, field, producers, injectors, locations):
    """Every CRM column is a well on the canvas: create missing wells and apply locations.csv positions."""
    pos = {r["name"]: r for r in (locations or [])}
    pairs = [(w, "producer") for w in producers] + [(w, "injector") for w in injectors]
    for name, role in pairs:
        well = store.find_well(db, name)
        if well is not None and store.WELL_TYPES[well["type"]]["role"] != role:
            raise UserError(f"{name} is a {store.WELL_TYPES[well['type']]['role']} on the canvas but a {role} column here.")
        if well is not None and well["field"] != field:
            raise UserError(f"Well ID {name} already exists in field {well['field'] or '(no field)'}.")
    for name, role in pairs:
        well = store.find_well(db, name)
        xy = {"easting": pos[name]["easting"], "northing": pos[name]["northing"]} if name in pos else {}
        if well is None:
            store.create_well(db, dict(code=name, name=name, type=role, field=field, **xy))
        elif xy:
            store.update_well(db, well["id"], xy)


def current_version(db, field):
    _table(db)
    return db.execute("SELECT MAX(version) FROM crm_data WHERE field=?", (field,)).fetchone()[0] or 0


def commit(db, body):
    field = body.get("field", "") or ""
    base = current(db, field)
    arrays = {}
    for kind in ("production", "injection", "time", "pressure"):
        new = staged_array(field, kind)
        arrays[kind] = new if new is not None else (np.array(base[kind]) if base and base[kind] is not None else None)
    locations = staged_array(field, "locations")
    for kind in ("production", "injection", "time"):
        if arrays[kind] is None:
            raise UserError(f"Upload {LABELS[kind]} first.")
    q, inj, t, p = np.atleast_2d(arrays["production"]), np.atleast_2d(arrays["injection"]), np.asarray(arrays["time"]).reshape(-1), arrays["pressure"]
    if len(t) < 2:
        raise UserError("At least two time steps are required (the CRM convolution reads time[1]).")
    for label, arr in (("production.csv", q), ("injection.csv", inj)):
        if arr.shape[0] != len(t):
            raise UserError(f"{label} has {arr.shape[0]} rows but time.csv has {len(t)}. Every series needs the same number of steps as time.")
    producers = _ids(body.get("producer_ids"), q.shape[1], "P")
    injectors = _ids(body.get("injector_ids"), inj.shape[1], "I")
    if set(producers) & set(injectors):
        raise UserError("A well ID cannot be both a producer and an injector.")
    if p is not None:
        p = np.atleast_2d(p)
        if p.shape != q.shape:
            raise UserError(f"pressure.csv must have the same shape as production.csv ({q.shape[0]} × {q.shape[1]}); it is {p.shape[0]} × {p.shape[1]}.")
    if locations:
        kinds = {r["name"]: r["kind"] for r in locations}
        missing = [w for w in producers + injectors if w not in kinds]
        if missing:
            raise UserError(f"locations.csv has no row for: {', '.join(missing)}. Names must match the well IDs.")
        wrong = [w for w in producers if kinds[w] != "Producer"] + [w for w in injectors if kinds[w] != "Injector"]
        if wrong:
            raise UserError(f"locations.csv kind does not match the role of: {', '.join(wrong)}.")
    _link_wells(db, field, producers, injectors, locations)
    files = [LABELS[k] for k in ("production", "injection", "time", "pressure", "locations") if (_folder(field) / f"{k}.csv").exists()]
    _insert(db, field, body.get("source") or ("uploaded " + ", ".join(files) if files else "edited well IDs"), producers, injectors, t, q, inj, p)
    unstage(field)
    return get(db, field)


def _insert(db, field, source, producers, injectors, t, q, inj, p):
    version = current_version(db, field) + 1
    with db:
        db.execute("INSERT INTO crm_data VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (field, version, now(), source[:300], json.dumps(producers), json.dumps(injectors), json.dumps(np.asarray(t).tolist()),
                    json.dumps(np.asarray(q).tolist()), json.dumps(np.asarray(inj).tolist()),
                    json.dumps(np.asarray(p).tolist()) if p is not None else None, None))
    return version


def set_column(db, body):
    """Edit one well's column of production (producer) or injection (injector) rates; saves a new dataset version."""
    field = body.get("field", "") or ""
    data = current(db, field)
    if not data:
        raise UserError("Save time.csv on Deep-CRM → Data first: the rate column needs the dataset's time axis.")
    well = store.find_well(db, str(body.get("well", "")))
    if well is None:
        raise UserError("Well not found.", 404)
    code = well["code"]
    role = store.WELL_TYPES[well["type"]]["role"]
    try:
        values = np.array([float(v) for v in body.get("values", [])], float)
    except (TypeError, ValueError):
        raise UserError("Every rate must be a number.")
    t = np.array(data["time"])
    if len(values) != len(t):
        raise UserError(f"Enter one rate per time step ({len(t)} rows).")
    if not np.isfinite(values).all() or (values < 0).any():
        raise UserError("Rates must be finite and non-negative (use 0 for shut-in).")
    q, inj = np.array(data["production"], float), np.array(data["injection"], float)
    producers, injectors = list(data["producers"]), list(data["injectors"])
    p = np.array(data["pressure"], float) if data["pressure"] else None
    if role == "producer":
        if code in producers:
            q[:, producers.index(code)] = values
        else:
            if p is not None:
                raise UserError("This dataset has pressure.csv, so a new producer also needs a pressure column. Add it on Deep-CRM → Data.")
            q = np.column_stack([q, values])
            producers.append(code)
    else:
        if code in injectors:
            inj[:, injectors.index(code)] = values
        else:
            inj = np.column_stack([inj, values])
            injectors.append(code)
    version = _insert(db, field, f"edited {code} {'production' if role == 'producer' else 'injection'} column", producers, injectors, t, q, inj, p)
    return dict(version=version, well=code, role=role)


def staged_data(field, kind):
    """Contents of a staged file, so its table can be shown for editing."""
    if kind not in KINDS:
        raise UserError("Unknown file type.", 404)
    data = staged_array(field, kind)
    if data is None:
        return dict(kind=kind, rows=None)
    if kind == "locations":
        return dict(kind=kind, rows=data)
    return dict(kind=kind, rows=np.asarray(data).tolist())


def get(db, field):
    data = current(db, field)
    _table(db)
    history = [dict(r) for r in db.execute("SELECT version, created_at, source FROM crm_data WHERE field=? ORDER BY version DESC LIMIT 15", (field,))]
    if data is None:
        return dict(field=field, data=None, staged=staged(field), versions=history)
    q, inj = np.array(data["production"]), np.array(data["injection"])
    balance = dict(production=q.sum(axis=1).tolist(), injection=inj.sum(axis=1).tolist(),
                   ratio=float(q.sum() / inj.sum()) if inj.sum() > 0 else None)
    return dict(field=field, data=data, staged=staged(field), versions=history, balance=balance)


# ---------- companion models ----------

def _locations_frame(locations):
    import pandas as pd
    from pywaterflood import translate_locations
    raw = pd.DataFrame(locations).set_index("name")
    raw.index.name = None  # as in the guide: an unnamed index of well IDs
    return translate_locations(raw, "easting", "northing", "kind")


def mpi(db, body):
    from pywaterflood import geometry_crm_gains
    from pywaterflood.multiwellproductivity import calc_gains_homogeneous
    data = current(db, body.get("field", "") or "")
    if not data or not data["locations"]:
        raise UserError("Every well in the CRM dataset needs a position: place the wells on the canvas or upload locations.csv on the Data tab.")
    loc = _locations_frame(data["locations"])
    try:
        x_e, y_e = float(body["x_e"]), float(body["y_e"])
    except (KeyError, TypeError, ValueError):
        raise UserError("Enter numeric reservoir extents x_e and y_e.")
    if not (x_e > 0 and y_e > 0 and np.isfinite(x_e) and np.isfinite(y_e)):
        raise UserError("x_e and y_e must be finite and positive.")
    if loc["X"].max() >= x_e or loc["Y"].max() >= y_e:
        raise UserError(f"Every well must sit inside the rectangle after translation: largest X is {loc['X'].max():g}, largest Y is {loc['Y'].max():g}.")
    prod, inj = data["producers"], data["injectors"]
    lam = calc_gains_homogeneous(loc, x_e, y_e)
    geo = geometry_crm_gains(loc, x_e, y_e, producer_ids=prod, injector_ids=inj)
    return dict(producers=prod, injectors=inj, x_e=x_e, y_e=y_e,
                translated=[dict(name=n, X=float(r.X), Y=float(r.Y), kind=r.Type) for n, r in loc.iterrows()],
                lambda_=lam.loc[prod, inj].to_numpy().round(5).tolist(), seed=np.asarray(geo).round(5).tolist())


def buckley_leverett(db, body):
    from pywaterflood.buckleyleverett import Reservoir, breakthrough_sw, water_front_velocity
    keys = ["phi", "viscosity_oil", "viscosity_water", "sat_oil_r", "sat_water_c", "sat_gas_c", "n_oil", "n_water", "flow_cross_section"]
    try:
        values = {k: float(body[k]) for k in keys}
        flow_rate = float(body.get("flow_rate", 1.0))
    except (KeyError, TypeError, ValueError):
        raise UserError("Enter all rock and fluid values as numbers.")
    try:
        rock = Reservoir(**values)
        sw_bt = float(breakthrough_sw(rock))
        v_bt = float(water_front_velocity(rock, sw_bt, flow_rate))
    except (ValueError, AssertionError) as error:
        raise UserError(f"Invalid rock or fluid values: {error}")
    saturations = np.linspace(0.0, 1.0 - rock.sat_oil_r, 250)
    velocity = [float(water_front_velocity(rock, sw, flow_rate)) for sw in saturations]
    times = list(range(0, 17, 2))
    s_w = np.linspace(0.0, 1.0 - rock.sat_oil_r, 120)
    v = np.array([water_front_velocity(rock, sw, flow_rate) for sw in s_w])
    profiles = [dict(Sw=float(s), **{f"t{t}": float(x * t) for t in times}) for s, x in zip(s_w, v)]
    out = dict(sw_bt=sw_bt, v_bt=v_bt, mobile_oil=1 - rock.sat_oil_r - sw_bt, flow_rate=flow_rate,
               curve=[dict(Sw=float(s), v=float(x)) for s, x in zip(saturations, velocity)], profiles=profiles, times=times)
    spacing = body.get("spacing_m")
    if spacing not in (None, ""):
        spacing = float(spacing)
        if spacing <= 0:
            raise UserError("Spacing must be positive.")
        out.update(spacing_m=spacing, travel_days=spacing / v_bt if v_bt > 0 else None)
    return out


def aquifer(db, body):
    field = body.get("field", "") or ""
    data = current(db, field)
    if not data:
        raise UserError("Save a CRM dataset on the Data tab first.")
    q, inj, t = np.array(data["production"]), np.array(data["injection"]), np.array(data["time"])
    mode = body.get("mode", "imbalance")
    params = {}
    if mode == "imbalance":
        rate = np.clip(q.sum(axis=1) - inj.sum(axis=1), 0.0, None)
    elif mode == "influx":
        from pywaterflood.aquifer import aquifer_production, effective_reservoir_radius
        dp = staged_array(field, "aquifer_dp")
        if dp is None:
            raise UserError("Upload the pressure-drop CSV (one value per time step) first.")
        if len(dp) != len(t):
            raise UserError(f"The pressure-drop CSV has {len(dp)} rows but time.csv has {len(t)}.")
        try:
            p = {k: float(body[k]) for k in ("pore_volume", "porosity", "thickness", "solid_angle", "k_a", "mu", "c_t", "phi_a", "r_ed", "aquifer_constant")}
        except (KeyError, TypeError, ValueError):
            raise UserError("Enter all aquifer values as numbers.")
        method = body.get("method", "klins")
        if method not in ("klins", "marsal-walsh"):
            raise UserError("Method must be klins or marsal-walsh.")
        years = t / 365.25 if body.get("time_unit", "days") == "days" else t
        r_o = effective_reservoir_radius(p["pore_volume"], p["porosity"], p["thickness"], p["solid_angle"])
        k_t = 2.309 * p["k_a"] / (p["phi_a"] * p["mu"] * p["c_t"] * r_o ** 2)
        t_d = k_t * years
        w_e = aquifer_production(dp, t_d, r_ed=p["r_ed"], method=method, aquifer_constant=p["aquifer_constant"])
        rate = np.diff(np.concatenate([[0.0], w_e]))
        params = dict(p, method=method, r_o=float(r_o), k_t=float(k_t), time_unit=body.get("time_unit", "days"),
                      cumulative=np.asarray(w_e).tolist())
    else:
        raise UserError("Mode must be imbalance or influx.")
    rate = np.clip(np.asarray(rate, float), 0.0, None)
    out = dict(mode=mode, time=t.tolist(), rate=rate.tolist(), params=params,
               positive_steps=int((rate > 0).sum()), steps=len(rate))
    if body.get("apply"):
        name = str(body.get("name") or "AQUIFER").strip()
        if name in data["producers"] + data["injectors"]:
            raise UserError(f"A well named {name} is already in the dataset.")
        version = _insert(db, field, f"added pseudo-injector {name} ({'field imbalance' if mode == 'imbalance' else 'aquifer influx'})",
                          data["producers"], data["injectors"] + [name], data["time"], data["production"],
                          np.column_stack([inj, rate]), data["pressure"])
        out["applied_version"] = version
    return out
