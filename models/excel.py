"""Templates (Excel or CSV), per-well dataset imports, and Excel/CSV exports of data and runs."""
import csv
import io
import json
import re
from pathlib import Path
from common import RUN
from schema import DATASETS, WELL_TYPES
import store
from store import UserError
import intake

EXPORTS = RUN / "exports"


def _header(col):
    return f"{col['label']} ({col['unit']})" if col.get("unit") else col["label"]


def _style(ws):
    from openpyxl.styles import Font, PatternFill
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="12664C")
    ws.freeze_panes = "A2"
    for column in ws.columns:
        width = max((len(str(c.value)) for c in column if c.value is not None), default=8)
        ws.column_dimensions[column[0].column_letter].width = min(max(10, width + 2), 48)


def _sheet(wb, title, header, rows):
    ws = wb.create_sheet(re.sub(r"[\[\]:*?/\\]", "-", title)[:31])
    ws.append(header)
    for r in rows:
        ws.append(r)
    _style(ws)
    return ws


def _save(wb, name, filename):
    EXPORTS.mkdir(parents=True, exist_ok=True)
    path = EXPORTS / name
    wb.save(path)
    return {"path": str(path), "filename": filename}


def _save_csv(header, rows, name, filename):
    EXPORTS.mkdir(parents=True, exist_ok=True)
    path = EXPORTS / name
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return {"path": str(path), "filename": filename, "type": "text/csv"}


# ---------- templates ----------

TEMPLATES = {
    "wells": (["well_id", "name", "role", "lift_type", "field", "status", "notes"],
              [["P-01", "Producer 1", "producer", "esp", "North", "active", ""], ["I-01", "Injector 1", "injector", "", "North", "active", ""]]),
}


def template(dataset, fmt="xlsx"):
    if dataset in TEMPLATES:
        header, example = TEMPLATES[dataset]
        if fmt == "csv":
            return _save_csv(header, example, f"template_{dataset}.csv", f"{dataset} template.csv")
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Data"
        ws.append(header)
        for r in example:
            ws.append(r)
        _style(ws)
        return _save(wb, f"template_{dataset}.xlsx", f"{dataset} template.xlsx")
    if dataset not in DATASETS:
        raise UserError("Unknown dataset.", 404)
    spec = DATASETS[dataset]
    headers = [_header(c) for c in spec["columns"]]
    if fmt == "csv":
        return _save_csv(headers, [], f"template_{dataset}.csv", f"{spec['label']} template.csv")
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(headers)
    _style(ws)
    for i, col in enumerate(spec["columns"], start=1):
        fmt_code = "yyyy-mm-dd" if col["kind"] == "date" else "yyyy-mm-dd hh:mm:ss" if col["kind"] == "time" else None
        if fmt_code:
            letter = ws.cell(row=1, column=i).column_letter
            for row in range(2, 2002):
                ws[f"{letter}{row}"].number_format = fmt_code
    info = wb.create_sheet("Instructions")
    info.append(["Column", "Unit", "Required", "Notes"])
    for col in spec["columns"]:
        notes = {"date": "Calendar date. One row per day.", "time": "UTC date and time. Times without an offset are read as UTC.",
                 "choice": "One of: " + ", ".join(col.get("options", []))}.get(col["kind"], "")
        limits = ", ".join(filter(None, [f"min {col['min']:g}" if "min" in col else "", f"max {col['max']:g}" if "max" in col else ""]))
        info.append([_header(col), col.get("unit", ""), "Yes" if col["required"] else "No", " ".join(filter(None, [notes, limits]))])
    info.append([])
    info.append([spec["label"], "", "", spec["description"]])
    info.append(["CSV", "", "", "The same headers work in a .csv file (comma, semicolon or tab separated)."])
    _style(info)
    return _save(wb, f"template_{dataset}.xlsx", f"{spec['label']} template.xlsx")


# ---------- per-well dataset import / export ----------

def import_file(db, well_id, dataset, path, filename=""):
    import pandas as pd
    well = store.get_well(db, well_id)
    spec = store.dataset_for(well, dataset)
    frame = intake.read_table(path, filename)
    lookup = {}
    for col in spec["columns"]:
        for alias in (col["key"], col["label"], _header(col)):
            lookup[intake._norm(alias)] = col["key"]
    mapping, unknown = {}, []
    for header in frame.columns:
        key = lookup.get(intake._norm(header))
        if key and key not in mapping.values():
            mapping[header] = key
        elif not str(header).startswith("Unnamed"):
            unknown.append(str(header))
    missing = [_header(c) for c in spec["columns"] if c["required"] and c["key"] not in mapping.values()]
    if missing:
        raise UserError("Missing required columns: " + ", ".join(missing) + ". Download the template to see the expected headers.")
    frame = frame.rename(columns=mapping)[list(mapping.values())]
    for col in spec["columns"]:
        if col["kind"] in ("date", "time") and col["key"] in frame:
            parsed = pd.to_datetime(frame[col["key"]], errors="coerce", utc=True, format="mixed")
            frame[col["key"]] = [p.to_pydatetime() if not pd.isna(p) else raw for p, raw in zip(parsed, frame[col["key"]])]
    records = []
    for line, row in zip(frame.index, frame.to_dict("records")):
        row["_line"] = int(line) + 2
        records.append(row)
    result = store.upsert_rows(db, well_id, dataset, records, source=f"{Path(filename).name or 'upload'}")
    result["ignored_columns"] = unknown
    return result


def export_dataset(db, well_id, dataset, fmt="csv"):
    well = store.get_well(db, well_id)
    spec = store.dataset_for(well, dataset)
    header = [_header(c) for c in spec["columns"]]
    rows = [[r.get(c["key"]) for c in spec["columns"]] for r in store.read_rows(db, well_id, dataset)]
    base = f"{store.safe_name(well['name'])} {spec['label']}"
    if fmt == "csv":
        return _save_csv(header, rows, f"data_{well_id}_{dataset}.csv", base + ".csv")
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    _sheet(wb, spec["label"], header, rows)
    return _save(wb, f"data_{well_id}_{dataset}.xlsx", base + ".xlsx")


# ---------- run exports ----------

def run_tables(run):
    """The tables that describe a run's outputs, as (title, header, rows)."""
    o = run.get("outputs") or {}
    kind = run["kind"]
    tables = []
    if kind == "crm_match" and o:
        prod, inj = o["producers"], o["injectors"]
        # Same sheets as pywaterflood CRM.to_excel: Gains, Taus, Primary production.
        tables.append(("Gains", ["Producer / injector", *inj], [[pr, *o["gains"][j]] for j, pr in enumerate(prod)]))
        if o["tau_selection"] == "per-pair":
            tables.append(("Taus", ["Producer / injector", *inj], [[pr, *o["tau"][j]] for j, pr in enumerate(prod)]))
        else:
            tables.append(("Taus", ["Producer", "Tau"], [[pr, o["tau"][j]] for j, pr in enumerate(prod)]))
        tables.append(("Primary production", ["Producer", "Producer gains", "Producer taus"],
                       [[pr, o["gains_producer"][j], o["tau_producer"][j]] for j, pr in enumerate(prod)]))
        tables.append(("Injector sums", ["Injector", "Sum of gains (<= 1 for up-to one)"], [[i, o["injector_sums"][k]] for k, i in enumerate(inj)]))
        rmse_rows = [[pr, o["rmse_fit"][j], o["rmse_holdout"][j] if o["rmse_holdout"] else None] for j, pr in enumerate(prod)]
        rmse_rows.append(["Overall", o["rmse_fit_overall"], o["rmse_holdout_overall"]])
        tables.append(("RMSE", ["Producer", "RMSE (fit)", "RMSE (held out)"], rmse_rows))
        if o.get("productivity_index"):
            tables.append(("Productivity index", ["Producer", "J"], [[pr, o["productivity_index"][j]] for j, pr in enumerate(prod)]))
        tables.append(("History match", ["Time", *[f"{pr} observed" for pr in prod], *[f"{pr} CRM" for pr in prod], "Period"],
                       [[t, *o["observed"][k], *o["predicted"][k], "fit" if k < o["train_rows"] else "held out"] for k, t in enumerate(o["time"])]))
        if o.get("loss_history"):
            keys = list(o["loss_history"][0])
            tables.append(("Loss history", ["Step", *keys], [[k, *[h.get(x) for x in keys]] for k, h in enumerate(o["loss_history"])]))
    elif kind == "crm_forecast" and o:
        tables.append(("Plan", ["Time", *o["producers"]], [[t, *o["plan"][k]] for k, t in enumerate(o["time"])]))
        tables.append(("Injection plan", ["Time", *o["injectors"]], [[t, *o["injection"][k]] for k, t in enumerate(o["time"])]))
    elif kind == "optimize" and o:
        tables.append(("Recommendation", ["Well", "Lift", "Current setpoint", "Recommended setpoint", "Unit", "Current oil (m³/d)",
                                          "Predicted oil (m³/d)", "Change (m³/d)", "Liquid (m³/d)", "Lift gas (sm³/d)", "Power (kW)",
                                          "Risk", "Envelope rule", "Curve source"],
                       [[w["well"], w["lift"], w["current"], w["recommended"], w["unit"], w["current_oil"], w["oil"], w["delta_oil"],
                         w["liquid"], w["lift_gas"], w["power"], w["risk"], w["policy_applied"], w["source"]["label"]] for w in o.get("wells", []) if "recommended" in w]))
        tables.append(("Constraints", ["Constraint", "Unit", "Capacity", "Used", "Slack", "Binding", "Minimum needed"],
                       [[c["name"], c["unit"], c["capacity"], c.get("used"), c.get("slack"), c.get("binding"), c.get("minimum_needed")] for c in o.get("constraints", [])]))
        tables.append(("Not optimised", ["Well", "Reason"], [[e["well"], e["reason"]] for e in o.get("excluded", [])]))
    elif kind == "well_curve" and o:
        keys = [k for k in ("setpoint", "valve", "liquid_m3d", "oil_m3d", "water_m3d", "gas_sm3d", "bhp_bar", "whp_bar", "head_m",
                            "power_kw", "stable", "note") if any(k in p for p in o["points"])]
        tables.append(("Operating points", keys, [[p.get(k) for k in keys] for p in o["points"]]))
        nodal = o.get("nodal")
        if nodal:
            heads = [f"Liquid rate ({nodal['rate_unit']})", f"IPR ({nodal['pressure_unit']})"] +                     [f"VLP {nodal['curve_label'].lower()} {c['setpoint']:,.6g} {nodal['curve_unit']}" for c in nodal["curves"]]
            tables.append(("Nodal curves", heads, [[ip["q"], ip["p"]] + [c["points"][k]["p"] for c in nodal["curves"]]
                                                  for k, ip in enumerate(nodal["ipr"])]))
    elif kind == "vfm" and o and o.get("lift") == "gas_lift":
        tables.append(("Virtual flow meter", ["Timestamp (UTC)", "Estimated liquid (m³/h)", "Well test (m³/h)", "Lift gas (sm³/d)",
                                              "Estimated BHP (bar a)", "Measured BHP (bar a)", "State from"],
                       [[r["timestamp"], r["vfm_m3h"], r["test_m3h"], r["lift_gas_sm3d"], r["bhp_bar"], r["bhp_measured_bar"], r["mode"]]
                        for r in o["rows"]]))
    elif kind == "vfm" and o:
        tables.append(("Virtual flow meter", ["Timestamp (UTC)", "Estimated liquid (m³/h)", "Well test (m³/h)"],
                       [[r["timestamp"], r["vfm_m3h"], r["test_m3h"]] for r in o["rows"]]))
    elif kind in ("esp_risk", "valve_risk") and o:
        key = "date" if kind == "esp_risk" else "timestamp"
        extra = ["Outside training range"] if kind == "esp_risk" else ["Casing heading"]
        tables.append(("Risk", ["Time", "Risk score", "Status", *extra],
                       [[r[key], r["risk"], r["status"], r.get("outside_training_range") if kind == "esp_risk" else r.get("casing_heading")] for r in o["rows"]]))
    return tables


def export_run(db, run, fmt="xlsx"):
    tables = run_tables(run)
    if not tables:
        raise UserError("This run has no outputs to export.")
    name = store.safe_name(run["name"])
    if fmt == "csv":
        title, header, rows = tables[0]
        if len(tables) > 1:  # several tables: one CSV with a table column
            header = ["Table", *max((t[1] for t in tables), key=len)]
            rows = [[t[0], *r] for t in tables for r in [t[1]] + t[2]]
        return _save_csv(header, rows, f"run_{run['id']}.csv", f"{name}.csv")
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Run"
    for k, v in [("Run", run["name"]), ("Kind", run["kind_label"]), ("Field", run["field"] or "—"), ("Created", run["created_at"]),
                 ("Finished", run.get("finished_at")), ("Parameters", json.dumps(run["params"])),
                 ("Pinned inputs", json.dumps(run["inputs"].get("datasets", {}))), ("Models", json.dumps(run["inputs"].get("models", {}))),
                 ("Needs review", "; ".join(run.get("stale", [])) or "no")]:
        ws.append([k, v])
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 18, 90
    for title, header, rows in tables:
        _sheet(wb, title, header, rows)
    return _save(wb, f"run_{run['id']}.xlsx", f"{name}.xlsx")


def export_well(db, well_id):
    from openpyxl import Workbook
    well = store.well_detail(db, well_id)
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    for line in [["Well ID", well["code"]], ["Well", well["name"]], ["Type", WELL_TYPES[well["type"]]["label"]],
                 ["Field", well["field"]], ["Status", well["status"]], ["Notes", well["notes"]]] + \
                [[k.replace("_", " "), v] for k, v in well["settings"].items()]:
        ws.append(line)
    ws.append([])
    ws.append(["Result", "Latest value", "Status", "At", "Run at"])
    for kind, result in well["results"].items():
        s = result.get("summary", {})
        ws.append([s.get("label", kind), s.get("value"), s.get("status", ""), s.get("at", ""), result.get("created_at")])
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 26, 40
    for kind, result in well["results"].items():
        for title, header, rows in run_tables(dict(kind=kind, outputs=result)):
            _sheet(wb, title, header, rows)
    for dataset in WELL_TYPES[well["type"]]["datasets"]:
        spec = DATASETS[dataset]
        _sheet(wb, spec["label"], [_header(c) for c in spec["columns"]],
               [[r.get(c["key"]) for c in spec["columns"]] for r in store.read_rows(db, well_id, dataset)])
    return _save(wb, f"well_{well_id}.xlsx", f"{store.safe_name(well['name'])} results.xlsx")


def export_field(db, field=None):
    from openpyxl import Workbook
    wells = store.list_wells(db, field or None)
    wb = Workbook()
    ws = wb.active
    ws.title = "Wells"
    ws.append(["Well ID", "Well", "Type", "Field", "Status", *[DATASETS[d]["label"] + " rows" for d in DATASETS], "Latest results"])
    for w in wells:
        ws.append([w["code"], w["name"], WELL_TYPES[w["type"]]["label"], w["field"], w["status"],
                   *[w["counts"].get(d, {}).get("rows", 0) if d in WELL_TYPES[w["type"]]["datasets"] else None for d in DATASETS],
                   "; ".join(f"{v.get('label', k)}: {v.get('status') or v.get('value')}" for k, v in w["results"].items())])
    _style(ws)
    _sheet(wb, "Datasets", ["Well", "Dataset", "Version", "Rows", "Updated", "Source"],
           [[d["name"], d["label"], d["version"], d["rows"], d["created_at"], d["source"]] for d in store.dataset_registry(db, field or None)])
    return _save(wb, "field_results.xlsx", f"{field or 'All fields'} report.xlsx")
