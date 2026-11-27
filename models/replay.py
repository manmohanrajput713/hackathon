import csv
import json
from datetime import datetime, timezone
import numpy as np
from common import ROOT, RUN, save

ESP_INPUTS = (ROOT / "pinn/model_adam_lbfgs/model.pt",
              ROOT / "pinn/dataset/BCS_data_train_limitado_f_zc_pm_pr_opera.npz")

def esp_series():
    """Reuse the last ESP inference while its checkpoint and input data are unchanged."""
    key = [[str(p), p.stat().st_mtime_ns, p.stat().st_size] for p in ESP_INPUTS]
    cache = RUN / "esp.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("key") == key:
            return cached["series"]
    from esp import series
    result = series()
    save("esp.json", dict(key=key, series=result))
    return result

def refresh():
    wells = [esp_series()] + json.loads((RUN/"crm.json").read_text())
    with np.load(ROOT / "gas_pinc/runs/well1/mpc.npz") as gas:
        wells.append(dict(name="Gas-lift test well",source="Gas-lift controller simulation",model="Gas-lift MPC",
                          quantity="Bottom-hole pressure",unit="bar",xLabel="Simulation time (s)",
                          points=np.column_stack([gas["pinc_t"],gas["pinc_pbh"],gas["pinc_ref"]]).tolist(),
                          refLabel="Pressure target",predLabel="Simulated pressure"))
    alerts = json.loads((RUN/"pump.json").read_text())
    latest = {}
    with (ROOT / "failure_2/results/synthetic/latest_scores.csv").open() as f:
        for row in csv.DictReader(f):
            if row["well_id"] not in latest or row["timestamp"] > latest[row["well_id"]]["timestamp"]:
                latest[row["well_id"]]=row
    for r in latest.values():
        alerts.append(dict(well=r["well_id"],time=r["timestamp"],score=float(r["risk_score"]) if r["risk_score"] else None,
                           status=r["alert_status"],source="Synthetic valve replay",kind="Valve"))
    curves = json.loads((RUN/"curves.json").read_text())
    minimum = sum(min(p["gas"] for p in w["points"]) for w in curves)
    maximum = sum(max(p["gas"] for p in w["points"]) for w in curves)
    field_file = RUN / "field.json"
    field = json.loads(field_file.read_text()) if field_file.exists() else []
    save("snapshot.json",dict(updated=datetime.now(timezone.utc).isoformat(), wells=wells, alerts=alerts, field=field,
                              budget=dict(min=minimum,max=maximum)))

if __name__ == "__main__":
    refresh()

