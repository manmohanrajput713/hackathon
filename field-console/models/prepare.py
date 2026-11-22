"""Fit the local Deep-CRM and reconstruct the ESP classifier offline."""
import json
import sys
import numpy as np
import pandas as pd
from common import ROOT, save

def reservoir():
    from pywaterflood import CRM
    if not hasattr(CRM, "_fit_pinn"):
        raise RuntimeError("Install the local pywaterflood checkout to use Deep-CRM")
    src = ROOT / "pywaterflood/testing/data"
    rates, inj, times = [np.loadtxt(src / (key + ".csv"), delimiter=",") for key in ("production","injection","time")]
    end = int(len(times) * .8)
    model = CRM(primary=True, tau_selection="per-producer", constraints="up-to one")
    model.fit(rates[:end], inj[:end], times[:end], epochs=250, seed=0)
    forecast = model.predict(production=rates[:1], injection=inj, time=times)
    results = []
    for index in range(rates.shape[1]):
        points = np.column_stack([times[end:], forecast[end:,index], rates[end:,index]])
        results.append(dict(name=f"CRM producer {index+1}", model="Deep-CRM", quantity="Producer rate",
                            source="Waterflood test history · held-out final 20%", unit="dataset rate units",
                            xLabel="Dataset time", points=points.round(4).tolist()))
    save("crm.json", results)

def pump_risk():
    sys.path.insert(0, str(ROOT / "failure_1"))
    from src.model import fit_final, PC_COLUMNS
    src = ROOT / "failure_1"
    card = json.loads((src / "results/table5_best_hyperparameters.json").read_text())
    fixed = card["fixed_parameters"]
    frame = pd.read_parquet(src / "dataset/processed_esp_data.parquet")
    train = frame[frame.split == "train"]
    test = frame[frame.split == "test"].copy()
    model, _ = fit_final(train[PC_COLUMNS], train.label, card["best_hyperparameters"], fixed["n_estimators"])
    test["score"] = model.predict_proba(test[PC_COLUMNS].to_numpy())[:,1]
    latest = test.sort_values("day_index").groupby("esp_id").tail(1)
    rows = [dict(well=str(r.esp_id),time=f"Day {r.day_index}",score=float(r.score),
                 status="alert" if r.score >= fixed["decision_threshold"] else "no_alert",
                 source="ESP holdout replay", kind="Pump") for r in latest.itertuples()]
    save("pump.json", rows)

if __name__ == "__main__":
    options = {"reservoir": reservoir, "pump": pump_risk}
    for name in sys.argv[1:] or options:
        options[name]()

