"""Wrappers that run the project models on registry data. Orchestration lives in runs.py."""
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from common import ROOT, RUN
import store

MODELS = RUN / "models"
RESULT_ROW_CAP = 20000

MODEL_FILES = {
    "esp_risk": ["failure_1/results/table5_best_hyperparameters.json", "failure_1/dataset/processed_esp_data.parquet"],
    "vfm": ["pinn/model_adam_lbfgs/model.pt"],
    "gas_lift_vfm": ["gas_pinc/runs/well1/algebraic/model.pt", "gas_pinc/runs/well1/lbfgs/model.pt", "gas_pinc/data/well_ode.py",
                     "field-console/models/gaslift.py"],
    "valve_risk": ["failure_2/results/synthetic/model_bundle.joblib"],
    "esp_curve": ["pinn/data/param.py", "field-console/models/wellmodels.py"],
    "gas_lift_curve": ["gas_pinc/data/param.py", "gas_pinc/data/well_ode.py", "field-console/models/gaslift.py"],
    "crm": ["pywaterflood/python/pywaterflood/crm.py"],
}


def model_version(name):
    """A stamp that changes when a model's files change."""
    parts = []
    for rel in MODEL_FILES[name]:
        p = ROOT / rel
        parts.append(f"{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}" if p.exists() else f"{p.name}:missing")
    return "|".join(parts)


def _finite(value):
    return None if value is None or (isinstance(value, float) and not math.isfinite(value)) else value


# ---------- ESP failure risk (failure_1) ----------

def _esp_risk_bundle():
    """Rebuild the published ESP classifier once and cache it with its exact scaling and projection."""
    import joblib
    path = MODELS / "esp_risk.joblib"
    source = ROOT / "failure_1"
    card_path = source / "results/table5_best_hyperparameters.json"
    data_path = source / "dataset/processed_esp_data.parquet"
    stamp = [card_path.stat().st_mtime_ns, data_path.stat().st_mtime_ns]
    if path.exists():
        bundle = joblib.load(path)
        if bundle.get("stamp") == stamp:
            return bundle
    import pandas as pd
    sys.path.insert(0, str(source))
    from src.model import fit_final, PC_COLUMNS
    from src.preprocessing import SENSOR_COLUMNS
    from src.features import FEATURE_COLUMNS
    card = json.loads(card_path.read_text())
    frame = pd.read_parquet(data_path)
    mean, std = [], []
    for col in SENSOR_COLUMNS:  # recover the training standardizer from stored raw/scaled pairs
        slope, intercept = np.polyfit(frame[col].to_numpy(float), frame[f"raw_{col}"].to_numpy(float), 1)
        std.append(slope)
        mean.append(intercept)
    design = np.column_stack([frame[FEATURE_COLUMNS].to_numpy(float), np.ones(len(frame))])
    projection, *_ = np.linalg.lstsq(design, frame[PC_COLUMNS].to_numpy(float), rcond=None)
    error = np.abs(design @ projection - frame[PC_COLUMNS].to_numpy(float)).max()
    if error > 1e-6:
        raise RuntimeError(f"Could not reproduce the ESP feature projection (max error {error:.2e})")
    train = frame[frame.split == "train"]
    fixed = card["fixed_parameters"]
    model, _ = fit_final(train[PC_COLUMNS], train.label, card["best_hyperparameters"], fixed["n_estimators"])
    raw_train = train[[f"raw_{c}" for c in SENSOR_COLUMNS]].to_numpy(float)
    bundle = dict(stamp=stamp, model=model, mean=np.array(mean), std=np.array(std), projection=projection,
                  threshold=float(fixed["decision_threshold"]), sensors=SENSOR_COLUMNS,
                  p01=np.percentile(raw_train, 1, axis=0), p99=np.percentile(raw_train, 99, axis=0))
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


ESP_COLUMNS = ["frequency_hz", "discharge_psi", "intake_psi", "pump_avg_psi", "motor_temp_f", "esp_temp_f", "current_a", "voltage_v"]
ESP_LABELS = ["Frequency", "Discharge pressure", "Intake pressure", "Pump avg pressure", "Motor temperature", "ESP temperature",
              "Drive current", "Drive voltage"]


def esp_risk(rows):
    if len(rows) < 2:
        return {"skipped": "Add at least two consecutive days of ESP daily sensor data."}
    import pandas as pd
    b = _esp_risk_bundle()
    frame = pd.DataFrame(rows)
    frame["day"] = pd.to_datetime(frame.date)
    raw = frame[ESP_COLUMNS].to_numpy(float)
    scaled = (raw - b["mean"]) / b["std"]
    diffs = np.vstack([np.full(len(ESP_COLUMNS), np.nan), np.diff(scaled, axis=0)])
    consecutive = frame.day.diff().dt.days.eq(1).to_numpy()
    features = np.column_stack([scaled, diffs, np.ones(len(frame))])
    ok = consecutive & np.isfinite(features).all(axis=1)
    scores = np.full(len(frame), np.nan)
    if ok.any():
        scores[ok] = b["model"].predict_proba(features[ok] @ b["projection"])[:, 1]
    outside = ((raw < b["p01"]) | (raw > b["p99"])).any(axis=1)
    out = [dict(date=d, risk=_finite(float(s)), status=("unavailable" if not math.isfinite(s) else
                "alert" if s >= b["threshold"] else "no_alert"), outside_training_range=bool(o))
           for d, s, o in zip(frame.date, scores, outside)]
    scored = [r for r in out if r["risk"] is not None]
    if not scored:
        return {"skipped": "No two consecutive calendar days were found. Risk needs today's and yesterday's readings."}
    # Evidence: on alert days, the sensors furthest from the training fleet (level z-score) and with the largest one-day change.
    evidence = []
    for i, r in enumerate(out):
        if r["status"] == "alert":
            order = np.argsort(-np.abs(scaled[i]))[:3]
            moves = np.argsort(-np.abs(np.nan_to_num(diffs[i])))[:3]
            evidence.append(dict(date=r["date"], risk=r["risk"],
                                 levels=[dict(sensor=ESP_LABELS[j], value=float(raw[i, j]), z=float(scaled[i, j])) for j in order],
                                 changes=[dict(sensor=ESP_LABELS[j], change_sd=float(diffs[i, j])) for j in moves]))
    latest = scored[-1]
    share = float(outside.mean())
    warnings = ["This classifier was trained on the 2023 Energy AI hackathon ESP fleet. Its holdout precision for "
                "the 7-day warning is about 5%, so most alerts are false positives. Use it to prioritise inspections."]
    if share > 0.2:
        warnings.append(f"{share:.0%} of days have a sensor outside the 1st–99th percentile of the training wells. "
                        "Treat the scores as indicative.")
    return {"threshold": b["threshold"], "horizon_days": 7, "rows": out[-RESULT_ROW_CAP:], "evidence": evidence[-100:],
            "warnings": warnings,
            "summary": dict(label="ESP failure risk", status=latest["status"], value=latest["risk"],
                            at=latest["date"], alerts=sum(r["status"] == "alert" for r in scored[-30:]))}


# ---------- ESP virtual flow meter (pinn) ----------

def esp_vfm(rows, settings):
    if not rows:
        return {"skipped": "Add ESP high-frequency telemetry to estimate flow."}
    import pandas as pd
    from esp import PinnPredictor
    predictor = PinnPredictor(str(ROOT / "pinn/model_adam_lbfgs/model.pt"))
    p = predictor.parameters
    t = pd.DataFrame(rows)
    t["ts"] = pd.to_datetime(t.timestamp, utc=True)
    u = t[["frequency_hz", "choke_percent", "manifold_bar", "reservoir_bar"]].to_numpy(float)
    u[:, 2:] *= 1e5
    pressures = t[["bottomhole_bar", "wellhead_bar"]].to_numpy(float) * 1e5
    inputs = np.column_stack([p.normalizar_u(u), p.normalizar_x(pressures)])
    width = predictor.model.n_steps_in
    step = float(settings.get("vfm_sample_seconds", 1))
    continuous = (t.ts.diff().dt.total_seconds() - step).abs().lt(1e-3 * max(step, 1))
    valid = continuous.rolling(width).sum().eq(width).to_numpy()
    valid &= np.array([i >= width and np.isfinite(inputs[i - width:i]).all() for i in range(len(t))])
    # Coverage timeline: runs of samples where a full input window is (or is not) available.
    segments, start = [], 0
    for i in range(1, len(t) + 1):
        if i == len(t) or valid[i] != valid[start]:
            segments.append(dict(start=t.timestamp.iloc[start], end=t.timestamp.iloc[i - 1], valid=bool(valid[start]), samples=i - start))
            start = i
    channels = {c: round(float(t[c].notna().mean()), 4) for c in ["frequency_hz", "choke_percent", "manifold_bar", "reservoir_bar",
                                                                   "bottomhole_bar", "wellhead_bar", "test_m3h"]}
    spacing = t.ts.diff().dt.total_seconds().dropna()
    coverage = dict(samples=len(t), valid_windows=int(valid.sum()), window=width, step_s=step, segments=segments[-500:],
                    channels=channels, spacing_s=dict(median=float(spacing.median()) if len(spacing) else None,
                                                      off_cadence=int((~continuous.iloc[1:]).sum())))
    idx = list(np.flatnonzero(valid))
    if not idx:
        return {"skipped": f"No run of {width + 1} consecutive samples spaced {step:g} s apart was found. "
                           "Check the well's VFM sampling interval.", "coverage": coverage}
    idx = idx[-RESULT_ROW_CAP:]
    windows = np.stack([inputs[i - width:i] for i in idx])
    batches = np.array_split(windows, max(1, math.ceil(len(windows) / 512)))
    y = np.concatenate([predictor.model.predict(batch)[:, 0, 2] for batch in batches])
    rate = (y * p.qc + p.qmin) * 3600
    test = t["test_m3h"] if "test_m3h" in t else pd.Series([None] * len(t))
    out = [dict(timestamp=t.timestamp.iloc[i], vfm_m3h=round(float(r), 4), test_m3h=_finite(test.iloc[i])) for i, r in zip(idx, rate)]
    warnings = []
    for label, values, lo, hi, unit, scale in [("Pump frequency", u[:, 0], 30, 75, "Hz", 1), ("Manifold pressure", u[:, 2], 12e5, 21e5, "bar", 1e5),
                                               ("Reservoir pressure", u[:, 3], 120e5, 140e5, "bar", 1e5)]:
        share = float(((values < lo) | (values > hi)).mean())
        if share > 0.05:
            warnings.append(f"{label} is outside the checkpoint's {lo/scale:g}–{hi/scale:g} {unit} training range in {share:.0%} of samples.")
    if coverage["valid_windows"] < 0.5 * len(t):
        warnings.append(f"Only {coverage['valid_windows']} of {len(t)} samples have a complete {width}-sample history. "
                        "Gaps or off-cadence samples block inference.")
    pairs = [(r["vfm_m3h"], r["test_m3h"]) for r in out if r["test_m3h"] is not None]
    metrics = {}
    if pairs:
        err = np.array([a - b for a, b in pairs])
        metrics = dict(tests=len(pairs), mean_error_m3h=float(err.mean()), mean_abs_error_m3h=float(np.abs(err).mean()))
    latest = out[-1]
    quality = "good" if not warnings else "check"
    return {"rows": out, "warnings": warnings, "metrics": metrics, "coverage": coverage,
            "summary": dict(label="Estimated liquid rate", value=latest["vfm_m3h"], unit="m³/h", at=latest["timestamp"], quality=quality)}


# ---------- Gas-lift virtual flow meter (gas_pinc) ----------

def gas_lift_vfm(rows, settings):
    import gaslift
    return gaslift.vfm(rows, settings)


# ---------- Gas-lift valve risk (failure_2) ----------

FAILURE2_EVENTS = {"installation", "replacement", "planned_replacement", "valve_leak", "valve_malfunction", "tubing_leak",
                   "casing_heading", "coverage_end"}


def valve_risk(well, rows, events):
    if len(rows) < 2:
        return {"skipped": "Add at least two gas-lift pressure readings."}
    source = ROOT / "failure_2"
    bundle = source / "results/synthetic/model_bundle.joblib"
    with tempfile.TemporaryDirectory(dir=RUN) as temp:
        temp = Path(temp)
        with (temp / "telemetry.csv").open("w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["well_id", "timestamp", "casing_pressure", "tubing_head_pressure", "lift_gas_rate", "choke_position"])
            for r in rows:
                w.writerow([well["id"], r["timestamp"], r["casing_kpa"], r["tubing_head_kpa"],
                            "" if r.get("lift_gas_sm3d") is None else r["lift_gas_sm3d"],
                            "" if r.get("choke_percent") is None else r["choke_percent"]])
        with (temp / "maintenance.csv").open("w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["well_id", "event_type", "timestamp", "event_id", "confirmed", "detected_at"])
            for e in events:
                kind = e["event_type"] if e["event_type"] in FAILURE2_EVENTS else "other"
                fault = kind in ("valve_leak", "valve_malfunction")
                w.writerow([well["id"], kind, e["timestamp"], e.get("event_id") or "", e.get("confirmed") or "",
                            e["timestamp"] if fault else ""])
        run = subprocess.run([sys.executable, "main.py", "score", "--telemetry", str(temp / "telemetry.csv"),
                              "--maintenance", str(temp / "maintenance.csv"), "--bundle", str(bundle),
                              "--output", str(temp / "scores.csv"), "--heading-output", str(temp / "heading.csv")],
                             cwd=source, capture_output=True, text=True, timeout=240)
        if run.returncode != 0:
            print(run.stderr[-2000:], file=sys.stderr)
            tail = run.stderr.strip().splitlines()[-1:] or ["unknown error"]
            return {"error": f"Valve model could not score this data: {tail[0]}"}
        with (temp / "scores.csv").open(encoding="utf8") as f:
            scores = list(csv.DictReader(f))
        heading = {}
        if (temp / "heading.csv").exists():
            with (temp / "heading.csv").open(encoding="utf8") as f:
                for r in csv.DictReader(f):
                    heading[r["timestamp"]] = r.get("casing_heading_status")
    meta_path = source / "results/synthetic/model_bundle.json"
    threshold = json.loads(meta_path.read_text()).get("alert_threshold") if meta_path.exists() else None
    out = [dict(timestamp=store.parse_time(r["timestamp"], "time")[0], risk=float(r["risk_score"]) if r["risk_score"] else None,
                status=r["alert_status"], casing_heading=heading.get(r["timestamp"])) for r in scores]
    scored = [r for r in out if r["risk"] is not None]
    if not scored:
        return {"skipped": "The valve model needs consecutive pressure readings no more than 36 hours apart."}
    latest = scored[-1]
    heading_now = next((r["casing_heading"] for r in reversed(out) if r["casing_heading"] not in (None, "", "unavailable")), "unavailable")
    return {"threshold": threshold, "horizon_days": 7, "rows": out[-RESULT_ROW_CAP:], "events": events,
            "evidence": [r for r in out if r["status"] == "alert"][-100:], "casing_heading": heading_now,
            "warnings": ["This valve model was trained on the project's synthetic gas-lift fixture. Retrain it on "
                         "field telemetry with confirmed valve events before relying on alerts."],
            "summary": dict(label="Valve failure risk", status=latest["status"], value=latest["risk"], at=latest["timestamp"],
                            alerts=sum(r["status"] == "alert" for r in scored[-50:]), casing_heading=heading_now)}
