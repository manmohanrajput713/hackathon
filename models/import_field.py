"""Build a matched historical well replay from client inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import save
from esp import PinnPredictor

def build(folder):
    from pywaterflood import CRM
    cfg = json.loads((folder / "field.json").read_text())
    wells, injectors = cfg["producers"], cfg["injectors"]
    if not wells or not injectors or len(set(wells+injectors)) != len(wells+injectors):
        raise ValueError("Supply distinct producer and injector column names")
    hist = pd.read_csv(folder / "reservoir.csv")
    hist["timestamp"] = pd.to_datetime(hist.timestamp,utc=True,errors="raise")
    hist = hist.sort_values("timestamp")
    if hist.timestamp.duplicated().any():
        raise ValueError("Repeated reservoir timestamp")
    times = (hist.timestamp-hist.timestamp.iloc[0]).dt.total_seconds().to_numpy()/86400
    q = hist[wells].to_numpy(float)
    injection = hist[injectors].to_numpy(float)
    if not np.isfinite(np.column_stack([q,injection])).all() or (q<0).any() or (injection<0).any():
        raise ValueError("Reservoir rates must be finite and nonnegative")
    cutoff = pd.Timestamp(cfg["training_end"])
    if cutoff.tzinfo is None:
        raise ValueError("training_end needs a timezone")
    n = int((hist.timestamp<=cutoff).sum())
    if n<2 or n>=len(hist):
        raise ValueError("Need at least two training observations and a later replay period")
    model = CRM(primary=True,tau_selection="per-producer",constraints="up-to one")
    model.fit(q[:n],injection[:n],times[:n],epochs=250,seed=0)
    forecast = model.predict(production=q[:1],injection=injection,time=times)
    telemetry = pd.read_csv(folder/"telemetry.csv",dtype={"well_id":str})
    telemetry["timestamp"] = pd.to_datetime(telemetry.timestamp,utc=True,errors="raise")
    if telemetry.duplicated(["well_id","timestamp"]).any() or set(telemetry.well_id)-set(wells):
        raise ValueError("Check telemetry well IDs and duplicate timestamps")
    reports = []
    for wid,observations in telemetry.groupby("well_id"):
        t = observations.sort_values("timestamp").reset_index(drop=True)
        m = cfg["esp"][wid]
        predictor = PinnPredictor(str(folder / m["checkpoint"]))
        p = predictor.parameters
        u = t[["frequency_hz","choke_percent","manifold_pa","reservoir_pa"]].to_numpy(float)
        pressures=t[["bottomhole_pa","wellhead_pa"]].to_numpy(float)
        inputs=np.column_stack([p.normalizar_u(u),p.normalizar_x(pressures)])
        # Accumulate only consecutive observations on the model's training cadence.
        width=predictor.model.n_steps_in
        continuous=(t.timestamp.diff().dt.total_seconds()==float(m["sample_seconds"]))
        valid=(continuous.rolling(width).sum()==width) & (t.timestamp>cutoff)
        idx=[i for i in np.flatnonzero(valid) if np.isfinite(inputs[i-width:i]).all()]
        if not idx:
            raise ValueError(f"No complete post-training telemetry windows for {wid}")
        windows=np.stack([inputs[i-width:i] for i in idx])
        y=np.concatenate([predictor.model.predict(batch)[:,0,2] for batch in np.array_split(windows, max(1,int(np.ceil(len(windows)/256))))])
        case=pd.DataFrame({"timestamp":t.timestamp.iloc[idx].to_numpy(),"vfm":(y*p.qc+p.qmin)*3600,
                           "test":t.test_m3h.iloc[idx].to_numpy()})
        # Use the most recent CRM value within the configured freshness interval.
        crm=pd.DataFrame({"crm_at":hist.timestamp.iloc[n:],"crm":forecast[n:,wells.index(wid)]})
        joined=pd.merge_asof(case,crm,left_on="timestamp",right_on="crm_at",direction="backward",
                             tolerance=pd.Timedelta(seconds=float(cfg["crm_max_age_seconds"])))
        records=json.loads(joined.to_json(orient="records",date_format="iso"))
        reports.append(dict(name=wid,records=records))
    if not reports:
        raise ValueError("No matching ESP wells in telemetry.csv")
    return reports

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    save("field.json", build(parser.parse_args().folder))

