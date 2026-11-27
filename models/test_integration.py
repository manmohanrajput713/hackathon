import itertools
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, RUN
from allocate import solve
from import_field import build

class IntegrationTests(unittest.TestCase):
    def test_allocator_matches_enumerated_choices(self):
        wells = json.loads((RUN / "curves.json").read_text())
        cap = 1.8
        feasible = [combo for combo in itertools.product(*(w["points"] for w in wells))
                    if sum(p["gas"] for p in combo) <= cap]
        objective = lambda row: row["oil"]-.05*row["energy"]-.1*row["emissions"]
        actual = solve(wells,cap,.05,.1)
        self.assertAlmostEqual(objective(actual["totals"]),max(sum(map(objective,c)) for c in feasible))
        self.assertFalse(solve(wells,0,0,0)["feasible"])

    def test_matched_well_uses_prior_forecast_and_preserves_missing_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            start = pd.Timestamp("2025-01-01",tz="UTC")
            reservoir_clock = pd.date_range(start, periods=8, freq="20s")
            pd.DataFrame({"timestamp":reservoir_clock,"P1":np.arange(8)+22,"I1":np.arange(8)+25}).to_csv(folder/"reservoir.csv",index=False)
            clock=pd.date_range(start,periods=170,freq="s")
            with np.load(ROOT/"pinn/dataset/BCS_data_train_limitado_f_zc_pm_pr_opera.npz") as inputs:
                data=pd.DataFrame(inputs["U"][:len(clock)],columns=["frequency_hz","choke_percent","manifold_pa","reservoir_pa"])
                data["bottomhole_pa"]=inputs["x1"][:len(clock)].ravel()
                data["wellhead_pa"]=inputs["x2"][:len(clock)].ravel()
            data["well_id"]="P1"
            data["timestamp"]=clock
            data["test_m3h"]=np.nan
            data.loc[130,"test_m3h"]=26
            data.drop(index=100).to_csv(folder/"telemetry.csv",index=False)
            cfg={"producers":["P1"],"injectors":["I1"],"training_end":reservoir_clock[3].isoformat(),
                 "crm_max_age_seconds":5,"esp":{"P1":{"checkpoint":str(ROOT/"pinn/model_adam_lbfgs/model.pt"),"sample_seconds":1}}}
            (folder/"field.json").write_text(json.dumps(cfg))
            records=build(folder)[0]["records"]
            self.assertTrue(any(r["crm"] is not None for r in records))
            self.assertTrue(any(r["crm"] is None for r in records))
            self.assertEqual([r["test"] for r in records if r["test"] is not None],[26])
            for r in records:
                if r["crm_at"]:
                    self.assertLessEqual(pd.Timestamp(r["crm_at"]),pd.Timestamp(r["timestamp"]))
                self.assertFalse(clock[100]<=pd.Timestamp(r["timestamp"])<=clock[120])
if __name__=="__main__":
    unittest.main()

