import json
import sys
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from common import RUN

def solve(wells, gas, energy_weight, emissions_weight):
    rows = [(i,p) for i,w in enumerate(wells) for p in w["points"]]
    matrix = [[float(i == j) for i,p in rows] for j in range(len(wells))]
    matrix.append([p["gas"] for i,p in rows])
    objective = [energy_weight*p["energy"] + emissions_weight*p["emissions"] - p["oil"] for i,p in rows]
    bounds = LinearConstraint(matrix,[1]*len(wells)+[-np.inf],[1]*len(wells)+[gas])
    result = milp(objective,integrality=np.ones(len(rows)),bounds=Bounds(0,1),constraints=bounds)
    if result.status == 2:
        return {"feasible":False,"message":"The gas budget is below the minimum for the available operating points."}
    if not result.success:
        raise RuntimeError(result.message)
    selected = [dict(well=wells[i]["well"],**point) for chosen,(i,point) in zip(result.x,rows) if chosen>.5]
    return {"feasible":True,"wells":selected,"totals":{k:sum(p[k] for p in selected) for k in ("oil","gas","energy","emissions")}}

if __name__ == "__main__":
    print(json.dumps(solve(json.loads((RUN/"curves.json").read_text()), *map(float,sys.argv[1:])),allow_nan=False))

