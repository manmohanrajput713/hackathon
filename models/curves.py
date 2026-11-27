"""Generate frozen operating points from the gas_pinc numerical plant."""
import sys
import numpy as np
from common import ROOT, save
sys.path.insert(0, str(ROOT / "gas_pinc"))
from data.param import get_well
from data.sample import integrate_haaland
from data.well_ode import algebraics_numpy

def build():
    wells = []
    for number in (1, 2):
        well = get_well(number)
        points = []
        for valve in (.2,.35,.5,.65,.8):
            controls = [.5, valve]
            initial_mass = [3500,300,8000] if number == 1 else [5400,300,9000]
            result = integrate_haaland(initial_mass,controls,well,600,max_step=15)
            if not result["reached"]:
                continue
            pressure, gas, _, liquid = algebraics_numpy(result["y_end"], controls, well)
            # Scenario assumptions: water cut 25%, compressor demand 60 kW/(kg/s),
            # electricity emissions factor 0.4 kg CO2e/kWh.
            oil = liquid / well.rho_L * 86400 * .75
            energy = gas * 60 * 24
            points.append(dict(setting=valve, gas=float(gas),oil=float(oil),energy=float(energy),
                               emissions=float(energy*.4), pressure=float(pressure)))
        if len(points)<2:
            raise ValueError(f"Insufficient feasible operating points for well {number}")
        wells.append(dict(well=f"GL simulation {number}",points=points))
    save("curves.json",wells)

if __name__ == "__main__":
    build()

