# Companion models

CRM needs rate history. The rest of pywaterflood answers questions you have
before that history exists, or after CRM tells you a producer is over- or
under-supported. Use these tools with CRM, not as a replacement for it.

## Multiwell productivity index (geometry)

On a rectangular, roughly homogeneous reservoir in semi-steady state, the
multiwell productivity index (Kaviani and Valkó, 2010) turns well \(X,Y\)
locations into a producer–injector influence matrix \(\Lambda\).

\(\Lambda\) is negative. Positive CRM gains are \(-\Lambda\). That is the
matrix you want as a first guess when CRM has many \(f_{ij}\) to learn.

```python
import pandas as pd
from pywaterflood import (
    calc_gains_homogeneous,
    geometry_crm_gains,
    translate_locations,
)

raw = pd.DataFrame({
    "easting":  [1200, 4800, 3000, 1200, 4800, 3000, 1800, 4200, 3000],
    "northing": [4800, 4800, 3000, 1200, 1200, 4800, 3000, 3000, 1200],
    "kind": [
        "Injector", "Injector", "Injector", "Injector", "Injector",
        "Producer", "Producer", "Producer", "Producer",
    ],
}, index=["I1", "I2", "I3", "I4", "I5", "P1", "P2", "P3", "P4"])

locations = translate_locations(raw, "easting", "northing", "kind")
x_e, y_e = 6000.0, 6000.0

lambda_df = calc_gains_homogeneous(locations, x_e, y_e)
gains = geometry_crm_gains(
    locations,
    x_e,
    y_e,
    producer_ids=["P1", "P2", "P3", "P4"],
    injector_ids=["I1", "I2", "I3", "I4", "I5"],
)
# gains.shape == (4, 5), all entries >= 0
```

`translate_locations` shifts the lower-left well to the origin and renames
columns to `X`, `Y`, `Type`. Rotate the map first if the field axes are not
east–north.

Pass the same `locations`, `x_e`, `y_e`, and IDs into `CRM.fit` with
`initial_guess="geometry"`. That path is in {doc}`crm-in-practice`. The
notebook {doc}`multiwell-productivity-index` plots a 5×4 pattern and used to
splice \(-\Lambda\) into `x0` by hand; the `initial_guess="geometry"` argument
replaces that splice.

### When MPI is enough on its own

Use the geometry matrix without CRM when you are still placing wells, or when
you have no rate history. Treat it as a ranking: which injector should matter
to which producer if the rock were uniform. After rates arrive, let CRM move
the numbers.

## Buckley-Leverett displacement

Buckley-Leverett describes a 1-D waterflood from injector to producer: where
the saturation front is, and how fast it moves. Pywaterflood uses Brooks–Corey
relative permeability.

```python
from pywaterflood.buckleyleverett import Reservoir, breakthrough_sw, water_front_velocity

rock = Reservoir(
    phi=0.25,
    viscosity_oil=4.2e-3,      # Pa·s
    viscosity_water=1e-3,
    sat_oil_r=0.2,
    sat_water_c=0.1,
    sat_gas_c=0.05,
    n_oil=2,
    n_water=2,
    flow_cross_section=1.0,    # m²
)

sw_bt = breakthrough_sw(rock)
v_bt = water_front_velocity(rock, sw_bt, flow_rate=1.0)
```

`water_front_velocity` is the Buckley-Leverett \(\mathrm{d}x/\mathrm{d}t\) at a
given water saturation. Below breakthrough saturation the speed equals the
front speed. Multiply by time to get a saturation profile along the path.

This does not replace CRM. Use it to sanity-check a pair that CRM says is
strongly connected: a very large \(f_{ij}\) with a tiny \(\tau_{ij}\) should
be a short, permeable path, consistent with a fast front.

Worked plots: {doc}`buckley-leverett`.

## Aquifer influx

Most reservoirs sit on an aquifer. As you produce, water encroaches and
supports pressure. van Everdingen and Hurst tabulated that influx; Klins and
Marsal–Walsh approximations make it a function you can fit.

Typical sequence:

1. Effective reservoir radius from pore volume, porosity, thickness, and the
   open angle of the aquifer.
2. Dimensionless time \(t_D\) from aquifer permeability, porosity, viscosity,
   and compressibility.
3. `aquifer_production` from the pressure drop history.

```python
import numpy as np
from pywaterflood.aquifer import aquifer_production, effective_reservoir_radius

r_o = effective_reservoir_radius(
    reservoir_pore_volume=385e6,  # reservoir bbl
    porosity=0.31,
    thickness=20.0,               # ft
    flow_solid_angle=180.0,       # semicircle
)

# Build t_D and pressure drops from your PVT / aquifer table, then:
w_e = aquifer_production(
    delta_pressure,   # pressure drop per step
    t_d,              # dimensionless time
    r_ed=5.0,         # aquifer / reservoir radius
    method="klins",   # or "marsal-walsh"
    aquifer_constant=U,
)
```

If CRM residuals are one-sided (producers systematically above the injection
support), the missing fluid is often this influx. Two practical options:

- Add a **pseudo-injector** whose rate is \(W_e\) (or the field imbalance) and
  refit CRM. Gains to that column are aquifer support.
- Fit `r_ed` and \(U\) with `scipy.optimize.curve_fit` as in {doc}`aquifer`.

`method="klins"` is the default finite-aquifer approximation.
`"marsal-walsh"` is the other built-in table fit.

## Putting the four tools on one field

1. **Map** → MPI geometry gains, or a first well-spacing decision.
2. **Rates** → CRM-IP history match, seeded with geometry when the map exists.
3. **Material-balance gap** → aquifer influx as a pseudo-injector, then CRM
   again.
4. **A pair CRM trusts** → Buckley-Leverett front between those two wells, if
   you have relative-permeability curves, to see whether breakthrough time is
   plausible.

None of these steps requires a full-physics simulator. Each one is cheap
enough to rerun when a new month of rates arrives.
