# CRM in practice

This page is the working manual for `CRM` and `CrmCompensated`. It covers data
shape, which options to pick, how history matching is trained, how to seed
gains from well geometry, and how to forecast a new injection plan.

A short end-to-end session is in {doc}`7-minutes-to-pywaterflood`. Model
variants are compared in {doc}`choosing-crm`.

## What the model is doing

CRM is a material-balance model. For producer \(j\), injection support and
(optionally) a pressure change explain the rate \(q_j\):

$$
\tau_j \dot q_j + q_j
= \sum_i f_{ij}\, I_i
- \tau_j J_j \dot p_{\mathrm{wf},j}
$$

In words: the producer’s rate tracks the weighted injection that reaches it,
with a lag \(\tau\). A falling bottomhole pressure adds extra rate; a rising
pressure cuts it. Primary production, when enabled, is an exponential decline
from the first observed rate and is removed before that balance is applied.

`fit()` trains a Deep-CRM PINN so the physical parameters \(f_{ij}\), \(\tau\),
and (if used) \(J_j\) match both the observed rates and that differential
equation. `predict()` then runs the closed-form CRM solution with those
identified numbers. You can change the injection schedule in `predict()`
without training again.

## Data layout

All arrays share the same time axis. Units must be consistent: reservoir
volume per time for rates, the same time unit for `time` and \(\tau\).

| Array | Shape | Meaning |
|---|---|---|
| `production` | `(n_time, n_producers)` | Producer columns in a fixed order |
| `injection` | `(n_time, n_injectors)` | Injector columns in a fixed order |
| `time` | `(n_time,)` | Elapsed time, typically starting near 0 |
| `pressure` | `(n_time, n_producers)` | Producer BHP, same column order as production |

```python
import numpy as np
from pywaterflood import CRM

production = np.genfromtxt("production.csv", delimiter=",")
injection = np.genfromtxt("injection.csv", delimiter=",")
time = np.genfromtxt("time.csv", delimiter=",")

crm = CRM(primary=True, tau_selection="per-pair", constraints="up-to one")
crm.fit(production, injection, time, epochs=500, seed=0)
print(crm.gains.shape)  # (n_producers, n_injectors)
q_hat = crm.predict()
```

### Rules the fitter enforces

- Every series has the same number of steps as `time`.
- Values are finite and non-negative.
- At least two timesteps are required (the convolution reads `time[1]`).
- Column order is the well order. `gains[j, i]` is injector \(i\) supporting
  producer \(j\).

### Before you fit

1. Convert stock-tank rates to reservoir volume if formation volume factors
   matter on your field.
2. Align allocation: a well that is shut in should be a zero column, not a
   missing column.
3. Plot field injection versus field production. A large, persistent surplus of
   production often means aquifer support; add a pseudo-injector or use the
   aquifer helpers in {doc}`companion-models`.
4. Keep well names in a list that matches the columns. You will need those
   names for geometry initialization and for reading the gain table.

```python
import pandas as pd

producer_ids = ["P1", "P2", "P3", "P4"]
injector_ids = ["I1", "I2", "I3", "I4", "I5"]
connectivity = pd.DataFrame(crm.gains, index=producer_ids, columns=injector_ids)
```

## Choosing the CRM variant

Three knobs matter: `primary`, `tau_selection`, and `constraints`.

### Primary production

Set `primary=True` when the field still has depletion drive, or when rates
would fall if injection stopped. That is the usual choice.

Set `primary=False` when the history is clearly injection-dominated from the
first timestep (rates rise with fill-up and there is no decline to explain).

### Time constants

| `tau_selection` | Literature name | \(\tau\) shape | Use when |
|---|---|---|---|
| `"per-pair"` | CRM-IP | `(n_producers, n_injectors)` | Default. Enough history, injectors not in phase |
| `"per-producer"` | CRM-P | `(n_producers,)` | Short history or many injectors; faster, fewer parameters |

CRM-T (one tank for the whole field) is the nuclear option: sum all producers
into one column and all injectors into one column, then fit. It matches the
field total and tells you nothing about well pairs. See {doc}`choosing-crm`.

### Gain constraints

| `constraints` | What it enforces | Use when |
|---|---|---|
| `"up-to one"` | Each injector’s gains across producers sum to \(\le 1\) | Default. Some injection can be lost to aquifer or unswept rock |
| `"sum-to-one"` | Each producer’s gains across injectors sum to 1 | You believe every barrel a producer makes is explained by the injectors in the model |
| `"positive"` | Gains \(\ge 0\), no sum | Exploratory fits; easiest to over-allocate injection |
| `"sum-to-one injector"` | Not implemented | — |

`"up-to one"` is the constraint that matches injector material balance and is
the one to start with. `"sum-to-one"` is a producer-side partition; it is a
different statement.

## History matching: the PINN

`fit()` trains per-producer rate networks \(t \mapsto \hat q_j(t)\) plus a
parameter head for \(f_{ij}\) and \(\tau\). When you pass pressure,
`CrmCompensated` also trains \(t \mapsto \hat p_{\mathrm{wf},j}(t)\) so
\(\mathrm{d}p/\mathrm{d}t\) comes from automatic differentiation.

The loss is data MSE + physics residual + (optional) pressure MSE + a soft
constraint penalty. After training, identified parameters are copied to NumPy
on the CPU and `predict()` uses the Rust CRM kernels.

### Training knobs

Default `epochs` is 2000. Tests use 3; a real field usually needs a few hundred
to a few thousand.

| Argument | Default | Role |
|---|---|---|
| `epochs` or `options={"maxiter": N}` | 2000 | Training steps |
| `lr` | `1e-3` | RMSprop / Adam step size. Parameter head uses `5 * lr` |
| `optimizer` | `"rmsprop"` | `"adam"` is the other choice |
| `hidden_layers`, `hidden_units` | 4, 20 | Size of each well’s time MLP |
| `data_weight` | 1 | Weight on rate MSE |
| `physics_weight` | 1 | Weight on CRM residual (scaled by \(q^2\)) |
| `pressure_weight` | 1 | Weight on BHP MSE (`CrmCompensated`) |
| `constraint_weight` | 1 | Extra penalty on gain/tau bounds |
| `collocation_multiplier` | 1 | Extra physics points between observations |
| `device` | `"cpu"` | `"cuda"` if available. MPS falls back to CPU (no float64 autograd) |
| `seed` | `0` when `random=False` | Reproducible network init |
| `num_cores` | 1 | PyTorch CPU threads |

```python
crm.fit(
    production,
    injection,
    time,
    epochs=800,
    lr=1e-3,
    optimizer="rmsprop",
    physics_weight=1.0,
    seed=0,
    num_cores=4,
)
```

`crm.loss_history_` is a list of dicts with `loss`, `data`, `physics`,
`pressure`, and `constraints`. A healthy run has data and physics losses
falling together. If physics stays huge while data goes to zero, the networks
are memorizing rates and ignoring the tank equation — raise `physics_weight`
or train longer.

### Initialization

| `initial_guess` | What you get |
|---|---|
| omitted, `random=False` | Uniform gains (normalized) |
| omitted, `random=True` | Random positive gains |
| `"uniform"` / `"random"` | Same as the flag, as a string |
| `"geometry"` | MPI-based \(-\Lambda\) (requires well map) |
| list of per-producer `x0` vectors | Exact numeric start (legacy) |

Geometry is the high-value option when you have well coordinates. It is
described in the next section.

Several random restarts (`random=True` with different `seed` values) are still
useful on noisy fields. Keep the run with the lowest held-out RMSE, not the
lowest training loss.

## Geometry-based connectivity

CRM has many \(f_{ij}\). A homogeneous rectangular multiwell productivity
index (Kaviani and Valkó, 2010) is a strong first guess: nearby injectors start
with larger gains, distant ones start near zero, and training only has to
correct the map.

`calc_gains_homogeneous` returns \(\Lambda\), which is negative. CRM wants
positive gains, so the library uses \(f_{ij}^{(0)} = -\Lambda_{ij}\).

```python
import pandas as pd
from pywaterflood import CRM, translate_locations

wells = pd.DataFrame({
    "easting":  [...],   # one row per well
    "northing": [...],
    "kind":     [...],   # "Producer" or "Injector"
}, index=["P1", "P2", "I1", "I2"])

locations = translate_locations(wells, "easting", "northing", "kind")

crm = CRM(primary=True, tau_selection="per-pair", constraints="up-to one")
crm.fit(
    production,
    injection,
    time,
    initial_guess="geometry",
    locations=locations,
    x_e=reservoir_length,   # same units as X after translate
    y_e=reservoir_width,
    producer_ids=["P1", "P2"],          # production column order
    injector_ids=["I1", "I2"],          # injection column order
    epochs=800,
    seed=0,
)
```

### Geometry rules

- `locations` must have columns `X`, `Y`, `Type` with values `"Producer"` and
  `"Injector"`. `translate_locations` builds that table and shifts the origin
  to (0, 0).
- `x_e` and `y_e` are the rectangle extents. They must be finite and positive,
  and large enough that every well sits inside the rectangle.
- `producer_ids` / `injector_ids` must match production and injection column
  order. If you omit them, the order is the order those wells appear in
  `locations`, which can differ from a sorted index.
- Missing `locations`, `x_e`, or `y_e` raises. There is no silent fallback to
  uniform gains.
- Connectivity stays **trainable**. Geometry is a start, not a freeze.
- MPI assumes a roughly rectangular, homogeneous reservoir. Rotate your map
  first if the bounding box is skewed.

`geometry_crm_gains(...)` returns the same `(n_producers, n_injectors)` array
if you want to inspect \(-\Lambda\) before fitting.

## Pressure-compensated CRM

When producer BHP is known (or even a smooth trend), use `CrmCompensated`.
The extra term stops a workover or choke change from being explained as a
new injector connection.

```python
from pywaterflood import CrmCompensated

crm = CrmCompensated(primary=True, tau_selection="per-pair", constraints="up-to one")
crm.fit(production, pressure, injection, time, epochs=800, seed=0)
q_hat = crm.predict()
```

`pressure` has the same shape as `production`. Geometry kwargs work the same
way as on `CRM`.

## Reading the fitted model

After `fit()`:

| Attribute | Shape | Meaning |
|---|---|---|
| `crm.gains` | `(n_producers, n_injectors)` | \(f_{ij}\) |
| `crm.tau` | pair or producer shape | Time constants |
| `crm.gains_producer`, `crm.tau_producer` | `(n_producers,)` | Primary decline |
| `crm.productivity_index` | `(n_producers,)` | \(J_j\) (PINN) |
| `crm.loss_history_` | list | Per-epoch losses |

A useful gain is large, stable across random seeds, and belongs to a
geologically plausible pair (same pattern, short distance, or a known thief
zone). A gain that flips sign or jumps when you change `seed` is not a
decision you should act on.

```python
import pandas as pd

gains = pd.DataFrame(crm.gains, index=producer_ids, columns=injector_ids)
print(gains.round(3))
print("injector allocation (should be <= 1):", gains.sum(axis=0).round(3))
```

`residual()` is `production - predict()`. RMSE by producer is the first
quality check; a well with much higher RMSE than the others is often an
allocation error or a missing aquifer/injector.

## Forecasting a new injection plan

`predict()` is the closed-form CRM. Pass a future injection table with a
matching `time` vector. For primary decline it still needs the original
production (it uses the first row).

```python
# history
q_hist = crm.predict()

# plan: hold last injection for another 12 steps
dt = time[1] - time[0]
time_future = time[-1] + dt * np.arange(1, 13)
injection_future = np.repeat(injection[[-1], :], 12, axis=0)

q_plan = crm.predict(
    injection_future,
    time_future,
    production=production,  # first row used for primary
)
```

You can also evaluate a known parameter set without fitting, by passing a
`connections` dict (`gains`, `tau`, and if needed `gains_producer`,
`tau_producer`).

## Saving results

```python
crm.to_excel("crm_history_match.xlsx")
crm.to_pickle("crm_history_match.pkl")
```

Excel gets sheets for gains, taus, and primary production. Pickle stores the
whole object, including the PINN, so you can `predict()` later.

## A high-value workflow

This is the sequence that uses the library as far as it will go on a typical
pattern flood.

1. **Clean rates** into `(n_time, n_well)` arrays, reservoir barrels, no gaps.
2. **Material balance.** If production \(\gg\) injection, plan an aquifer
   pseudo-well ({doc}`companion-models`).
3. **Geometry seed** if you have a well map:
   `initial_guess="geometry"` with `producer_ids` in column order.
4. **Fit CRM-IP** (`per-pair`, `up-to one`, `primary=True`) for several hundred
   epochs with a fixed `seed`.
5. **Repeat** with `random=True` or a geometry-free uniform start. Keep the
   model whose held-out months match best.
6. **If BHP exists**, refit with `CrmCompensated` and compare residuals.
7. **If CRM-IP overfits** (great training match, wild gains, poor next-year
   match), drop to `tau_selection="per-producer"`.
8. **Forecast** with `predict(new_injection, new_time)`.
9. **Export** gains as a table for surveillance: which injector is actually
   supporting which producer.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `ValueError: cannot have NaNs` / `cannot be negative` | Empty months, shut-in coded as −999 | Fill shut-in with 0, interpolate holes |
| `same number of timesteps` | Misaligned CSV | Trim to the common calendar |
| Geometry `ValueError` listing `locations` / `x_e` / `y_e` | Incomplete map | Pass all three; use `translate_locations` |
| Geometry well-ID error | Labels ≠ column order | Set `producer_ids` / `injector_ids` explicitly |
| Gains almost uniform after a short fit | Too few epochs | Use hundreds of epochs; geometry init helps |
| Gains hit the cap of 1 everywhere | `"positive"` with no sum | Switch to `"up-to one"` |
| Good match, nonsense pairs | Too many parameters | `per-producer`, or geometry seed, or fewer wells |
| Training is slow on Apple GPU | MPS cannot run float64 autograd | Leave `device="cpu"` (automatic fallback) |
| `predict()` disagrees with a hand-built `connections` | `gains` oriented `(n_inj, n_prod)` | Use `(n_producers, n_injectors)` |

## See also

- {doc}`7-minutes-to-pywaterflood` — first plot of history match vs rates
- {doc}`choosing-crm` — CRM-IP vs CRM-P vs CRM-T on the sample field
- {doc}`companion-models` — MPI, Buckley-Leverett, aquifer
- {doc}`multiwell-productivity-index` — map-only connectivity example
