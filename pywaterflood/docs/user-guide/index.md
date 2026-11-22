# User Guide

Pywaterflood turns injection and production history into well-pair connectivity,
time constants, and usable forecasts. It also has two planning tools: a geometric
first guess for connectivity, and Buckley-Leverett / aquifer helpers for
displacement and influx.

This guide is the practical path through the library. Worked notebooks sit at
the end of each section.

## How the pieces fit

Think of a waterflood as a set of tanks linked by pipes.

- Each producer is a tank. Fluid leaves it as oil and water production.
- Each injector is a source that feeds several tanks.
- The pipe from injector \(i\) to producer \(j\) has a strength \(f_{ij}\)
  (how much of that injection supports that producer) and a lag \(\tau_{ij}\)
  (how slowly the support arrives).
- If the producer still has depletion drive, a primary-production term decays
  from the first observed rate.
- If you measure bottomhole pressure, `CrmCompensated` adds a pressure term so
  a choke change is not mistaken for an injector effect.

History matching trains a Deep-CRM physics-informed network (Yewgat et al.,
ECMOR 2020). After `fit()`, `predict()` uses the identified \(f_{ij}\) and
\(\tau\) in the closed-form CRM superposition, so you can forecast a new
injection schedule without retraining.

::::{admonition} Default that works on most fields
:class: tip

```python
from pywaterflood import CRM

crm = CRM(primary=True, tau_selection="per-pair", constraints="up-to one")
crm.fit(production, injection, time)
q_hat = crm.predict()
```

That is CRM-IP: one time constant per injector–producer pair, gains between 0
and 1, and each injector’s gains summing to at most 1.
::::

## Choose a path

| If you have… | Start here |
|---|---|
| Rate history only | {doc}`crm-in-practice` |
| Rates plus producer BHP | {doc}`crm-in-practice` (`CrmCompensated`) |
| Well map, no rates yet | {doc}`companion-models` (MPI geometry) |
| Rates and a well map | Geometry-initialized CRM in {doc}`crm-in-practice` |
| Rock/fluid properties for a linear flood | {doc}`companion-models` (Buckley-Leverett) |
| Pressure support from an aquifer | {doc}`companion-models` (van Everdingen–Hurst) |
| A cluster or container job | {doc}`using-docker` |

## Reading order

1. {doc}`crm-in-practice` — data layout, model choice, PINN knobs, geometry
   seed, forecasting, and what “good” looks like.
2. {doc}`companion-models` — MPI, Buckley-Leverett, and aquifer influx, and
   how they feed CRM.
3. Worked notebooks below, in this order: first match → model choice → geometry
   seed → front-speed check → aquifer pseudo-injector.
4. {doc}`using-docker` if you need a repeatable batch run.

| Notebook | You will be able to |
|---|---|
| {doc}`7-minutes-to-pywaterflood` | Load rates, fit CRM-IP, read $f_{ij}$, check RMSE, forecast a plan, try BHP |
| {doc}`choosing-crm` | Choose primary / $\tau$ layout / constraints, read PINN losses, use `CrmCompensated` |
| {doc}`multiwell-productivity-index` | Build $-\Lambda$ and seed CRM with `initial_guess="geometry"` |
| {doc}`buckley-leverett` | Check a CRM pair against Buckley-Leverett travel time |
| {doc}`aquifer` | Fit van Everdingen–Hurst influx and add it as a CRM pseudo-injector |

```{toctree}
:maxdepth: 2
:caption: Practice

crm-in-practice.md
companion-models.md
```

```{toctree}
:maxdepth: 1
:caption: Worked examples

7-minutes-to-pywaterflood.ipynb
choosing-crm.ipynb
multiwell-productivity-index.ipynb
buckley-leverett.ipynb
aquifer.ipynb
using-docker.md
```
